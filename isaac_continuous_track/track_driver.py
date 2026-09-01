"""Physics-step driver for continuous tracks.

Port of ``ContinuousTrack::UpdateTrack()`` (gazebo_continuous_track.hpp:392).

What is kept from the original
------------------------------
* ``track_pos`` / ``track_vel`` are read off the sprocket joint and scaled by
  the sprocket pitch radius.
* The belt phase is ``track_pos`` wrapped into ``[-pitch/2, +pitch/2)``; because
  the grousers are evenly spaced, rewinding by one pitch leaves the shape of the
  track unchanged.
* Each segment joint is commanded to ``track_vel / joint_to_track`` every step,
  which is what ``SetJointMotorVelocity()`` did through ODE's ``fmax`` / ``vel``.

What is deliberately different (plan section 3)
-----------------------------------------------
* The position is written only on the steps where the wrap index changes, not on
  every step.  ODE rebuilds contacts every step and tolerates teleporting; PhysX
  keeps contact patches and friction anchors between steps, so teleporting every
  step would keep invalidating the anchors and friction would never build up.
  Drift is corrected by the rewind itself.
* Variants are gone: with a single grouser shape there is nothing to swap, so
  the runtime collision-bit and visibility toggling disappears entirely.

The whole update is written in array form over the leading (environment) axis,
so it works unchanged for a single robot or for an Isaac Lab batch of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .isaac_compat import subscribe_physics_step
from .track_geometry import TrackHandle

__all__ = ["TrackDriver"]


# ---------------------------------------------------------------------------
# tiny numpy / torch shim so the same code runs on either backend
# ---------------------------------------------------------------------------


def _xp(array: Any):
    if type(array).__module__.split(".")[0] == "torch":
        import torch

        return torch
    return np


def _clone(array: Any):
    return array.clone() if hasattr(array, "clone") else array.copy()


def _stack_last(xp, columns: Sequence[Any]):
    if xp.__name__ == "torch":
        return xp.stack(list(columns), dim=-1)
    return xp.stack(list(columns), axis=-1)


def _clip(xp, array, low: float, high: float):
    if xp.__name__ == "torch":
        return xp.clamp(array, low, high)
    return xp.clip(array, low, high)


def _all_true(xp, reference):
    if xp.__name__ == "torch":
        return xp.ones_like(reference, dtype=xp.bool)
    return np.ones_like(reference, dtype=bool)


def _dof_index(view: Any, name: str) -> int:
    """Resolve a DOF index from the joint prim name, across Isaac Sim versions."""
    getter = getattr(view, "get_dof_index", None)
    if getter is not None:
        try:
            return int(getter(name))
        except Exception:  # noqa: BLE001 - version dependent, fall through
            pass
    names = getattr(view, "dof_names", None)
    if names is not None and name in list(names):
        return list(names).index(name)
    raise KeyError(
        f"joint {name!r} is not a DOF of this articulation; known DOFs: {list(names or [])}"
    )


# ---------------------------------------------------------------------------
# per-track runtime state
# ---------------------------------------------------------------------------


@dataclass
class _TrackState:
    handle: TrackHandle
    sprocket_dof: int
    sprocket_radius: float  # Track::Sprocket::joint_to_track
    belt_dofs: np.ndarray  # int32, one per segment
    joint_to_track: np.ndarray  # float64, one per segment
    pitch: float  # Track::Belt: perimeter / elements_per_round
    max_track_speed: Optional[float]
    prev_wrap_index: Any = None

    @property
    def name(self) -> str:
        return self.handle.name


class TrackDriver:
    """Drives one or more tracks belonging to a single articulation view.

    Parameters
    ----------
    view:
        An ``isaacsim.core.prims.Articulation`` (or legacy ``ArticulationView``)
        already initialised on the robot that owns the tracks.
    handles:
        The :class:`~isaac_continuous_track.track_builder.TrackHandle` objects
        returned by ``build_track()``.
    """

    def __init__(self, view: Any, handles: Sequence[TrackHandle]) -> None:
        if not handles:
            raise ValueError("TrackDriver needs at least one track handle")
        self._view = view
        self._subscription: Any = None
        # Diagnostic switch: with the rewind off the belts still run on their
        # velocity drives, but nothing is ever teleported.  If the vehicle keeps
        # moving at belt speed with this off, contact is doing the work; if it
        # stops, the rewind was pushing it.
        self.rewind_enabled = True
        # Set to a callable to inspect what a rewind does to the articulation
        # root; it is handed (stage, "before"/"after").  Used to check whether
        # set_joint_positions() disturbs the base, which would silently cancel
        # any yaw the tracks manage to build up.
        self.on_rewind_probe = None
        self._tracks: List[_TrackState] = []

        for handle in handles:
            segments = handle.segments
            self._tracks.append(
                _TrackState(
                    handle=handle,
                    sprocket_dof=_dof_index(view, handle.sprocket_joint_name),
                    sprocket_radius=handle.sprocket_joint_to_track,
                    belt_dofs=np.array(
                        [_dof_index(view, s.joint_name) for s in segments], dtype=np.int32
                    ),
                    joint_to_track=np.array(
                        [s.geometry.joint_to_track for s in segments], dtype=np.float64
                    ),
                    pitch=handle.geometry.pitch,
                    max_track_speed=handle.cfg.drive.max_track_speed,
                )
            )

    # -- properties ---------------------------------------------------------

    @property
    def tracks(self) -> List[TrackHandle]:
        return [t.handle for t in self._tracks]

    def track_state(self) -> Dict[str, Dict[str, Any]]:
        """``{track_name: {"track_pos": ..., "track_vel": ...}}``, for plan section 6."""
        q = self._view.get_joint_positions()
        qd = self._view.get_joint_velocities()
        out: Dict[str, Dict[str, Any]] = {}
        if q is None or qd is None:  # timeline not playing; see _apply()
            return out
        for track in self._tracks:
            out[track.name] = {
                "track_pos": q[:, track.sprocket_dof] * track.sprocket_radius,
                "track_vel": qd[:, track.sprocket_dof] * track.sprocket_radius,
                "pitch": track.pitch,
                "perimeter": track.handle.geometry.perimeter,
            }
        return out

    # -- lifecycle ----------------------------------------------------------

    def register(self) -> Any:
        """Subscribe to physics steps.

        Counterpart of ``event::Events::ConnectWorldUpdateBegin()``.  Isaac Sim
        users driving the sim through ``World`` may prefer
        ``world.add_physics_callback(name, driver.on_physics_step)`` instead.
        """
        if self._subscription is None:
            self._subscription = subscribe_physics_step(self.on_physics_step)
        return self._subscription

    def unregister(self) -> None:
        if self._subscription is not None:
            try:
                self._subscription.unsubscribe()
            except AttributeError:
                pass
            self._subscription = None

    def reset(self) -> None:
        """Put the belts in phase with the sprockets and arm the wrap detector.

        Called once after the articulation is initialised, so that the first
        physics step does not see a bogus wrap crossing (plan section 4.3).
        """
        for track in self._tracks:
            track.prev_wrap_index = None
        self._apply(rewind_all=True, set_velocity_targets=False)

    # -- the update ---------------------------------------------------------

    def on_physics_step(self, dt: float) -> None:  # noqa: ARG002 - signature fixed by omni.physx
        self._apply(rewind_all=False, set_velocity_targets=True)

    def _apply(self, rewind_all: bool, set_velocity_targets: bool) -> None:
        view = self._view
        q = view.get_joint_positions()
        qd = view.get_joint_velocities()
        if q is None or qd is None:
            # The tensor API returns None whenever the timeline is not playing --
            # paused or stopped from the GUI, or during teardown.  The driver is
            # called from a physics callback, so it has to ride that out rather
            # than take the process down.
            return
        xp = _xp(q)

        q_new = None
        qd_backup = None

        for track in self._tracks:
            radius = track.sprocket_radius
            track_pos = q[:, track.sprocket_dof] * radius
            track_vel = qd[:, track.sprocket_dof] * radius
            if track.max_track_speed is not None:
                # Joint::GetVelocityLimit() clamp of SetJointMotorVelocity()
                track_vel = _clip(xp, track_vel, -track.max_track_speed, track.max_track_speed)

            pitch = track.pitch
            wrap_index = xp.floor(track_pos / pitch)
            # track pos normalised into [-pitch/2, +pitch/2)
            wrapped = track_pos - pitch * wrap_index - pitch / 2.0

            # 1) velocity target on every step -- the ODE joint motor
            if set_velocity_targets:
                targets = _stack_last(
                    xp,
                    [track_vel / float(scale) for scale in track.joint_to_track],
                )
                view.set_joint_velocity_targets(targets, joint_indices=track.belt_dofs)

            # 2) position rewind only when the wrap index changes
            if rewind_all or track.prev_wrap_index is None:
                crossed = _all_true(xp, wrap_index)
            else:
                crossed = wrap_index != track.prev_wrap_index
            track.prev_wrap_index = wrap_index

            if not bool(crossed.any()):
                continue

            if q_new is None:
                q_new = _clone(q)
                # preserveWorldVelocity=true of wrap::SetPosition(): applying a
                # position may or may not zero velocities depending on the Isaac
                # version, so back them up and put them back afterwards
                qd_backup = _clone(qd)
            for dof, scale in zip(track.belt_dofs, track.joint_to_track):
                q_new[crossed, int(dof)] = wrapped[crossed] / float(scale)

        if q_new is not None and self.rewind_enabled:
            if self.on_rewind_probe is not None:
                self.on_rewind_probe("before")
            # write the whole DOF vector: a partial write would risk clobbering
            # the sprocket and suspension state (plan section 4.3)
            view.set_joint_positions(q_new)
            view.set_joint_velocities(qd_backup)
            if self.on_rewind_probe is not None:
                self.on_rewind_probe("after")

    # -- convenience --------------------------------------------------------

    def set_track_speeds(self, speeds: Dict[str, float]) -> None:
        """Command sprocket joints from desired track speeds in m/s.

        Purely a convenience for examples and tests: the sprocket is the user's
        actuator, the driver only reads it.
        """
        view = self._view
        q = view.get_joint_positions()
        if q is None:  # timeline not playing; see _apply()
            return
        xp = _xp(q)
        dofs: List[int] = []
        columns: List[Any] = []
        for track in self._tracks:
            if track.name not in speeds:
                continue
            omega = float(speeds[track.name]) / track.sprocket_radius
            dofs.append(track.sprocket_dof)
            columns.append(xp.zeros_like(q[:, track.sprocket_dof]) + omega)
        if not dofs:
            return
        view.set_joint_velocity_targets(
            _stack_last(xp, columns), joint_indices=np.array(dofs, dtype=np.int32)
        )
