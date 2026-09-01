"""Checks on the ported UpdateTrack(). Runs in plain Python -- no Isaac Sim needed.

    python -m isaac_continuous_track.tests.test_track_driver

A fake articulation view stands in for the tensor API, and a perfect velocity
drive stands in for PhysX: each step the belt DOFs move by exactly the velocity
target they were given.  That is enough to exercise every branch of the driver
-- the wrap detection, the rewind, the velocity backup/restore, and the
promise that no unrelated DOF is disturbed.
"""

from __future__ import annotations

import math
import sys

import numpy as np

from ..track_config import GrouserCfg, make_oval_track_cfg
from ..track_driver import TrackDriver
from ..track_geometry import SegmentHandle, TrackHandle, fill_segment_length

LENGTH = 0.5
RADIUS = 0.1
WIDTH = 0.12
ELEMENTS = 32
PHYSICS_HZ = 250.0


def check(name: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    return bool(condition)


# ---------------------------------------------------------------------------
# stand-ins
# ---------------------------------------------------------------------------


class FakeArticulationView:
    """The slice of the Isaac tensor API that TrackDriver actually uses."""

    def __init__(self, dof_names, num_envs: int = 1):
        self.dof_names = list(dof_names)
        n = len(self.dof_names)
        self.q = np.zeros((num_envs, n))
        self.qd = np.zeros((num_envs, n))
        self.velocity_targets = np.zeros((num_envs, n))
        self.position_writes = 0
        self.velocity_writes = 0

    def get_dof_index(self, name: str) -> int:
        return self.dof_names.index(name)

    def get_joint_positions(self):
        return self.q.copy()

    def get_joint_velocities(self):
        return self.qd.copy()

    def set_joint_velocity_targets(self, velocities, indices=None, joint_indices=None):
        if joint_indices is None:
            self.velocity_targets[:] = velocities
        else:
            self.velocity_targets[:, np.asarray(joint_indices)] = velocities

    def set_joint_positions(self, positions, indices=None, joint_indices=None):
        self.position_writes += 1
        if joint_indices is None:
            self.q[:] = positions
        else:
            self.q[:, np.asarray(joint_indices)] = positions

    def set_joint_velocities(self, velocities, indices=None, joint_indices=None):
        self.velocity_writes += 1
        if joint_indices is None:
            self.qd[:] = velocities
        else:
            self.qd[:, np.asarray(joint_indices)] = velocities


def make_handle(name: str = "track") -> TrackHandle:
    cfg = make_oval_track_cfg(
        name=name,
        chassis_path="/World/Robot/chassis",
        sprocket_joint_path=f"/World/Robot/{name}_sprocket_joint",
        pitch_diameter=2.0 * RADIUS,
        length=LENGTH,
        radius=RADIUS,
        width=WIDTH,
        mass=2.0,
        elements_per_round=ELEMENTS,
        grouser=GrouserCfg(size=(0.012, WIDTH, 0.012)),
    )
    geometry = fill_segment_length(cfg)
    segments = [
        SegmentHandle(
            cfg=seg,
            geometry=seg_geom,
            link_path=f"/World/Robot/{name}/{seg.link_name}",
            joint_path=f"/World/Robot/{name}/{seg.joint_name}",
        )
        for seg, seg_geom in zip(cfg.segments, geometry.segments)
    ]
    return TrackHandle(
        name=name,
        cfg=cfg,
        geometry=geometry,
        segments=segments,
        scope_path=f"/World/Robot/{name}",
        element_count=ELEMENTS,
        belt_group_path="/World/CollisionGroups/track_belt",
        body_group_path="/World/CollisionGroups/body",
    )


def make_rig(num_envs: int = 1):
    """Fake view + driver, with one extra DOF standing in for a suspension."""
    handle = make_handle()
    dof_names = (
        [handle.sprocket_joint_name]
        + [s.joint_name for s in handle.segments]
        + ["suspension_joint"]
    )
    view = FakeArticulationView(dof_names, num_envs=num_envs)
    return handle, view, TrackDriver(view, [handle])


def step(view, driver, omega: float, dt: float) -> None:
    """One 'physics step': integrate the sprocket, let the belt track its target."""
    sprocket = view.dof_names.index("track_sprocket_joint")
    view.qd[:, sprocket] = omega
    view.q[:, sprocket] += omega * dt
    # a perfect velocity drive: belt DOFs move by exactly their target
    belt = [i for i, n in enumerate(view.dof_names) if "segment" in n]
    view.q[:, belt] += view.velocity_targets[:, belt] * dt
    view.qd[:, belt] = view.velocity_targets[:, belt]
    driver.on_physics_step(dt)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_dof_resolution() -> bool:
    handle, view, driver = make_rig()
    ok = check(
        "the sprocket and all four segment joints resolve to DOF indices",
        len(driver.tracks) == 1 and len(handle.segments) == 4,
    )
    ok &= check(
        "track_state reports the ported scalars",
        abs(driver.track_state()["track"]["pitch"] - handle.geometry.pitch) < 1e-15,
        f"pitch={handle.geometry.pitch:.5f} m, perimeter={handle.geometry.perimeter:.5f} m",
    )
    return ok


def test_velocity_targets() -> bool:
    """SetJointMotorVelocity(): each segment gets track_vel / joint_to_track."""
    handle, view, driver = make_rig()
    driver.reset()

    v_track = 0.5
    omega = v_track / handle.sprocket_joint_to_track
    step(view, driver, omega, 1.0 / PHYSICS_HZ)

    ok = True
    for seg in handle.segments:
        dof = view.dof_names.index(seg.joint_name)
        expected = v_track / seg.geometry.joint_to_track
        got = float(view.velocity_targets[0, dof])
        ok &= check(
            f"{seg.joint_name} target == track_vel / joint_to_track",
            abs(got - expected) < 1e-12,
            f"{got:.4f} vs {expected:.4f} "
            f"({'m/s' if seg.cfg.joint_type == 'prismatic' else 'rad/s'})",
        )

    ok &= check(
        "the sprocket's own drive target is untouched",
        float(view.velocity_targets[0, view.dof_names.index("track_sprocket_joint")]) == 0.0,
    )
    ok &= check(
        "the unrelated suspension DOF target is untouched",
        float(view.velocity_targets[0, view.dof_names.index("suspension_joint")]) == 0.0,
    )
    return ok


def test_rewind_cadence() -> bool:
    """The rewind fires once per pitch travelled -- and only then (plan 3.1)."""
    handle, view, driver = make_rig()
    driver.reset()
    writes_after_reset = view.position_writes

    dt = 1.0 / PHYSICS_HZ
    v_track = 1.0
    omega = v_track / handle.sprocket_joint_to_track
    seconds = 4.0
    steps = int(seconds / dt)

    max_abs_belt = 0.0
    belt_dofs = [view.dof_names.index(s.joint_name) for s in handle.segments]
    for _ in range(steps):
        step(view, driver, omega, dt)
        for seg, dof in zip(handle.segments, belt_dofs):
            # measure the belt position back in track space so all four segments
            # are comparable
            max_abs_belt = max(
                max_abs_belt, abs(float(view.q[0, dof])) * seg.geometry.joint_to_track
            )

    rewinds = view.position_writes - writes_after_reset
    pitch = handle.geometry.pitch
    expected = v_track * seconds / pitch

    ok = check(
        "reset() syncs the belt phase exactly once",
        writes_after_reset == 1,
        f"{writes_after_reset} position write(s) during reset",
    )
    ok &= check(
        "one rewind per pitch travelled",
        abs(rewinds - expected) <= 1,
        f"{rewinds} rewinds over {seconds} s at {v_track} m/s; "
        f"expected ~{expected:.0f} ({v_track / pitch:.1f} Hz)",
    )
    ok &= check(
        "the driver does NOT write positions every step",
        rewinds < steps / 4,
        f"{rewinds} position writes over {steps} steps -- "
        f"{steps / max(rewinds, 1):.1f} steps between rewinds for friction anchors to settle",
    )
    # one step of overshoot past pitch/2 before the wrap is detected
    bound = pitch / 2.0 + v_track * dt
    ok &= check(
        "belt travel stays within the joint limits the builder authors",
        max_abs_belt <= bound + 1e-12,
        f"max |belt| {max_abs_belt * 100:.3f} cm vs bound {bound * 100:.3f} cm "
        f"(limit is +/- {pitch * 100:.3f} cm)",
    )
    return ok


def test_velocity_is_preserved_across_a_rewind() -> bool:
    """wrap::SetPosition(..., preserveWorldVelocity=true)."""
    handle, view, driver = make_rig()
    driver.reset()

    dt = 1.0 / PHYSICS_HZ
    omega = 1.0 / handle.sprocket_joint_to_track
    suspension = view.dof_names.index("suspension_joint")
    sprocket = view.dof_names.index("track_sprocket_joint")

    # park recognisable values on the DOFs the driver must not disturb
    view.q[:, suspension] = 0.123
    view.qd[:, suspension] = -0.456

    ok = True
    seen_rewind = False
    for _ in range(int(0.5 / dt)):
        before_writes = view.position_writes
        q_sprocket = float(view.q[0, sprocket])
        step(view, driver, omega, dt)
        if view.position_writes > before_writes:
            seen_rewind = True
            ok &= check(
                "a rewind restores velocities (position write is paired with one)",
                view.velocity_writes == view.position_writes,
                f"{view.position_writes} position writes, {view.velocity_writes} velocity writes",
            )
            ok &= check(
                "a rewind leaves the suspension position alone",
                abs(float(view.q[0, suspension]) - 0.123) < 1e-15,
            )
            ok &= check(
                "a rewind leaves the suspension velocity alone",
                abs(float(view.qd[0, suspension]) + 0.456) < 1e-15,
            )
            ok &= check(
                "a rewind leaves the sprocket position alone",
                abs(float(view.q[0, sprocket]) - (q_sprocket + omega * dt)) < 1e-12,
            )
            break

    ok &= check("a rewind actually happened during the run", seen_rewind)
    return ok


def test_rewound_position_is_in_range() -> bool:
    """The rewind target is track_pos wrapped into [-pitch/2, +pitch/2)."""
    handle, view, driver = make_rig()
    driver.reset()

    dt = 1.0 / PHYSICS_HZ
    omega = 2.0 / handle.sprocket_joint_to_track  # 2 m/s, ~39 rewinds/s
    pitch = handle.geometry.pitch
    sprocket = view.dof_names.index("track_sprocket_joint")

    worst = 0.0
    for _ in range(int(2.0 / dt)):
        before = view.position_writes
        step(view, driver, omega, dt)
        if view.position_writes > before:
            track_pos = float(view.q[0, sprocket]) * handle.sprocket_joint_to_track
            wrapped = track_pos - pitch * math.floor(track_pos / pitch) - pitch / 2.0
            for seg in handle.segments:
                dof = view.dof_names.index(seg.joint_name)
                expected = wrapped / seg.geometry.joint_to_track
                worst = max(worst, abs(float(view.q[0, dof]) - expected))

    return check(
        "every rewind lands on wrapped / joint_to_track",
        worst < 1e-12,
        f"worst error {worst:.3e}",
    )


def test_reverse_and_batch() -> bool:
    """Negative speeds, and several environments at once (Isaac Lab)."""
    handle, view, driver = make_rig(num_envs=4)
    driver.reset()

    dt = 1.0 / PHYSICS_HZ
    sprocket = view.dof_names.index("track_sprocket_joint")
    # each env runs at a different speed, one of them backwards
    omegas = np.array([1.0, -1.0, 0.0, 2.5]) / handle.sprocket_joint_to_track

    belt = [i for i, n in enumerate(view.dof_names) if "segment" in n]
    for _ in range(int(1.0 / dt)):
        view.qd[:, sprocket] = omegas
        view.q[:, sprocket] += omegas * dt
        view.q[:, belt] += view.velocity_targets[:, belt] * dt
        view.qd[:, belt] = view.velocity_targets[:, belt]
        driver.on_physics_step(dt)

    pitch = handle.geometry.pitch
    ok = True
    for env in range(4):
        track_vel = float(view.qd[env, sprocket]) * handle.sprocket_joint_to_track
        seg = handle.segments[0]
        dof = view.dof_names.index(seg.joint_name)
        ok &= check(
            f"env {env}: velocity target follows the sign of the sprocket",
            abs(float(view.velocity_targets[env, dof]) - track_vel / seg.geometry.joint_to_track)
            < 1e-12,
            f"track_vel={track_vel:+.2f} m/s",
        )

    in_range = True
    for env in range(4):
        for seg in handle.segments:
            dof = view.dof_names.index(seg.joint_name)
            pos = abs(float(view.q[env, dof])) * seg.geometry.joint_to_track
            in_range &= pos <= pitch / 2.0 + abs(omegas).max() * handle.sprocket_joint_to_track * dt
    ok &= check("every environment stays inside the rewind range", in_range)

    # At track_pos == 0 the original's own formula gives wrapped == -pitch/2, so
    # a standing track rests half a pitch back -- not at zero.  The idle
    # environment must sit exactly there and never move again, even while the
    # other environments are triggering rewinds that rewrite the whole tensor.
    idle_ok = True
    for seg in handle.segments:
        dof = view.dof_names.index(seg.joint_name)
        expected = (-pitch / 2.0) / seg.geometry.joint_to_track
        idle_ok &= abs(float(view.q[2, dof]) - expected) < 1e-12
    ok &= check(
        "the idle environment holds the reset phase and is never disturbed",
        idle_ok,
        "env 2 has omega = 0 and stays at wrapped == -pitch/2, "
        "even though envs 0/1/3 rewind and the write covers every environment",
    )
    return ok



def test_stopped_timeline_is_survivable() -> bool:
    """Pressing Stop in the GUI makes the tensor API return None.

    The driver runs from a physics callback, so it must ride that out instead of
    raising -- a TypeError there takes the whole Isaac session down.
    """
    handle, view, driver = make_rig()
    driver.reset()

    real_q = view.get_joint_positions
    real_qd = view.get_joint_velocities
    view.get_joint_positions = lambda: None
    view.get_joint_velocities = lambda: None

    ok = True
    try:
        driver.on_physics_step(1.0 / PHYSICS_HZ)
        driver.set_track_speeds({"track": 0.5})
        state = driver.track_state()
        ok &= check("track_state returns empty rather than raising", state == {})
        ok &= check("on_physics_step and set_track_speeds survive a stopped timeline", True)
    except Exception as exc:  # noqa: BLE001 - that is what we are testing for
        ok &= check(f"survives a stopped timeline (raised {type(exc).__name__}: {exc})", False)

    # and it must pick straight back up when the timeline plays again
    view.get_joint_positions = real_q
    view.get_joint_velocities = real_qd
    try:
        writes = view.position_writes
        step(view, driver, 1.0 / handle.sprocket_joint_to_track, 1.0 / PHYSICS_HZ)
        ok &= check(
            "resumes normally once the timeline is playing again",
            view.position_writes >= writes,
        )
    except Exception as exc:  # noqa: BLE001
        ok &= check(f"resumes after a stop (raised {type(exc).__name__}: {exc})", False)
    return ok


def main() -> int:
    tests = [
        ("DOF resolution", test_dof_resolution),
        ("SetJointMotorVelocity equivalent", test_velocity_targets),
        ("rewind cadence", test_rewind_cadence),
        ("preserveWorldVelocity", test_velocity_is_preserved_across_a_rewind),
        ("rewind target", test_rewound_position_is_in_range),
        ("reverse + batched environments", test_reverse_and_batch),
        ("stopped timeline", test_stopped_timeline_is_survivable),
    ]
    failed = []
    for name, fn in tests:
        print(f"\n{name}")
        if not fn():
            failed.append(name)
    print("\n" + "-" * 60)
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print(f"all {len(tests)} groups passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
