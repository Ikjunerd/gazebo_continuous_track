"""Single place where Isaac Sim / Omniverse imports happen.

Plan section 8 lists "Python API paths move between Isaac Sim versions
(``omni.isaac.core`` <-> ``isaacsim.core``)" as a risk and asks for the imports
to be isolated in one module.  Everything version dependent lives here; the
rest of the package imports from this file only.

``pxr`` (Usd, UsdPhysics, PhysxSchema, ...) has been stable across versions and
is imported directly where needed.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "get_articulation_cls",
    "get_physx_interface",
    "get_current_stage",
    "subscribe_physics_step",
]


def get_articulation_cls() -> type:
    """Return the batched articulation view class.

    Isaac Sim >= 4.5 renamed ``ArticulationView`` to ``isaacsim.core.prims.Articulation``.
    Both expose the tensor API the driver needs (``get_joint_positions``,
    ``set_joint_velocity_targets``, ``get_dof_index``, ...).
    """
    try:  # Isaac Sim >= 4.5
        from isaacsim.core.prims import Articulation  # type: ignore

        return Articulation
    except ImportError:
        pass
    try:  # Isaac Sim <= 4.2
        from omni.isaac.core.articulations import ArticulationView  # type: ignore

        return ArticulationView
    except ImportError as exc:  # pragma: no cover - depends on the runtime
        raise ImportError(
            "Neither isaacsim.core.prims.Articulation nor "
            "omni.isaac.core.articulations.ArticulationView is available. "
            "Run this inside Isaac Sim's python environment."
        ) from exc


def get_current_stage() -> Any:
    try:  # Isaac Sim >= 4.5
        from isaacsim.core.utils.stage import get_current_stage as _get  # type: ignore

        return _get()
    except ImportError:
        pass
    try:  # Isaac Sim <= 4.2
        from omni.isaac.core.utils.stage import get_current_stage as _get  # type: ignore

        return _get()
    except ImportError:
        pass
    import omni.usd  # type: ignore

    return omni.usd.get_context().get_stage()


def get_physx_interface() -> Any:
    from omni.physx import get_physx_interface as _get  # type: ignore

    return _get()


def subscribe_physics_step(callback) -> Any:
    """Register ``callback(dt)`` on every physics step.

    Returns the subscription handle; keep a reference to it or the callback is
    dropped.  This is the Isaac counterpart of Gazebo's
    ``event::Events::ConnectWorldUpdateBegin()``.
    """
    return get_physx_interface().subscribe_physics_step_events(callback)
