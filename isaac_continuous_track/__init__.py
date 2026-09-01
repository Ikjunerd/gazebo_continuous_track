"""Isaac Sim (PhysX 5) port of the ``gazebo_continuous_track`` plugin.

The track is not a chain of links.  Four segment links -- two straight runs and
two arcs -- hang off the chassis on prismatic / revolute joints, slide by at
most one grouser pitch, and are rewound to zero once they pass it.  Because the
grousers are evenly spaced, the shape before and after a rewind is identical, so
the track looks and behaves continuous.

Typical use::

    from isaac_continuous_track import build_track, make_oval_track_cfg, TrackDriver

    cfg = make_oval_track_cfg(
        name="left_track",
        chassis_path="/World/Robot/chassis",
        sprocket_joint_path="/World/Robot/left_sprocket_joint",
        pitch_diameter=0.2, length=0.5, radius=0.1, width=0.12,
        mass=2.0, elements_per_round=32,
        origin_pos=(0.0, 0.2, -0.2),
    )
    handle = build_track(stage, cfg, body_paths=["/World/Robot/chassis"])
    # ... start the sim, initialise the articulation view ...
    driver = TrackDriver(view, [handle])
    driver.reset()
    driver.register()

Only ``build_track`` needs ``pxr``; the configuration, the ported geometry, the
handles and the driver all import in plain Python, which is what lets ``tests/``
run outside Isaac Sim.
"""

from .math_utils import Pose, child_pose_offset, dist_point_to_line
from .track_config import (
    BoxGeom,
    ColliderCfg,
    CylinderGeom,
    DriveCfg,
    GrouserCfg,
    SegmentCfg,
    SprocketCfg,
    SurfaceCfg,
    TrackCfg,
    make_oval_track_cfg,
)
from .track_driver import TrackDriver
from .track_geometry import (
    GrouserPlacement,
    SegmentGeometry,
    SegmentHandle,
    TrackGeometry,
    TrackHandle,
    fill_segment_length,
    iter_grouser_placements,
)

__all__ = [
    "Pose",
    "child_pose_offset",
    "dist_point_to_line",
    "BoxGeom",
    "ColliderCfg",
    "CylinderGeom",
    "DriveCfg",
    "GrouserCfg",
    "SegmentCfg",
    "SprocketCfg",
    "SurfaceCfg",
    "TrackCfg",
    "make_oval_track_cfg",
    "GrouserPlacement",
    "SegmentGeometry",
    "SegmentHandle",
    "TrackGeometry",
    "TrackHandle",
    "fill_segment_length",
    "iter_grouser_placements",
    "TrackDriver",
    # resolved lazily, see __getattr__ -- this one needs pxr
    "build_track",
]

__version__ = "0.1.0"

_LAZY = {"build_track"}


def __getattr__(name: str):
    """Defer the USD-dependent symbols so the rest imports without ``pxr``."""
    if name in _LAZY:
        from . import track_builder

        return getattr(track_builder, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
