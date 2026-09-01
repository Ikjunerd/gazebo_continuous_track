"""The Talon's measured geometry, and the track configs built from it.

Import-safe: this module never boots Kit, so the rigging script, the runtime
driver and the tests can all share it.  (Anything that calls ``SimulationApp``
at import time can only ever be the process entry point -- importing two of
them crashes the process with an access violation.)

All numbers were measured off ``talon_p_v4_0c.usd`` and are expressed in the
``vflip_link`` frame of one flipper: X forward along the flipper, Z up, Y the
wheel axis.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from ..track_config import (
    DriveCfg,
    GrouserCfg,
    SurfaceCfg,
    TrackCfg,
    make_two_pulley_track_cfg,
)
from ..track_geometry import SegmentHandle, TrackHandle, fill_segment_length

__all__ = [
    "BODY",
    "FLIPPERS",
    "FLIPPER_FRAG",
    "DRIVE_R",
    "IDLER_R",
    "IDLER_OFFSET",
    "BELT_WIDTH",
    "ROLLER_COUNT",
    "PLANE_Y",
    "WHEEL_SIGN",
    "FLIPPER_SIGN",
    "SIDE",
    "TRACK_WIDTH",
    "flipper_paths",
    "make_flipper_track_cfg",
    "make_flipper_handle",
]

BODY = "TALON_Assy_Body_0625_fin"

# (tag, subassembly path fragment).  The asset mixes two mirrored subassemblies,
# which is why the sign tables below are not uniform.
FLIPPERS: Tuple[Tuple[str, str], ...] = (
    ("FL", "FL_virtual/virtual_Flipper_1_0625_TEST"),
    ("FR", "FR_virtual/virtual_Flipper_2"),
    ("RL", "RL_virtual/virtual_Flipper_2"),
    ("RR", "RR_virtual/virtual_Flipper_1_0625_TEST"),
)
FLIPPER_FRAG = dict(FLIPPERS)

DRIVE_R = 0.17903  # large_wheel collider radius
IDLER_OFFSET = (0.43845, -0.070001)  # small_wheel_joint localPos0, x and z
BELT_WIDTH = 0.065  # wheel width
ROLLER_COUNT = 13

# The 13 rollers all sit at z = -0.11349 with radius 0.06553, so the belt's
# bottom run is dead flat on their underside (z = -0.17902).  A horizontal
# tangent to both pulleys needs R - r = -dz, which *derives* the idler radius.
# That beats measuring it: the idler mesh bbox gives 0.10915 across x but
# 0.10885 across z, and either value leaves the bottom run tilted and 0.2 mm off
# the roller line.
IDLER_R = DRIVE_R + IDLER_OFFSET[1]

# y of the wheel plane inside vflip_link, per flipper (large_wheel_joint localPos0)
PLANE_Y = {"FL": 0.0425, "FR": -0.0425, "RL": -0.0425, "RR": 0.0425}

# Which side of the vehicle each flipper drives.
SIDE = {"FL": "L", "FR": "R", "RL": "L", "RR": "R"}

# Sign that makes all four sprockets push the vehicle the same way.  The rear
# subassemblies are mirrored (their local +X points backwards), so they turn the
# other way.  Measured in simulation by the previous conveyor driver.
WHEEL_SIGN = {"FL": +1.0, "FR": +1.0, "RL": -1.0, "RR": -1.0}

# Sign that makes "+ raises the flipper tip" hold for all four arms.
FLIPPER_SIGN = {"FL": -1.0, "FR": +1.0, "RL": +1.0, "RR": -1.0}

# Left/right track centre separation (m), base_link local y +0.2403 / -0.2423.
TRACK_WIDTH = 0.4826


def flipper_paths(root_path: str, tag: str) -> dict:
    """Absolute prim paths for one flipper under a referenced Talon."""
    root = root_path.rstrip("/")
    flip = f"{root}/{BODY}/{FLIPPER_FRAG[tag]}"
    return {
        "flip": flip,
        "vflip": f"{flip}/vflip_link",
        # names as they are in the SOURCE asset
        "drive_joint": f"{flip}/joints/large_wheel_joint",
        "idler_joint": f"{flip}/joints/small_wheel_joint",
        # names after rigging, made unique so the merged articulation can tell
        # the four flippers apart (see rig_talon_track._rename_prim)
        "drive_joint_rigged": f"{flip}/joints/{tag}_large_wheel_joint",
        "idler_joint_rigged": f"{flip}/joints/{tag}_small_wheel_joint",
        "large_wheel": f"{flip}/large_wheel_link",
        "small_wheel": f"{flip}/small_wheel_link",
        "arm_joint": f"{root}/{BODY}/joints/{tag}_link_joint",
        # the pivot arm body; its visual spans the whole flipper
        "arm_link": f"{root}/{BODY}/{tag}_link",
        "scope": f"{flip}/{tag}_track",
    }


def make_flipper_track_cfg(
    root_path: str,
    tag: str,
    elements_per_round: int = 36,
    grouser_height: float = 0.014,
    drive_arc_margin: float = 0.01,
    idler_arc_margin: float = 0.01,
    belt_mass: float = 3.0,
    drive: Optional[DriveCfg] = None,
    surface: Optional[SurfaceCfg] = None,
) -> TrackCfg:
    """The continuous-track config for one flipper.

    The belt is tapered: a big drive sprocket at the pivot and a smaller idler at
    the tip, so it is a trapezoid rather than an oval.
    """
    p = flipper_paths(root_path, tag)
    return make_two_pulley_track_cfg(
        name=f"{tag}_track",
        chassis_path=p["vflip"],
        sprocket_joint_path=p["drive_joint_rigged"],
        drive_radius=DRIVE_R,
        idler_radius=IDLER_R,
        idler_offset=IDLER_OFFSET,
        width=BELT_WIDTH,
        mass=belt_mass,
        elements_per_round=elements_per_round,
        drive_arc_margin=drive_arc_margin,
        idler_arc_margin=idler_arc_margin,
        grouser=GrouserCfg(size=(0.012, BELT_WIDTH, grouser_height)),
        origin_pos=(0.0, PLANE_Y[tag], 0.0),
        # ~800 N of tractive force per flipper is far more than the 83 kg machine
        # needs; the damping holds the belt within about 1 cm/s of the command
        drive=drive if drive is not None else DriveCfg(damping=4.0e4, max_force=800.0),
        surface=surface
        if surface is not None
        else SurfaceCfg(
            static_friction=0.9, dynamic_friction=0.8, contact_offset=0.015, rest_offset=0.0
        ),
        scope_path=p["scope"],
    )


def make_flipper_handle(root_path: str, tag: str, **kwargs) -> TrackHandle:
    """Rebuild the handle for an already-rigged flipper, authoring nothing.

    ``build_track()`` returns a handle as a side effect of creating the prims.
    At runtime the prims already exist, so the driver just needs the same handle
    reconstructed from the same config -- no stage access required.
    """
    cfg = make_flipper_track_cfg(root_path, tag, **kwargs)
    geometry = fill_segment_length(cfg)
    scope = flipper_paths(root_path, tag)["scope"]
    segments = [
        SegmentHandle(
            cfg=seg,
            geometry=seg_geom,
            link_path=f"{scope}/{seg.link_name}",
            joint_path=f"{scope}/{seg.joint_name}",
        )
        for seg, seg_geom in zip(cfg.segments, geometry.segments)
    ]
    return TrackHandle(
        name=cfg.name,
        cfg=cfg,
        geometry=geometry,
        segments=segments,
        scope_path=scope,
        element_count=cfg.elements_per_round,
        belt_group_path="",
        body_group_path="",
    )
