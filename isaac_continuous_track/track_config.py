"""Configuration dataclasses for a continuous track.

The plugin-facing part mirrors ``sdf/continuous_track_plugin.sdf`` one to one::

    <sprocket><joint>                   -> SprocketCfg.joint_path
    <sprocket><pitch_diameter>          -> SprocketCfg.pitch_diameter
    <trajectory><segment><joint>        -> SegmentCfg (the joint is built, not looked up)
    <trajectory><segment><end_position> -> SegmentCfg.end_position
    <pattern><elements_per_round>       -> TrackCfg.elements_per_round
    <pattern><element>                  -> TrackCfg.grouser (single element, plan 3.2)

The remaining fields carry what the ``make_track`` xacro macro used to provide
(link poses, masses, collider sizes), because in Isaac the prims are created by
:mod:`isaac_continuous_track.track_builder` instead of being read back from an
already-loaded model.

``make_oval_track_cfg()`` reproduces ``macros_track_gazebo.urdf.xacro`` exactly:
straight / arc / straight / arc, all four joints parented to the chassis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple, Union

Vec3 = Tuple[float, float, float]
JointType = Literal["prismatic", "revolute"]

__all__ = [
    "BoxGeom",
    "CylinderGeom",
    "ColliderCfg",
    "SurfaceCfg",
    "DriveCfg",
    "SprocketCfg",
    "SegmentCfg",
    "GrouserCfg",
    "TrackCfg",
    "make_oval_track_cfg",
    "two_pulley_geometry",
    "make_two_pulley_track_cfg",
]


# ---------------------------------------------------------------------------
# geometry primitives
# ---------------------------------------------------------------------------


@dataclass
class BoxGeom:
    size: Vec3  # full extents (x, y, z)


@dataclass
class CylinderGeom:
    radius: float
    length: float
    axis: Literal["X", "Y", "Z"] = "Z"


Geom = Union[BoxGeom, CylinderGeom]


@dataclass
class ColliderCfg:
    """A collider attached to a segment link, with its pose in the link frame."""

    geom: Geom
    pos: Vec3 = (0.0, 0.0, 0.0)
    rpy: Vec3 = (0.0, 0.0, 0.0)


@dataclass
class SurfaceCfg:
    """Contact properties.

    ``static_friction`` / ``dynamic_friction`` replace the SDF
    ``<surface><friction><ode><mu>`` (0.5 by default in the xacro macros).
    ``contact_offset`` / ``rest_offset`` replace ``<contact><ode><min_depth>``.
    Plan section 5 holds the tuning baseline.
    """

    static_friction: float = 0.5
    dynamic_friction: float = 0.5
    restitution: float = 0.0
    contact_offset: float = 0.02
    rest_offset: float = 0.0


@dataclass
class DriveCfg:
    """Segment joint drive.

    Replaces the ODE joint motor (``fmax`` / ``vel``) that
    ``SetJointMotorVelocity()`` programmed.  ``stiffness`` is 0 so the drive is
    pure velocity control, exactly like the ODE motor.

    All three gains are expressed **in track space**, so that one number
    describes the whole belt regardless of how its segments are split between
    prismatic and revolute joints:

    * ``damping``   -- N per (m/s) of track speed error
    * ``stiffness`` -- N per m of track position error (0 = pure velocity drive)
    * ``max_force`` -- N of tractive force, i.e. the ceiling on what the track
      can push against the ground.  Corresponds to ``Joint::GetEffortLimit()``.

    ``track_builder`` converts these to per-joint units with the segment's
    ``joint_to_track`` factor ``s``: a revolute segment sees a torque limit of
    ``max_force * s`` and a damping of ``damping * s**2``, which is exactly the
    gain that produces the same tangential force at the belt surface.

    ``max_track_speed`` is the counterpart of ``Joint::GetVelocityLimit()``, also
    expressed once along the track (m/s); ``None`` means unlimited, matching the
    "negative value means unlimited" rule of the original.
    """

    stiffness: float = 0.0
    damping: float = 1.0e5
    max_force: float = 1.0e6
    max_track_speed: Optional[float] = None


# ---------------------------------------------------------------------------
# plugin configuration
# ---------------------------------------------------------------------------


@dataclass
class SprocketCfg:
    """``<sprocket>``. The joint must already exist in the articulation."""

    joint_path: str
    pitch_diameter: float

    @property
    def joint_name(self) -> str:
        return self.joint_path.rsplit("/", 1)[-1]

    @property
    def joint_to_track(self) -> float:
        """``ComposeSprocket()``: scale from joint position to length along track."""
        return self.pitch_diameter / 2.0


@dataclass
class SegmentCfg:
    """``<trajectory><segment>`` plus the link/joint description the xacro provided.

    ``axis`` and ``joint_pos`` are expressed in the *child link* frame, which is
    what SDF means by ``<axis><use_parent_model_frame>0`` and ``<joint><pose>``.

    ``zero_pos`` / ``zero_rpy`` place the child link, at joint position 0,
    relative to the track origin frame (``TrackCfg.origin_pos`` / ``origin_rpy``).
    """

    name: str
    joint_type: JointType
    end_position: float

    axis: Vec3
    joint_pos: Vec3 = (0.0, 0.0, 0.0)

    zero_pos: Vec3 = (0.0, 0.0, 0.0)
    zero_rpy: Vec3 = (0.0, 0.0, 0.0)

    mass: float = 1.0
    inertia_diag: Vec3 = (1.0e-3, 1.0e-3, 1.0e-3)
    com_pos: Vec3 = (0.0, 0.0, 0.0)
    com_rpy: Vec3 = (0.0, 0.0, 0.0)

    # the belt body itself; plan 1.3 -- "do not forget the segment's own collider"
    body_collider: Optional[ColliderCfg] = None

    @property
    def joint_name(self) -> str:
        return f"{self.name}_joint"

    @property
    def link_name(self) -> str:
        return f"{self.name}_link"


@dataclass
class GrouserCfg:
    """``<pattern><element>``, reduced to a single collider (plan 3.2).

    ``pos`` defaults to half the height along +Z so the grouser sticks out of the
    belt surface: every segment link frame is authored on the belt surface with
    +Z pointing away from the track interior, so one expression covers all four
    segments.
    """

    size: Vec3 = (0.01, 0.1, 0.01)
    pos: Optional[Vec3] = None
    rpy: Vec3 = (0.0, 0.0, 0.0)
    visual: bool = True

    def resolved_pos(self) -> Vec3:
        if self.pos is not None:
            return self.pos
        return (0.0, 0.0, self.size[2] / 2.0)


@dataclass
class TrackCfg:
    """One continuous track == one ``<plugin>`` instance."""

    name: str
    chassis_path: str
    sprocket: SprocketCfg
    segments: List[SegmentCfg]
    elements_per_round: int
    grouser: GrouserCfg = field(default_factory=GrouserCfg)

    # pose of the track origin frame in the chassis frame (the xacro x/y/z/rpy args)
    origin_pos: Vec3 = (0.0, 0.0, 0.0)
    origin_rpy: Vec3 = (0.0, 0.0, 0.0)

    drive: DriveCfg = field(default_factory=DriveCfg)
    surface: SurfaceCfg = field(default_factory=SurfaceCfg)

    # prim scope the generated belt links go under; defaults to <chassis parent>/<name>
    scope_path: Optional[str] = None

    def validate(self) -> None:
        """Mirrors the GZ_ASSERTs in ``ContinuousTrackProperties``."""
        if self.sprocket.pitch_diameter <= 0.0:
            raise ValueError("[sprocket][pitch_diameter] must be a positive real number")
        if not self.segments:
            raise ValueError("[trajectory] needs at least one [segment]")
        for seg in self.segments:
            if seg.end_position <= 0.0:
                raise ValueError(
                    f"[trajectory][segment][end_position] of {seg.name!r} must be positive"
                )
            if seg.joint_type not in ("prismatic", "revolute"):
                raise ValueError(
                    "[trajectory][segment][joint] must be a rotational or translational joint"
                )
        if self.elements_per_round <= 0:
            raise ValueError("[pattern][elements_per_round] must be a positive integer")


# ---------------------------------------------------------------------------
# make_track: the standard 4-segment oval
# ---------------------------------------------------------------------------


def _box_inertia(mass: float, size: Vec3) -> Vec3:
    """``make_box_inertial`` from macros_common_gazebo.urdf.xacro."""
    x, y, z = size
    return (
        mass * (y * y + z * z) / 12.0,
        mass * (z * z + x * x) / 12.0,
        mass * (x * x + y * y) / 12.0,
    )


def _cylinder_inertia(mass: float, length: float, radius: float) -> Vec3:
    """``make_cylinder_inertial`` from macros_common_gazebo.urdf.xacro."""
    ixx = mass * (radius * radius / 4.0 + length * length / 12.0)
    return (ixx, ixx, mass * radius * radius / 2.0)


def make_oval_track_cfg(
    name: str,
    chassis_path: str,
    sprocket_joint_path: str,
    pitch_diameter: float,
    length: float,
    radius: float,
    width: float,
    mass: float,
    elements_per_round: int,
    belt_overlap: Optional[float] = None,
    grouser: Optional[GrouserCfg] = None,
    origin_pos: Vec3 = (0.0, 0.0, 0.0),
    origin_rpy: Vec3 = (0.0, 0.0, 0.0),
    drive: Optional[DriveCfg] = None,
    surface: Optional[SurfaceCfg] = None,
    scope_path: Optional[str] = None,
) -> TrackCfg:
    """Build the configuration the ``make_track`` xacro macro used to produce.

    Reproduces ``macros_track_gazebo.urdf.xacro`` together with
    ``populate_straight_segments`` / ``populate_arc_segments`` numerically::

        segment            joint         link zero pose (track frame)   end_position
        straight_segment0  prismatic X   (-length/2, 0,  radius)        length
        arc_segment0       revolute  Y   ( length/2, 0,  radius)        pi
        straight_segment1  prismatic X   ( length/2, 0, -radius) rpy=(0,pi,0)  length
        arc_segment1       revolute  Y   (-length/2, 0, -radius) rpy=(0,pi,0)  pi

    The traversal order is: top run forward, front wheel, bottom run backward,
    rear wheel.  Every link frame sits on the belt surface with +Z pointing
    outwards.
    """
    seg_mass = mass / 4.0
    # See make_two_pulley_track_cfg: a straight run authored at exactly its path
    # length opens a gap against the arc it just left, because every segment
    # slides by up to +/- pitch/2.  Pad the collider box, not the path.  Pass
    # belt_overlap=0.0 for the xacro's literal geometry.
    pitch = (2.0 * length + 2.0 * math.pi * radius) / elements_per_round
    overlap = pitch / 2.0 if belt_overlap is None else float(belt_overlap)
    straight_size: Vec3 = (length + 2.0 * overlap, width, radius)
    straight_inertia_size: Vec3 = (length, width, radius)
    straight_com: Vec3 = (length / 2.0, 0.0, -radius / 2.0)
    arc_com: Vec3 = (0.0, 0.0, -radius)
    arc_rpy: Vec3 = (math.pi / 2.0, 0.0, 0.0)

    def straight(index: int, pos: Vec3, rpy: Vec3) -> SegmentCfg:
        return SegmentCfg(
            name=f"{name}_straight_segment{index}",
            joint_type="prismatic",
            end_position=length,
            axis=(1.0, 0.0, 0.0),
            joint_pos=(0.0, 0.0, 0.0),
            zero_pos=pos,
            zero_rpy=rpy,
            mass=seg_mass,
            inertia_diag=_box_inertia(seg_mass, straight_inertia_size),
            com_pos=straight_com,
            body_collider=ColliderCfg(BoxGeom(straight_size), pos=straight_com),
        )

    def arc(index: int, pos: Vec3, rpy: Vec3) -> SegmentCfg:
        return SegmentCfg(
            name=f"{name}_arc_segment{index}",
            joint_type="revolute",
            end_position=math.pi,
            axis=(0.0, 1.0, 0.0),
            joint_pos=(0.0, 0.0, -radius),
            zero_pos=pos,
            zero_rpy=rpy,
            mass=seg_mass,
            inertia_diag=_cylinder_inertia(seg_mass, width, radius),
            com_pos=arc_com,
            com_rpy=arc_rpy,
            body_collider=ColliderCfg(
                CylinderGeom(radius=radius, length=width, axis="Z"), pos=arc_com, rpy=arc_rpy
            ),
        )

    flipped: Vec3 = (0.0, math.pi, 0.0)
    segments = [
        straight(0, (-length / 2.0, 0.0, radius), (0.0, 0.0, 0.0)),
        arc(0, (length / 2.0, 0.0, radius), (0.0, 0.0, 0.0)),
        straight(1, (length / 2.0, 0.0, -radius), flipped),
        arc(1, (-length / 2.0, 0.0, -radius), flipped),
    ]

    cfg = TrackCfg(
        name=name,
        chassis_path=chassis_path,
        sprocket=SprocketCfg(joint_path=sprocket_joint_path, pitch_diameter=pitch_diameter),
        segments=segments,
        elements_per_round=elements_per_round,
        grouser=grouser if grouser is not None else GrouserCfg(size=(0.01, width, 0.01)),
        origin_pos=origin_pos,
        origin_rpy=origin_rpy,
        drive=drive if drive is not None else DriveCfg(),
        surface=surface if surface is not None else SurfaceCfg(),
        scope_path=scope_path,
    )
    cfg.validate()
    return cfg


# ---------------------------------------------------------------------------
# make_two_pulley_track_cfg: the general tapered belt
# ---------------------------------------------------------------------------


def two_pulley_geometry(
    drive_radius: float, idler_radius: float, idler_offset: Tuple[float, float]
) -> dict:
    """Closed-form geometry of a belt wrapped around two pulleys of unequal radius.

    ``idler_offset`` is ``(dx, dz)`` of the idler centre relative to the drive
    centre, in the track plane (X forward, Z up, Y the wheel axis).

    For an open belt the wrap angles are ``pi +/- 2*alpha`` with
    ``sin(alpha) = (R - r) / d``; the straight runs are the external tangents,
    of length ``sqrt(d**2 - (R - r)**2)`` each.  Setting ``R == r`` degenerates
    to the symmetric oval that :func:`make_oval_track_cfg` builds.
    """
    R, r = float(drive_radius), float(idler_radius)
    dx, dz = float(idler_offset[0]), float(idler_offset[1])
    d = math.hypot(dx, dz)
    if d <= abs(R - r):
        raise ValueError(
            f"pulleys are too close: centre distance {d:.4f} must exceed "
            f"|R - r| = {abs(R - r):.4f}"
        )
    theta = math.atan2(dz, dx)  # tilt of the centre line
    alpha = math.asin((R - r) / d)
    tangent = math.sqrt(d * d - (R - r) ** 2)
    return {
        "R": R,
        "r": r,
        "d": d,
        "theta": theta,
        "alpha": alpha,
        "tangent": tangent,
        "wrap_drive": math.pi + 2.0 * alpha,
        "wrap_idler": math.pi - 2.0 * alpha,
        "perimeter": 2.0 * tangent + R * (math.pi + 2.0 * alpha) + r * (math.pi - 2.0 * alpha),
        "phi_top": theta + (math.pi / 2.0 - alpha),  # outward normal of the top run
        "phi_bottom": theta - (math.pi / 2.0 - alpha),  # outward normal of the bottom run
    }


def make_two_pulley_track_cfg(
    name: str,
    chassis_path: str,
    sprocket_joint_path: str,
    drive_radius: float,
    idler_radius: float,
    idler_offset: Tuple[float, float],
    width: float,
    mass: float,
    elements_per_round: int,
    pitch_diameter: Optional[float] = None,
    belt_thickness: float = 0.02,
    belt_overlap: Optional[float] = None,
    drive_arc_margin: float = 0.01,
    idler_arc_margin: float = 0.01,
    grouser: Optional[GrouserCfg] = None,
    origin_pos: Vec3 = (0.0, 0.0, 0.0),
    origin_rpy: Vec3 = (0.0, 0.0, 0.0),
    drive: Optional[DriveCfg] = None,
    surface: Optional[SurfaceCfg] = None,
    scope_path: Optional[str] = None,
) -> TrackCfg:
    """A four-segment belt around a drive pulley and an idler of a different radius.

    This is the shape a real flipper track has: a big drive sprocket at the
    pivot, a small idler at the tip, two straight runs tangent to both.  The
    track origin frame is the **drive pulley centre**, so ``origin_pos`` is
    simply where that wheel sits in the chassis (or flipper) frame.

    Segments, in traversal order::

        straight_segment0  top run, drive -> idler      (prismatic)
        arc_segment0       around the idler             (revolute, radius r)
        straight_segment1  bottom run, idler -> drive   (prismatic)
        arc_segment1       around the drive pulley      (revolute, radius R)

    Every segment link frame sits on the belt surface with +X along the
    direction of travel and +Z pointing outwards, the same convention
    :func:`make_oval_track_cfg` uses, so grouser offsets carry over unchanged.

    Unlike the symmetric oval -- where the original plugin gave every segment
    ``mass / 4`` because the four were pairwise equal -- mass here is split in
    proportion to segment length, since a tapered belt's segments differ a lot.
    """
    g = two_pulley_geometry(drive_radius, idler_radius, idler_offset)
    R, r = g["R"], g["r"]
    phi_top, phi_bottom = g["phi_top"], g["phi_bottom"]
    tangent = g["tangent"]
    cx, cz = float(idler_offset[0]), float(idler_offset[1])

    def on_circle(centre, radius, phi):
        return (centre[0] + radius * math.cos(phi), centre[1] + radius * math.sin(phi))

    drive_c = (0.0, 0.0)
    idler_c = (cx, cz)

    p_top_drive = on_circle(drive_c, R, phi_top)
    p_top_idler = on_circle(idler_c, r, phi_top)
    p_bot_idler = on_circle(idler_c, r, phi_bottom)
    p_bot_drive = on_circle(drive_c, R, phi_bottom)

    # travel direction is the outward normal rotated by -90 degrees; a segment
    # frame is a pure pitch rotation about Y, and a +Y rotation carries +X
    # towards -Z, so the pitch angle is the negated travel angle
    psi_top = phi_top - math.pi / 2.0
    psi_bottom = phi_bottom - math.pi / 2.0

    lengths = [tangent, r * g["wrap_idler"], tangent, R * g["wrap_drive"]]
    total_len = sum(lengths)
    masses = [mass * L / total_len for L in lengths]

    # Every segment slides by up to +/- pitch/2 at runtime, so a straight run
    # authored at exactly the tangent length pulls away from the pulley it just
    # left and opens a visible gap.  Extend the *collider box* past both tangent
    # points so the belt always reads as continuous.  The path length --
    # end_position, which sets where the grousers go -- is untouched: only the
    # body shape grows.  Belt segments are filtered against each other, so the
    # overlap costs nothing in physics.
    pitch = (2.0 * tangent + R * g["wrap_drive"] + r * g["wrap_idler"]) / elements_per_round
    overlap = pitch / 2.0 if belt_overlap is None else float(belt_overlap)

    def straight(index: int, start, psi: float, length: float, seg_mass: float) -> SegmentCfg:
        size: Vec3 = (length + 2.0 * overlap, width, belt_thickness)
        com: Vec3 = (length / 2.0, 0.0, -belt_thickness / 2.0)
        return SegmentCfg(
            name=f"{name}_straight_segment{index}",
            joint_type="prismatic",
            end_position=length,
            axis=(1.0, 0.0, 0.0),
            joint_pos=(0.0, 0.0, 0.0),
            zero_pos=(start[0], 0.0, start[1]),
            zero_rpy=(0.0, -psi, 0.0),
            mass=seg_mass,
            # inertia from the true belt length, not the padded collider
            inertia_diag=_box_inertia(seg_mass, (length, width, belt_thickness)),
            com_pos=com,
            body_collider=ColliderCfg(BoxGeom(size), pos=com),
        )

    def arc(
        index: int,
        start,
        psi: float,
        radius: float,
        wrap: float,
        seg_mass: float,
        margin: float,
    ) -> SegmentCfg:
        com: Vec3 = (0.0, 0.0, -radius)
        rpy: Vec3 = (math.pi / 2.0, 0.0, 0.0)
        return SegmentCfg(
            name=f"{name}_arc_segment{index}",
            joint_type="revolute",
            end_position=wrap,
            axis=(0.0, 1.0, 0.0),
            joint_pos=(0.0, 0.0, -radius),
            zero_pos=(start[0], 0.0, start[1]),
            zero_rpy=(0.0, -psi, 0.0),
            mass=seg_mass,
            inertia_diag=_cylinder_inertia(seg_mass, width, radius),
            com_pos=com,
            com_rpy=rpy,
            # ``margin`` grows the disc past the belt path.  0 puts its surface
            # exactly on the path, flush with the straight runs' outer face and
            # with the base of every grouser.  Keep it below the grouser height
            # or the disc swallows the grousers on that arc.  The disc is one
            # prim serving as both the visual and the collider, so this changes
            # what you see and what touches the ground together.
            body_collider=ColliderCfg(
                CylinderGeom(radius=radius + margin, length=width, axis="Z"),
                pos=com,
                rpy=rpy,
            ),
        )

    segments = [
        straight(0, p_top_drive, psi_top, tangent, masses[0]),
        arc(0, p_top_idler, psi_top, r, g["wrap_idler"], masses[1], idler_arc_margin),
        straight(1, p_bot_idler, psi_bottom, tangent, masses[2]),
        arc(1, p_bot_drive, psi_bottom, R, g["wrap_drive"], masses[3], drive_arc_margin),
    ]

    cfg = TrackCfg(
        name=name,
        chassis_path=chassis_path,
        sprocket=SprocketCfg(
            joint_path=sprocket_joint_path,
            pitch_diameter=pitch_diameter if pitch_diameter is not None else 2.0 * R,
        ),
        segments=segments,
        elements_per_round=elements_per_round,
        grouser=grouser if grouser is not None else GrouserCfg(size=(0.01, width, 0.01)),
        origin_pos=origin_pos,
        origin_rpy=origin_rpy,
        drive=drive if drive is not None else DriveCfg(),
        surface=surface if surface is not None else SurfaceCfg(),
        scope_path=scope_path,
    )
    cfg.validate()
    return cfg
