"""USD prim generation for a continuous track.

Port of the "building the track" half of ``ContinuousTrack`` --
``FillSegmentLength()``, ``ComposeSegments()`` and the collision-filter part of
``InitTrack()`` (gazebo_continuous_track.hpp:106, :256, :355).

The original had to synthesise SDF at runtime and reach into Gazebo private
members (``gazebo_patch.hpp``) to create nested models, links and joints.  Here
the same structure is authored directly as USD prims, which is why this file is
a fraction of the size of its C++ counterpart.

Phases follow plan section 4.2:

  A. compute ``joint_to_track`` / ``length`` / ``perimeter`` / ``pitch``
  B. create segment rigid bodies, their own belt colliders, and the joints that
     hang them off the chassis, with a velocity drive standing in for the ODE
     joint motor
  C. distribute the grousers along the perimeter
  D. collision groups so the belt collides with the environment only
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from .math_utils import Pose, quat_from_axis_angle, quat_from_rpy, quat_identity
from .track_config import BoxGeom, ColliderCfg, CylinderGeom, SurfaceCfg, TrackCfg
from .track_geometry import (
    SegmentGeometry,
    SegmentHandle,
    TrackGeometry,
    TrackHandle,
    fill_segment_length,
    iter_grouser_placements,
)

__all__ = [
    "SegmentGeometry",
    "TrackGeometry",
    "SegmentHandle",
    "TrackHandle",
    "fill_segment_length",
    "build_track",
]

_EPS = 1e-9


# ---------------------------------------------------------------------------
# USD helpers
# ---------------------------------------------------------------------------


def _to_quatf(quat: np.ndarray) -> Gf.Quatf:
    return Gf.Quatf(float(quat[0]), Gf.Vec3f(float(quat[1]), float(quat[2]), float(quat[3])))


def _to_quatd(quat: np.ndarray) -> Gf.Quatd:
    return Gf.Quatd(float(quat[0]), Gf.Vec3d(float(quat[1]), float(quat[2]), float(quat[3])))


def _matrix_scale(matrix: Gf.Matrix4d) -> np.ndarray:
    """Length of each basis row -- the scale baked into a local-to-world matrix."""
    return np.array([Gf.Vec3d(matrix.GetRow3(i)).GetLength() for i in range(3)])


def _gf_to_pose(matrix: Gf.Matrix4d) -> Pose:
    """Translation + rotation of a transform matrix, with any scale divided out.

    ``Gf.Matrix4d.ExtractRotationQuat()`` returns an unnormalised quaternion when
    the matrix carries scale, so the basis rows are normalised first.
    """
    t = matrix.ExtractTranslation()
    scale = _matrix_scale(matrix)
    rigid = Gf.Matrix4d(matrix)
    for i in range(3):
        if scale[i] > _EPS:
            rigid.SetRow3(i, Gf.Vec3d(matrix.GetRow3(i)) / scale[i])
    q = rigid.ExtractRotationQuat().GetNormalized()
    imag = q.GetImaginary()
    return Pose(
        np.array([t[0], t[1], t[2]]),
        np.array([q.GetReal(), imag[0], imag[1], imag[2]]),
    )


def _assert_unscaled(stage: Usd.Stage, path: str, role: str) -> None:
    """Reject a scaled rigid body before it silently corrupts the joint frames.

    USD physics multiplies a joint's ``localPos`` by the scale of the body it is
    authored against, so a chassis prim carrying ``xformOp:scale`` pulls every
    track and sprocket anchor towards its own origin -- the track ends up inside
    the hull and the vehicle sits on its belly.  It is an easy mistake to make
    (a ``UsdGeom.Cube`` sized by scale is the obvious way to build a box) and
    an expensive one to debug, so it is caught here.
    """
    prim = stage.GetPrimAtPath(path)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    scale = _matrix_scale(cache.GetLocalToWorldTransform(prim))
    if np.any(np.abs(scale - 1.0) > 1e-6):
        raise ValueError(
            f"{role} prim {path!r} has a non-identity world scale "
            f"({scale[0]:.4g}, {scale[1]:.4g}, {scale[2]:.4g}). USD physics applies that "
            "scale to joint local frames, so the track would be built at the wrong place. "
            "Make the rigid body an unscaled Xform and put the scale on a child geometry "
            "prim instead."
        )


def _pose_relative(stage: Usd.Stage, path: str, ref_path: str) -> Pose:
    """Pose of the prim at ``path`` expressed in the frame of ``ref_path``."""
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    target = stage.GetPrimAtPath(path)
    ref = stage.GetPrimAtPath(ref_path)
    if not target or not ref:
        raise ValueError(f"cannot resolve transform between {path!r} and {ref_path!r}")
    m_target = cache.GetLocalToWorldTransform(target)
    m_ref = cache.GetLocalToWorldTransform(ref)
    return _gf_to_pose(m_target * m_ref.GetInverse())


def _set_local_pose(prim: Usd.Prim, pose: Pose) -> UsdGeom.Xformable:
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pose.pos]))
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(_to_quatd(pose.quat))
    return xf


def _axis_alignment(axis: Sequence[float]) -> Tuple[str, np.ndarray]:
    """Map a free axis onto a USD joint axis token plus a joint-frame rotation.

    USD physics joints only accept "X" / "Y" / "Z"; a non-canonical axis is
    handled by rotating the joint frame so that its X maps onto the axis.
    """
    a = np.asarray(axis, dtype=np.float64)
    n = float(np.linalg.norm(a))
    if n < _EPS:
        raise ValueError("joint axis must be non-zero")
    a = a / n

    for token, base in (("X", (1.0, 0.0, 0.0)), ("Y", (0.0, 1.0, 0.0)), ("Z", (0.0, 0.0, 1.0))):
        if np.allclose(a, base, atol=1e-9):
            return token, quat_identity()

    x = np.array([1.0, 0.0, 0.0])
    if np.allclose(a, -x, atol=1e-9):
        return "X", quat_from_axis_angle((0.0, 0.0, 1.0), math.pi)
    cross = np.cross(x, a)
    return "X", quat_from_axis_angle(cross, math.atan2(float(np.linalg.norm(cross)), float(x @ a)))


def _apply_collider(
    prim: Usd.Prim, surface: SurfaceCfg, material: Optional[UsdShade.Material]
) -> None:
    UsdPhysics.CollisionAPI.Apply(prim)
    physx = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    physx.CreateContactOffsetAttr().Set(surface.contact_offset)
    physx.CreateRestOffsetAttr().Set(surface.rest_offset)
    if material is not None:
        binding = UsdShade.MaterialBindingAPI.Apply(prim)
        binding.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _create_box(
    stage: Usd.Stage, path: str, size: Sequence[float], pose: Pose, visible: bool = True
) -> Usd.Prim:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateExtentAttr([Gf.Vec3f(-0.5, -0.5, -0.5), Gf.Vec3f(0.5, 0.5, 0.5)])
    xf = _set_local_pose(cube.GetPrim(), pose)
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    if not visible:
        cube.CreatePurposeAttr(UsdGeom.Tokens.guide)
    return cube.GetPrim()


def _set_render_refinement(prim: Usd.Prim, level: int = 3) -> None:
    """Ask Hydra to tessellate an implicit shape more finely.

    ``UsdGeom.Cylinder`` is an analytic shape; the renderer picks how many
    segments to draw it with, and the default is coarse enough that a belt
    pulley reads as a polygon.  These two attributes are the Omniverse-side
    override.  Purely cosmetic -- collision uses the analytic shape either way.
    """
    prim.CreateAttribute("refinementEnableOverride", Sdf.ValueTypeNames.Bool).Set(True)
    prim.CreateAttribute("refinementLevel", Sdf.ValueTypeNames.Int).Set(int(level))


def _create_cylinder(
    stage: Usd.Stage,
    path: str,
    radius: float,
    height: float,
    axis: str,
    pose: Pose,
    visible: bool = True,
) -> Usd.Prim:
    # NOTE: PhysX represents an analytic cylinder as custom geometry.  If custom
    # geometry is disabled in the physics scene, Isaac falls back to a convex
    # hull approximation of this gprim -- acceptable for the belt body, which
    # only carries load through the grousers, but worth knowing.
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(height)
    cyl.CreateAxisAttr(axis)
    half = max(radius, height / 2.0)
    cyl.CreateExtentAttr([Gf.Vec3f(-half, -half, -half), Gf.Vec3f(half, half, half)])
    _set_local_pose(cyl.GetPrim(), pose)
    _set_render_refinement(cyl.GetPrim())
    if not visible:
        cyl.CreatePurposeAttr(UsdGeom.Tokens.guide)
    return cyl.GetPrim()


def _create_collider_prim(
    stage: Usd.Stage,
    path: str,
    collider: ColliderCfg,
    surface: SurfaceCfg,
    material: Optional[UsdShade.Material],
    visible: bool = True,
) -> Usd.Prim:
    pose = Pose.from_rpy(collider.pos, collider.rpy)
    geom = collider.geom
    if isinstance(geom, BoxGeom):
        prim = _create_box(stage, path, geom.size, pose, visible=visible)
    elif isinstance(geom, CylinderGeom):
        prim = _create_cylinder(
            stage, path, geom.radius, geom.length, geom.axis, pose, visible=visible
        )
    else:
        raise TypeError(f"unsupported collider geometry: {type(geom).__name__}")
    _apply_collider(prim, surface, material)
    return prim


def _get_or_create_physics_material(stage: Usd.Stage, path: str, surface: SurfaceCfg):
    prim = stage.GetPrimAtPath(path)
    if prim and prim.IsValid():
        return UsdShade.Material(prim)
    material = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr().Set(surface.static_friction)
    api.CreateDynamicFrictionAttr().Set(surface.dynamic_friction)
    api.CreateRestitutionAttr().Set(surface.restitution)
    return material


def _get_or_create_collision_group(stage: Usd.Stage, path: str) -> UsdPhysics.CollisionGroup:
    prim = stage.GetPrimAtPath(path)
    if prim and prim.IsValid():
        return UsdPhysics.CollisionGroup(prim)
    return UsdPhysics.CollisionGroup.Define(stage, path)


def _group_collection(group: UsdPhysics.CollisionGroup) -> Usd.CollectionAPI:
    try:
        return group.GetCollidersCollectionAPI()
    except AttributeError:  # pragma: no cover - very old USD
        return Usd.CollectionAPI.Apply(group.GetPrim(), "colliders")


def _add_group_members(group: UsdPhysics.CollisionGroup, paths: Sequence[str]) -> None:
    includes = _group_collection(group).CreateIncludesRel()
    existing = set(includes.GetTargets())
    for path in paths:
        sdf_path = Sdf.Path(path)
        if sdf_path not in existing:
            includes.AddTarget(sdf_path)
            existing.add(sdf_path)


def _add_filtered_groups(group: UsdPhysics.CollisionGroup, paths: Sequence[str]) -> None:
    rel = group.CreateFilteredGroupsRel()
    existing = set(rel.GetTargets())
    for path in paths:
        sdf_path = Sdf.Path(path)
        if sdf_path not in existing:
            rel.AddTarget(sdf_path)
            existing.add(sdf_path)


def _apply_drive(prim: Usd.Prim, joint_type: str, cfg: TrackCfg, joint_to_track: float) -> None:
    """Velocity drive standing in for ODE's ``fmax`` / ``vel`` joint motor.

    ``SetJointMotorVelocity()`` set a velocity target plus a force ceiling; a
    ``UsdPhysics.DriveAPI`` with ``stiffness == 0`` is the same controller.

    Two unit conversions happen here.

    1. ``DriveCfg`` gains are given in track space (N and N/(m/s)); a segment
       whose joint position scales by ``s = joint_to_track`` needs an effort of
       ``F * s`` and a damping of ``D * s**2`` to produce the same force at the
       belt surface.  For a prismatic segment ``s == 1``, so nothing changes.
    2. USD authors *angular* drive gains and targets per degree, while the
       tensor API the driver uses at runtime speaks radians.  Angular gains are
       scaled by ``pi/180`` so that both ends agree.
    """
    drive_cfg = cfg.drive
    scale = joint_to_track
    if joint_type == "prismatic":
        token = "linear"
        deg_scale = 1.0
    else:
        token = "angular"
        deg_scale = math.pi / 180.0

    drive = UsdPhysics.DriveAPI.Apply(prim, token)
    drive.CreateTypeAttr().Set("force")
    drive.CreateStiffnessAttr().Set(drive_cfg.stiffness * scale * scale * deg_scale)
    drive.CreateDampingAttr().Set(drive_cfg.damping * scale * scale * deg_scale)
    drive.CreateMaxForceAttr().Set(drive_cfg.max_force * scale)
    drive.CreateTargetPositionAttr().Set(0.0)
    drive.CreateTargetVelocityAttr().Set(0.0)


# ---------------------------------------------------------------------------
# Phases B-D
# ---------------------------------------------------------------------------


def build_track(
    stage: Usd.Stage,
    cfg: TrackCfg,
    body_paths: Optional[Sequence[str]] = None,
    collision_groups_scope: Optional[str] = None,
) -> TrackHandle:
    """Create the belt of one continuous track and return its handle.

    ``body_paths`` lists the prims that the belt must *not* collide with -- the
    chassis, the sprocket and any road wheels wrapped by the track.  It is the
    counterpart of the ``GZ_GHOST_COLLIDE`` bookkeeping in ``InitTrack()``
    (hpp:355), which collected the sprocket's parent/child links and the segment
    joints' parent links.  Defaults to the chassis alone.
    """
    cfg.validate()

    chassis_prim = stage.GetPrimAtPath(cfg.chassis_path)
    if not chassis_prim or not chassis_prim.IsValid():
        raise ValueError(f"chassis prim not found: {cfg.chassis_path}")
    _assert_unscaled(stage, cfg.chassis_path, "chassis")

    scope_path = cfg.scope_path or f"{Sdf.Path(cfg.chassis_path).GetParentPath()}/{cfg.name}"
    root_path = str(Sdf.Path(scope_path).GetParentPath())
    UsdGeom.Xform.Define(stage, scope_path)

    # ---- Phase A ----------------------------------------------------------
    geometry = fill_segment_length(cfg)
    pitch = geometry.pitch

    material = _get_or_create_physics_material(
        stage, f"{root_path}/PhysicsMaterials/{cfg.name}_belt", cfg.surface
    )

    # The track origin frame, expressed relative to the scope's parent, so that
    # the segments can be siblings of the chassis (PhysX forbids nesting rigid
    # bodies) while still being placed as if they were children of it.
    chassis_local = _pose_relative(stage, cfg.chassis_path, root_path)
    track_origin = chassis_local * Pose.from_rpy(cfg.origin_pos, cfg.origin_rpy)

    handles: List[SegmentHandle] = []

    # ---- Phase B ----------------------------------------------------------
    for seg, seg_geom in zip(cfg.segments, geometry.segments):
        link_path = f"{scope_path}/{seg.link_name}"
        zero_pose = Pose.from_rpy(seg.zero_pos, seg.zero_rpy)

        link = UsdGeom.Xform.Define(stage, link_path)
        _set_local_pose(link.GetPrim(), track_origin * zero_pose)

        UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
        mass_api = UsdPhysics.MassAPI.Apply(link.GetPrim())
        mass_api.CreateMassAttr().Set(seg.mass)
        mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*[float(v) for v in seg.inertia_diag]))
        mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*[float(v) for v in seg.com_pos]))
        mass_api.CreatePrincipalAxesAttr().Set(_to_quatf(quat_from_rpy(*seg.com_rpy)))

        # the belt body itself (plan 1.3): the segment link carries its own
        # box / cylinder collider on top of the grousers
        if seg.body_collider is not None:
            _create_collider_prim(
                stage, f"{link_path}/belt", seg.body_collider, cfg.surface, material
            )

        # joint hanging the segment off the chassis
        joint_path = f"{scope_path}/{seg.joint_name}"
        axis_token, axis_rot = _axis_alignment(seg.axis)
        # joint frame in the child link frame == SDF <joint><pose>, rotated so
        # that the USD axis token lines up with the SDF <axis><xyz>
        frame_in_child = Pose(np.asarray(seg.joint_pos, dtype=np.float64), axis_rot)
        frame_in_chassis = Pose.from_rpy(cfg.origin_pos, cfg.origin_rpy) * zero_pose * frame_in_child

        if seg.joint_type == "prismatic":
            joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
            lower, upper = -pitch, pitch
        else:
            joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
            # revolute limits are authored in degrees
            limit = math.degrees(pitch / seg_geom.joint_to_track)
            lower, upper = -limit, limit

        joint.CreateBody0Rel().SetTargets([cfg.chassis_path])
        joint.CreateBody1Rel().SetTargets([link_path])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in frame_in_chassis.pos]))
        joint.CreateLocalRot0Attr().Set(_to_quatf(frame_in_chassis.quat))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*[float(v) for v in frame_in_child.pos]))
        joint.CreateLocalRot1Attr().Set(_to_quatf(frame_in_child.quat))
        joint.CreateAxisAttr(axis_token)
        joint.CreateExcludeFromArticulationAttr(False)
        # plan 4.2 step 5: limits generous enough to contain the +/- pitch/2 rewind
        joint.CreateLowerLimitAttr(lower)
        joint.CreateUpperLimitAttr(upper)

        _apply_drive(joint.GetPrim(), seg.joint_type, cfg, seg_geom.joint_to_track)

        handles.append(
            SegmentHandle(
                cfg=seg, geometry=seg_geom, link_path=link_path, joint_path=joint_path
            )
        )

    # ---- Phase C -- ComposeSegments() (hpp:106) ----------------------------
    element_count = _place_grousers(stage, cfg, geometry, handles, material)

    # ---- Phase D ----------------------------------------------------------
    groups_scope = collision_groups_scope or f"{root_path}/CollisionGroups"
    belt_group_path = f"{groups_scope}/{cfg.name}_belt"
    body_group_path = f"{groups_scope}/body"

    belt_group = _get_or_create_collision_group(stage, belt_group_path)
    body_group = _get_or_create_collision_group(stage, body_group_path)

    _add_group_members(belt_group, [h.link_path for h in handles])
    _add_group_members(body_group, list(body_paths) if body_paths else [cfg.chassis_path])

    # belt does not collide with the wrapped body, nor with itself; everything
    # else (the environment) still collides -- the GZ_GHOST_COLLIDE behaviour
    _add_filtered_groups(belt_group, [body_group_path, belt_group_path])
    _add_filtered_groups(body_group, [belt_group_path])

    return TrackHandle(
        name=cfg.name,
        cfg=cfg,
        geometry=geometry,
        segments=handles,
        scope_path=scope_path,
        element_count=element_count,
        belt_group_path=belt_group_path,
        body_group_path=body_group_path,
    )


def _place_grousers(
    stage: Usd.Stage,
    cfg: TrackCfg,
    geometry: TrackGeometry,
    handles: Sequence[SegmentHandle],
    material: Optional[UsdShade.Material],
) -> int:
    """Author the grousers that ``iter_grouser_placements()`` lays out.

    The distribution itself lives in :mod:`isaac_continuous_track.track_geometry`
    so it can be tested without Isaac Sim; this function only turns each pose
    into a collider prim on the segment link it belongs to.
    """
    grouser = cfg.grouser
    count = 0
    for placement in iter_grouser_placements(cfg, geometry):
        handle = handles[placement.segment_index]
        prim = _create_box(
            stage,
            f"{handle.link_path}/element{placement.element_index}",
            grouser.size,
            placement.pose,
            visible=grouser.visual,
        )
        _apply_collider(prim, cfg.surface, material)
        count += 1
    return count
