r"""Standalone Isaac Sim example: a two-track vehicle driven by continuous tracks.

Run with Isaac Sim's python launcher.  Either invocation works.

By file path -- including through a symlink of the package dropped into the
Isaac Sim install root, which needs no PYTHONPATH at all::

    C:\isaac-sim-5.1.0\python.bat ^
        C:\isaac-sim-5.1.0\isaac_continuous_track\example\spawn_track_robot.py --headless

Or as a module, from the repository root::

    set PYTHONPATH=%CD%
    C:\isaac-sim-5.1.0\python.bat -m isaac_continuous_track.example.spawn_track_robot --headless

On Linux the launcher is ``./python.sh`` in the Isaac Sim install directory.

The robot is deliberately minimal -- a box chassis and one sprocket flywheel per
side -- so that what is being exercised is the track port and nothing else.

By default the run prints the flat-ground slip ratio,

    slip = 1 - v_body / (omega * pitch_diameter / 2)

which is verification step 2 of the port plan and the metric that tells you
whether the rewind is costing you traction.

``--turn`` drives the tracks in opposite directions and reports yaw rate instead
(step 5); slip is a straight-line metric and means nothing there.

``--no-driver`` is the ablation: the belts are built and the sprockets spin, but
TrackDriver is never registered.  The vehicle should stay put (slip == 1), which
is what attributes the motion in the default run to the belt segments rather
than to anything else in the scene.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import List, Tuple

# Running this file by path (rather than with -m) puts only its own directory on
# sys.path, so the package two levels up would not be importable.  Add the
# directory that *contains* the package.
#
# realpath() deliberately follows symlinks.  This file is often reached through a
# symlink dropped into the Isaac Sim install root, and that root must NOT end up
# on sys.path: it holds an `isaacsim/` directory with no __init__.py, which
# Python would happily import as a namespace package shadowing the real
# `isaacsim` module.  Resolving the symlink lands on the actual checkout
# instead.  Appending rather than inserting keeps Isaac's own paths ahead of
# ours in any case.
_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if _PKG_PARENT not in sys.path:
    sys.path.append(_PKG_PARENT)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="run without a viewport")
    parser.add_argument("--seconds", type=float, default=8.0, help="simulated duration")
    parser.add_argument("--track-speed", type=float, default=0.5, help="commanded track speed [m/s]")
    parser.add_argument(
        "--turn",
        action="store_true",
        help="drive the tracks in opposite directions (verification step 5)",
    )
    parser.add_argument("--physics-hz", type=float, default=250.0, help="physics rate, plan 5")
    parser.add_argument(
        "--elements-per-round", type=int, default=32, help="number of grousers on the track"
    )
    parser.add_argument(
        "--ground-friction",
        type=float,
        default=0.8,
        help="ground friction. Set 0 to prove the belts are driving by contact: "
        "with no friction a real track cannot move the vehicle at all.",
    )
    parser.add_argument(
        "--track-friction",
        type=float,
        default=0.8,
        help="belt friction. PhysX combines the two contacting materials "
        "(averaging them by default), so zeroing only the ground still leaves "
        "half the friction -- a frictionless test has to zero BOTH.",
    )
    parser.add_argument(
        "--no-driver",
        action="store_true",
        help="ablation: build the belts but never run TrackDriver, so only the "
        "sprocket drive is active. The vehicle should not move.",
    )
    return parser.parse_args(argv)


ARGS = parse_args(sys.argv[1:])

# SimulationApp has to exist before any omni / pxr import.
from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": ARGS.headless})

import numpy as np  # noqa: E402
from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402

from isaac_continuous_track import DriveCfg, GrouserCfg, SurfaceCfg, TrackDriver  # noqa: E402
from isaac_continuous_track.isaac_compat import (  # noqa: E402
    get_articulation_cls,
    get_current_stage,
)
from isaac_continuous_track.track_config import make_oval_track_cfg  # noqa: E402
from isaac_continuous_track.track_builder import build_track  # noqa: E402


# ---------------------------------------------------------------------------
# vehicle dimensions
# ---------------------------------------------------------------------------

ROOT = "/World/Robot"
CHASSIS = f"{ROOT}/chassis"

CHASSIS_SIZE = (0.6, 0.4, 0.2)
CHASSIS_MASS = 20.0
CHASSIS_Z = 0.32  # spawn height; the tracks settle onto the ground

TRACK_LENGTH = 0.5  # straight run length
TRACK_RADIUS = 0.1  # arc radius == sprocket pitch radius
TRACK_WIDTH = 0.12
TRACK_MASS = 2.0  # per track, split across the four segments
PITCH_DIAMETER = 2.0 * TRACK_RADIUS

TRACK_Y = 0.2  # lateral offset of each track from the chassis centre
TRACK_Z = -0.2  # track centre below the chassis frame
SPROCKET_X = 0.25

SIDES: Tuple[Tuple[str, float], ...] = (("left_track", TRACK_Y), ("right_track", -TRACK_Y))


# ---------------------------------------------------------------------------
# stage construction
# ---------------------------------------------------------------------------


def _xform(stage, path, pos=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)):
    prim = UsdGeom.Xform.Define(stage, path)
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    xf.AddRotateXYZOp().Set(Gf.Vec3d(*[math.degrees(a) for a in rpy]))
    return prim


def _box(stage, path, size, pos=(0.0, 0.0, 0.0)):
    """A unit cube scaled to ``size``.

    NEVER apply RigidBodyAPI to the prim this returns: USD physics multiplies a
    joint's local frame by the scale of the body it is authored against, so a
    scaled rigid body silently drags every joint anchor towards its own origin.
    Use ``_rigid_box()`` for bodies; this is for pure geometry only.
    """
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateExtentAttr([Gf.Vec3f(-0.5, -0.5, -0.5), Gf.Vec3f(0.5, 0.5, 0.5)])
    xf = UsdGeom.Xformable(cube)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    xf.AddScaleOp().Set(Gf.Vec3f(*size))
    return cube


def _rigid_box(stage, path, size, pos=(0.0, 0.0, 0.0), mass=1.0, collision=True):
    """An unscaled Xform rigid body with the scaled box geometry underneath it."""
    body = _xform(stage, path, pos=pos)
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(body.GetPrim())
    mass_api.CreateMassAttr().Set(mass)
    mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*_box_inertia(mass, size)))

    geom = _box(stage, f"{path}/geom", size)
    if collision:
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
    return body


def _yaw_of(quat) -> float:
    """Yaw from a (w, x, y, z) quaternion, the convention get_world_poses returns."""
    w, x, y, z = (float(v) for v in quat)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _box_inertia(mass, size):
    x, y, z = size
    return (
        mass * (y * y + z * z) / 12.0,
        mass * (z * z + x * x) / 12.0,
        mass * (x * x + y * y) / 12.0,
    )


def build_scene(stage) -> None:
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")

    light = UsdLux.DistantLight.Define(stage, "/World/light")
    light.CreateIntensityAttr(2500.0)

    # physics scene -- plan section 5 baseline
    scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    physx_scene.CreateSolverTypeAttr("TGS")
    physx_scene.CreateTimeStepsPerSecondAttr(int(ARGS.physics_hz))
    physx_scene.CreateEnableCCDAttr(False)

    ground_surface = SurfaceCfg(
        static_friction=ARGS.ground_friction, dynamic_friction=ARGS.ground_friction
    )
    ground_material = UsdShade.Material.Define(stage, "/World/PhysicsMaterials/ground")
    ground_api = UsdPhysics.MaterialAPI.Apply(ground_material.GetPrim())
    ground_api.CreateStaticFrictionAttr().Set(ground_surface.static_friction)
    ground_api.CreateDynamicFrictionAttr().Set(ground_surface.dynamic_friction)
    ground_api.CreateRestitutionAttr().Set(0.0)

    ground = _box(stage, "/World/ground", (60.0, 60.0, 0.2), pos=(0.0, 0.0, -0.1))
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(
        ground_material, UsdShade.Tokens.weakerThanDescendants, "physics"
    )


def build_robot(stage) -> List[str]:
    """Chassis plus one sprocket per side. Returns the paths of the body group."""
    root = _xform(stage, ROOT, pos=(0.0, 0.0, CHASSIS_Z))
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

    # The chassis is an *unscaled* Xform; the box geometry hangs underneath it.
    # Scaling the body prim itself would scale every joint frame authored
    # against it -- see _box().
    _rigid_box(stage, CHASSIS, CHASSIS_SIZE, mass=CHASSIS_MASS)

    body_paths = [CHASSIS]

    for name, y in SIDES:
        # The sprocket is a reference flywheel: it sets the track speed and is
        # not expected to touch anything, so it carries no collider.  This
        # mirrors the original, where the sprocket joint never feels track load
        # (plan section 3.3).
        sprocket_path = f"{ROOT}/{name}_sprocket"
        sprocket = _xform(stage, sprocket_path, pos=(SPROCKET_X, y, TRACK_Z))
        UsdPhysics.RigidBodyAPI.Apply(sprocket.GetPrim())
        sprocket_mass = UsdPhysics.MassAPI.Apply(sprocket.GetPrim())
        sprocket_mass.CreateMassAttr().Set(0.5)
        sprocket_mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(2.0e-3, 2.5e-3, 2.0e-3))

        marker = UsdGeom.Cylinder.Define(stage, f"{sprocket_path}/geom")
        marker.CreateRadiusAttr(TRACK_RADIUS * 0.5)
        marker.CreateHeightAttr(0.02)
        marker.CreateAxisAttr("Y")
        marker.CreateExtentAttr(
            [
                Gf.Vec3f(-TRACK_RADIUS, -TRACK_RADIUS, -TRACK_RADIUS),
                Gf.Vec3f(TRACK_RADIUS, TRACK_RADIUS, TRACK_RADIUS),
            ]
        )

        joint_path = f"{ROOT}/{name}_sprocket_joint"
        joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
        joint.CreateBody0Rel().SetTargets([CHASSIS])
        joint.CreateBody1Rel().SetTargets([sprocket_path])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(SPROCKET_X, y, TRACK_Z))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateAxisAttr("Y")
        # Free spinning.  Leave the limit attributes UNAUTHORED: an unlimited
        # USD physics joint is one with no limits, not one with lower > upper.
        # Authoring lower=1, upper=-1 locks the joint near zero instead.

        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateTypeAttr().Set("force")
        drive.CreateStiffnessAttr().Set(0.0)
        drive.CreateDampingAttr().Set(50.0 * math.pi / 180.0)  # N.m.s/rad -> per degree
        drive.CreateMaxForceAttr().Set(200.0)
        drive.CreateTargetVelocityAttr().Set(0.0)

        body_paths.append(sprocket_path)

    return body_paths


def build_tracks(stage, body_paths):
    handles = []
    for name, y in SIDES:
        cfg = make_oval_track_cfg(
            name=name,
            chassis_path=CHASSIS,
            sprocket_joint_path=f"{ROOT}/{name}_sprocket_joint",
            pitch_diameter=PITCH_DIAMETER,
            length=TRACK_LENGTH,
            radius=TRACK_RADIUS,
            width=TRACK_WIDTH,
            mass=TRACK_MASS,
            elements_per_round=ARGS.elements_per_round,
            grouser=GrouserCfg(size=(0.012, TRACK_WIDTH, 0.012)),
            origin_pos=(0.0, y, TRACK_Z),
            # gains in track space: ~500 N of tractive force is plenty for a 24 kg
            # vehicle, and the damping holds the belt within ~1 cm/s of the command
            drive=DriveCfg(stiffness=0.0, damping=5.0e4, max_force=500.0),
            surface=SurfaceCfg(
                static_friction=ARGS.track_friction,
                dynamic_friction=ARGS.track_friction,
                contact_offset=0.02,
                rest_offset=0.0,
            ),
        )
        handles.append(build_track(stage, cfg, body_paths=body_paths))
    return handles


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    try:
        from isaacsim.core.api import SimulationContext
    except ImportError:  # Isaac Sim <= 4.2
        from omni.isaac.core import SimulationContext

    sim = SimulationContext(
        physics_dt=1.0 / ARGS.physics_hz,
        rendering_dt=1.0 / 60.0,
        stage_units_in_meters=1.0,
    )
    stage = get_current_stage()

    build_scene(stage)
    body_paths = build_robot(stage)
    handles = build_tracks(stage, body_paths)

    geometry = handles[0].geometry
    # NOTE: flush on every print.  Kit shuts the process down without flushing
    # Python's stdout buffer, so buffered output is lost when the run ends.
    print(
        f"[continuous_track] perimeter={geometry.perimeter:.4f} m  "
        f"pitch={geometry.pitch:.4f} m  grousers/track={handles[0].element_count}",
        flush=True,
    )
    print(
        "[continuous_track] rewind rate at the commanded speed: "
        f"{abs(ARGS.track_speed) / geometry.pitch:.1f} Hz",
        flush=True,
    )

    sim.reset()

    articulation_cls = get_articulation_cls()
    view = articulation_cls(prim_paths_expr=ROOT, name="track_robot")
    try:
        view.initialize()
    except TypeError:  # older signature wants the physics sim view
        view.initialize(sim.physics_sim_view)

    driver = TrackDriver(view, handles)
    driver.reset()
    if not ARGS.no_driver:
        driver.register()  # subscribe_physics_step_events == ConnectWorldUpdateBegin
    else:
        print("[continuous_track] ABLATION: TrackDriver not registered", flush=True)

    commanded = {
        "left_track": ARGS.track_speed,
        "right_track": -ARGS.track_speed if ARGS.turn else ARGS.track_speed,
    }
    driver.set_track_speeds(commanded)

    steps = int(ARGS.seconds * ARGS.physics_hz)
    log_every = max(1, int(0.5 * ARGS.physics_hz))
    prev_pos = None
    prev_yaw = 0.0
    dt = 1.0 / ARGS.physics_hz

    for step in range(steps):
        # the sprocket command is a drive target, so it has to be re-issued only
        # if it changes; re-issuing every step keeps this example simple
        driver.set_track_speeds(commanded)
        sim.step(render=not ARGS.headless)

        positions, orientations = view.get_world_poses()
        pos = np.asarray(positions)[0]
        yaw = _yaw_of(np.asarray(orientations)[0])
        if prev_pos is not None and step % log_every == 0:
            window = log_every * dt
            v_body = float(np.linalg.norm((pos - prev_pos)[:2])) / window
            state = driver.track_state()["left_track"]
            v_track = abs(float(np.asarray(state["track_vel"])[0]))
            if ARGS.turn:
                # slip is a straight-line metric; for a turn in place report the
                # yaw rate instead (verification step 5)
                yaw_rate = math.degrees(_wrap_angle(yaw - prev_yaw)) / window
                print(
                    f"[t={step * dt:5.2f}s] "
                    f"yaw_rate={yaw_rate:7.2f} deg/s  v_body={v_body:6.3f} m/s  "
                    f"v_track={v_track:6.3f} m/s  z={pos[2]:6.3f} m",
                    flush=True,
                )
            else:
                slip = 1.0 - v_body / v_track if v_track > 1e-6 else float("nan")
                print(
                    f"[t={step * dt:5.2f}s] "
                    f"v_body={v_body:6.3f} m/s  v_track={v_track:6.3f} m/s  "
                    f"slip={slip:6.3f}  z={pos[2]:6.3f} m",
                    flush=True,
                )
        if step % log_every == 0:
            prev_pos = pos
            prev_yaw = yaw

    driver.unregister()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
