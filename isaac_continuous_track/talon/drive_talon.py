r"""Drive the re-rigged Talon on continuous tracks and report what it does.

    set ASSET=...\talon_p_v4_0c_ctrack.usd
    set SPEED=0.4  SECONDS=4
    C:\isaac-sim-5.1.0\python.bat isaac_continuous_track\talon\drive_talon.py

Three things this script exists to pin down, all of which cost real debugging
time:

1. ``omni.physx.bundle`` must be enabled before anything references the Talon.
   The asset carries PhysX-only schemas and the physics parse dies with an
   access violation without it.
2. Import ``talon_spec``, never ``rig_talon_track`` -- the latter boots its own
   ``SimulationApp`` at import time and a second one crashes the process.
3. PhysX merges the whole Talon into ONE 28-DOF articulation, so a single
   articulation view covers all four belts.  The four flippers ship joints with
   identical names, which Isaac silently disambiguates; the rig renames them so
   each sprocket can be addressed unambiguously.

The flipper arms must be held with a position target.  Left free they swing
under the belt thrust and forward slip roughly triples.
"""
import math
import os
import sys

from isaacsim import SimulationApp

# Defaults are tuned for "just run it and watch": a viewport opens, the robot
# tries to turn in place, and it keeps going for 30 s.  Every one of these is
# overridable from the environment -- set HEADLESS=1 for a batch run.
# CAPTURE=<dir> saves frames from a diagnostic camera either way.
HEADLESS = os.environ.get("HEADLESS", "0") != "0"
CAPTURE = os.environ.get("CAPTURE", "")

app = SimulationApp({"headless": HEADLESS})

# The Talon asset carries PhysX-only schemas; that bundle has to be up before
# anything references the asset or the physics parse dies with an access
# violation.  (Learned from multi_robot_control/main-ogn-gui-without-graph.py.)
import omni.kit.app  # noqa: E402

_em = omni.kit.app.get_app().get_extension_manager()
_em.set_extension_enabled_immediate("omni.physx.bundle", True)
print("RUN omni.physx.bundle enabled =", _em.is_extension_enabled("omni.physx.bundle"), flush=True)

import math  # noqa: E402
import numpy as np  # noqa: E402
from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402
from isaacsim.core.api import SimulationContext  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage  # noqa: E402

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if _PKG_PARENT not in sys.path:
    sys.path.append(_PKG_PARENT)
from isaac_continuous_track import TrackDriver  # noqa: E402
from isaac_continuous_track.isaac_compat import get_articulation_cls  # noqa: E402
# NOTE: import the spec, never rig_talon_track -- that module boots its own
# SimulationApp at import time and a second one crashes the process.
from isaac_continuous_track.talon import talon_spec as SPEC  # noqa: E402

_HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
ASSET = os.environ.get("ASSET", os.path.join(_HERE, "talon_p_v4_0c_ctrack.usd")).replace("\\", "/")
ROOT = "/World/Talon"
HZ = 250.0
SPEED = float(os.environ.get("SPEED", "0.2"))
SECONDS = float(os.environ.get("SECONDS", "5"))
# Raising the flippers shrinks the ground footprint.  The previous conveyor
# driver cruised at 25 deg for exactly that reason; flat (0 deg) is the worst
# case for turning because all four long patches resist yaw.
ARM_DEG = float(os.environ.get("ARM_DEG", "30"))
# With a viewport, hold at the start line until the user presses Play.  Headless
# has no button to press, so it just goes.
WAIT_FOR_PLAY = os.environ.get("WAIT_FOR_PLAY", "") != "0"
GROUND_MU = float(os.environ.get("GROUND_MU", "0.9"))
REWIND = os.environ.get("REWIND", "1") != "0"
MODE = os.environ.get("MODE", "seq")  # seq | fwd | back | turn | pivot | probe | belt

# Forward, back, turn in place, stop -- 10 s, then it repeats.
SEQ = ((1.0, "fwd"), (1.0, "back"), (3.0, "turn"))
SEQ_TOTAL = sum(d for d, _ in SEQ)


def phase_at(t: float) -> str:
    """Which leg of SEQ time t falls in (the sequence loops)."""
    if MODE != "seq":
        return MODE
    t = t % SEQ_TOTAL
    acc = 0.0
    for dur, kind in SEQ:
        if t < acc + dur:
            return kind
        acc += dur
    return "stop"


# MODE=probe drives one flipper at a time and reports which way the body goes.
# That is the only honest way to fix the per-flipper sign: the asset mixes two
# mirrored subassemblies, so a positive belt command pushes opposite ways on the
# front and rear pairs, and the old conveyor driver's WHEEL_SIGN table described
# *visual wheel spin*, not belt thrust.
PROBE_SECONDS = 1.5


def probe_tag(t: float) -> str:
    tags = [tag for tag, _ in SPEC.FLIPPERS]
    return tags[int(t / PROBE_SECONDS) % len(tags)]


def command_for(phase: str, t: float = 0.0) -> dict:
    """Per-flipper sprocket speeds, in joint sign convention."""
    if phase == "stop":
        return {f"{t}_track": 0.0 for t, _ in SPEC.FLIPPERS}
    if phase == "pivot":
        # only the left side runs; the right side is held at zero and brakes.
        # A pivot turn needs far less yaw torque than a spin turn, so if even
        # this produces no rotation the problem is not the amount of torque.
        return {
            f"{tag}_track": (SPEED * WHEEL_SIGN[tag] if SIDE[tag] == "L" else 0.0)
            for tag, _ in SPEC.FLIPPERS
        }
    if phase == "belt":
        return {f"{tag}_track": SPEED for tag, _ in SPEC.FLIPPERS}
    if phase == "probe":
        only = probe_tag(t)
        return {
            f"{tag}_track": (SPEED if tag == only else 0.0) for tag, _ in SPEC.FLIPPERS
        }
    forward = -1.0 if phase == "back" else 1.0
    out = {}
    for tag, _ in SPEC.FLIPPERS:
        v = SPEED * forward * WHEEL_SIGN[tag]
        if phase == "turn" and SIDE[tag] == "R":
            v = -v
        out[f"{tag}_track"] = v
    return out

# the rear subassemblies are mirrored, so their sprockets turn the other way
WHEEL_SIGN = SPEC.WHEEL_SIGN
SIDE = SPEC.SIDE


def P(*a):
    print("RUN", *a, flush=True)


def make_camera(stage_path, position, target):
    """A diagnostic camera; works headless, which is the point."""
    try:
        from isaacsim.sensors.camera import Camera
    except ImportError:
        from omni.isaac.sensor import Camera  # Isaac Sim <= 4.2
    import numpy as _np

    d = _np.asarray(target, dtype=float) - _np.asarray(position, dtype=float)
    yaw = math.atan2(d[1], d[0])
    pitch = math.atan2(d[2], math.hypot(d[0], d[1]))
    # Isaac cameras look down their local -Z with +Y up; this orientation is the
    # usual "look at" for the (roll, pitch, yaw) convention it accepts
    from isaacsim.core.utils.rotations import euler_angles_to_quat

    quat = euler_angles_to_quat(_np.array([0.0, -pitch, yaw]))
    cam = Camera(
        prim_path=stage_path,
        position=_np.asarray(position, dtype=float),
        orientation=quat,
        resolution=(960, 540),
    )
    cam.initialize()
    return cam


def save_frame(cam, path):
    rgba = cam.get_rgba()
    if rgba is None or rgba.size == 0:
        return False
    try:
        from PIL import Image
    except ImportError:
        np.save(path.replace(".png", ".npy"), rgba)
        return True
    Image.fromarray(rgba[:, :, :3].astype("uint8")).save(path)
    return True


sim = SimulationContext(physics_dt=1 / HZ, rendering_dt=1 / 60.0, stage_units_in_meters=1.0)
stage = get_current_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.Xform.Define(stage, "/World")
UsdLux.DistantLight.Define(stage, "/World/light").CreateIntensityAttr(2500.0)

scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
scene.CreateGravityMagnitudeAttr().Set(9.81)
px = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
px.CreateSolverTypeAttr("TGS")
px.CreateTimeStepsPerSecondAttr(int(HZ))

mat = UsdShade.Material.Define(stage, "/World/gmat")
m = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
m.CreateStaticFrictionAttr().Set(GROUND_MU)
m.CreateDynamicFrictionAttr().Set(GROUND_MU)
g = UsdGeom.Cube.Define(stage, "/World/ground")
g.CreateSizeAttr(1.0)
gx = UsdGeom.Xformable(g)
gx.ClearXformOpOrder()
gx.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.1))
gx.AddScaleOp().Set(Gf.Vec3f(80, 80, 0.2))
UsdPhysics.CollisionAPI.Apply(g.GetPrim())
UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(
    mat, UsdShade.Tokens.weakerThanDescendants, "physics"
)

P("loading", ASSET)
add_reference_to_stage(ASSET, ROOT)
rx = UsdGeom.Xformable(stage.GetPrimAtPath(ROOT))
rx.ClearXformOpOrder()
rx.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.35))
P("referenced ok")

sim.reset()
P("physics reset ok")

Art = get_articulation_cls()

# PhysX merges the whole Talon into ONE articulation (the flipper subassemblies
# are tied to base_link by fixed joints), so one view covers all four belts.
view = Art(prim_paths_expr=f"{ROOT}/{SPEC.BODY}/base_link", name="talon")
try:
    view.initialize()
except TypeError:
    view.initialize(sim.physics_sim_view)
names = list(view.dof_names)
P(f"articulation has {len(names)} dofs")
P("sprockets:", [n for n in names if "large_wheel" in n])

handles = [SPEC.make_flipper_handle(ROOT, tag) for tag, _ in SPEC.FLIPPERS]
driver = TrackDriver(view, handles)
driver.reset()
driver.rewind_enabled = REWIND

if os.environ.get("PROBE_ROOT", "0") == "1":
    _probe_state = {}

    def _root_vel():
        for getter in ("get_velocities", "get_angular_velocities"):
            fn = getattr(view, getter, None)
            if fn is None:
                continue
            try:
                v = np.asarray(fn())
            except Exception:
                continue
            if v is None or v.size == 0:
                continue
            return v[0][-3:]  # angular part
        return None

    def _probe(when):
        v = _root_vel()
        if v is None:
            return
        if when == "before":
            _probe_state["b"] = v.copy()
        else:
            b = _probe_state.get("b")
            if b is not None:
                P(f"        REWIND root angvel: before=({b[0]:+.4f},{b[1]:+.4f},{b[2]:+.4f}) "
                  f"after=({v[0]:+.4f},{v[1]:+.4f},{v[2]:+.4f})")

    driver.on_rewind_probe = _probe
driver.register()
P("rewind enabled =", REWIND)
P("one TrackDriver over", len(handles), "belts")

base = view

if not HEADLESS:
    # NOTE: the module is "viewports", plural.  Aiming the camera is a nicety,
    # so a miss here must not take the run down with it.
    try:
        try:
            from isaacsim.core.utils.viewports import set_camera_view
        except ImportError:
            from omni.isaac.core.utils.viewports import set_camera_view  # <= 4.2
        set_camera_view(eye=[2.5, -2.5, 1.6], target=[0.2, 0.2, 0.2])
        P("viewport camera aimed at the robot")
    except Exception as exc:  # noqa: BLE001 - cosmetic only
        P(f"could not aim the viewport camera ({exc}); drive the view by hand")

cam = None
if CAPTURE:
    os.makedirs(CAPTURE, exist_ok=True)
    _cp = os.environ.get("CAM_POS", "2.2,-2.2,1.5")
    _ct = os.environ.get("CAM_TGT", "0.3,0.2,0.2")
    cam = make_camera(
        "/World/diag_cam",
        position=[float(v) for v in _cp.split(",")],
        target=[float(v) for v in _ct.split(",")],
    )
    P("capturing frames to", CAPTURE)

# Hold the flipper arms.  Without a position target they are free to swing, and
# in a turn the opposing belt thrusts spin the arms instead of the vehicle.
ARM_DOFS = np.array([names.index(f"{t}_link_joint") for t, _ in SPEC.FLIPPERS], dtype=np.int32)
ARM_TARGET = np.array(
    [[math.radians(ARM_DEG) * SPEC.FLIPPER_SIGN[t] for t, _ in SPEC.FLIPPERS]]
)
P(f"arm dofs: {[names[i] for i in ARM_DOFS]}  held at {ARM_DEG:+.1f} deg")


# The bottom straight run is the part that touches the ground.  Its world
# velocity, minus the body's, is the belt surface velocity -- the thing that
# actually decides which way a flipper pushes.  Pure kinematics, so it answers
# the sign question without friction or the other belts confusing it.
bottom_view = None
if MODE in ("belt", "turn", "pivot"):
    try:
        from isaacsim.core.prims import RigidPrim
    except ImportError:
        from omni.isaac.core.prims import RigidPrimView as RigidPrim
    # one view per flipper: a joined path expression is not accepted, and this
    # also keeps the order aligned with SPEC.FLIPPERS
    # index 1 = idler arc, 2 = bottom straight run, 3 = drive arc
    SEG_LABEL = {1: "idlerArc", 2: "bottomRun", 3: "driveArc"}
    bottom_view = []
    seg_views = {}
    for tag, _ in SPEC.FLIPPERS:
        h = SPEC.make_flipper_handle(ROOT, tag)
        for idx in (1, 2, 3):
            v = RigidPrim(
                prim_paths_expr=h.segments[idx].link_path,
                name=f"seg{idx}_{tag}",
                track_contact_forces=True,
            )
            try:
                v.initialize()
            except TypeError:
                v.initialize(sim.physics_sim_view)
            seg_views[(tag, idx)] = v
        bottom_view.append(seg_views[(tag, 2)])
    P("bottom-run links:", [t for t, _ in SPEC.FLIPPERS])

if WAIT_FOR_PLAY and not HEADLESS:
    # The scene and the physics views are already built; pausing keeps those
    # handles valid, where a full stop would invalidate them.  app.update()
    # keeps the UI alive without advancing physics.
    import omni.timeline

    timeline = omni.timeline.get_timeline_interface()
    sim.pause()
    P("ready - press Play in Isaac Sim to start the run")
    while app.is_running() and not timeline.is_playing():
        app.update()
    if not app.is_running():
        app.close()
        raise SystemExit(0)
    P("play pressed - running")

dt = 1.0 / HZ
steps = int(SECONDS / dt)
log_every = max(1, int(0.5 * HZ))
prev = None
prev_yaw = 0.0


def _yaw(q):
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi

for i in range(steps):
    phase = phase_at(i * dt)
    cmd = command_for(phase, i * dt)
    # Everything below needs a playing timeline.  Pressing Stop in the GUI makes
    # the tensor API hand back None, so idle instead of erroring out.
    if view.get_joint_positions() is None:
        sim.step(render=not HEADLESS)
        continue
    driver.set_track_speeds(cmd)
    view.set_joint_position_targets(ARM_TARGET, joint_indices=ARM_DOFS)
    # With a viewport open we must render every step or the window stays blank.
    # Headless runs only render on the frames they capture.
    sim.step(render=(not HEADLESS) or (bool(CAPTURE) and i % log_every == 0))
    if cam is not None and i % log_every == 0:
        # the RTX denoiser needs a few frames before the image is readable
        for _ in range(6):
            sim.render()
        save_frame(cam, os.path.join(CAPTURE, f"f{i:05d}.png"))
    _p, _q = base.get_world_poses()
    if _p is None:
        continue
    pos = np.asarray(_p)[0]
    yaw = _yaw(np.asarray(_q)[0])
    if prev is not None and i % log_every == 0:
        win = log_every * dt
        v = float(np.linalg.norm((pos - prev)[:2])) / win
        # Body-independent: a prismatic belt joint's velocity IS the belt surface
        # speed relative to its own flipper, and the sprocket velocity times its
        # radius is what the driver commands from.  Neither depends on how the
        # chassis happens to be moving, unlike a world-frame difference.
        qd = view.get_joint_velocities()
        if qd is not None:
            parts = []
            for tag, _ in SPEC.FLIPPERS:
                spr = float(qd[0, names.index(f"{tag}_large_wheel_joint")]) * SPEC.DRIVE_R
                belt = float(qd[0, names.index(f"{tag}_track_straight_segment1_joint")])
                parts.append(f"{tag}: sprkt={spr:+.3f} belt={belt:+.3f}")
            P("        " + "   ".join(parts) + f"   m/s  (|cmd|={abs(SPEED):.3f})")
        if phase == "belt" and bottom_view is not None:
            body_v = (pos - prev) / win
            for k, (tag, _) in enumerate(SPEC.FLIPPERS):
                lin_k = np.asarray(bottom_view[k].get_linear_velocities())[0]
                rel = lin_k[:2] - body_v[:2]
                P(f"t={i*dt:5.2f}s [belt {tag}] cmd=+{SPEED}  "
                  f"bottom-run world v=({lin_k[0]:+.3f},{lin_k[1]:+.3f})  "
                  f"relative to body=({rel[0]:+.3f},{rel[1]:+.3f}) m/s  "
                  f"-> pushes {'+X' if rel[0] < 0 else '-X'}")
        elif phase == "probe":
            dx = float(pos[0] - prev[0]) / win
            dy = float(pos[1] - prev[1]) / win
            P(f"t={i*dt:5.2f}s [probe {probe_tag(i*dt)}] cmd=+{SPEED} "
              f"-> body dx={dx:+.3f} dy={dy:+.3f} m/s  "
              f"(dx>0 means +belt pushes the robot FORWARD)")
        elif phase in ("turn", "pivot"):
            yaw_rate = math.degrees(_wrap(yaw - prev_yaw)) / win
            ideal = math.degrees(2.0 * SPEED / SPEC.TRACK_WIDTH)
            P(f"t={i*dt:5.2f}s [{phase:5s}] yaw_rate={yaw_rate:7.2f} deg/s  "
              f"efficiency={abs(yaw_rate)/ideal:5.3f}  drift={v:5.3f} m/s  "
              f"pos=({pos[0]:+.3f},{pos[1]:+.3f})")
            q_raw = np.asarray(_q)[0]
            P(f"        raw root quat (w,x,y,z)=({q_raw[0]:+.5f},{q_raw[1]:+.5f},"
              f"{q_raw[2]:+.5f},{q_raw[3]:+.5f})  yaw={math.degrees(yaw):+.4f} deg")
            if bottom_view is not None:
                body_v = (pos - prev) / win
                parts = []
                for k, (tag, _) in enumerate(SPEC.FLIPPERS):
                    lin_k = np.asarray(bottom_view[k].get_linear_velocities())[0]
                    parts.append(f"{tag}={lin_k[0] - body_v[0]:+.3f}")
                P("        belt surface vX rel. body: " + "  ".join(parts)
                  + "   (negative pushes +X)")
                P("        commanded: "
                  + "  ".join(f"{t}={cmd[f'{t}_track']:+.2f}" for t, _ in SPEC.FLIPPERS))
                # is each belt actually on the ground?  the bottom-run link frame
                # rides on the belt surface, grousers hang ~0.012 below it, so a
                # touching belt sits at z ~ +0.012
                zs = []
                for k, (tag, _) in enumerate(SPEC.FLIPPERS):
                    zk = float(np.asarray(bottom_view[k].get_world_poses()[0])[0][2])
                    zs.append(f"{tag}={zk:+.4f}")
                P("        bottom-run world z: " + "  ".join(zs)
                  + "   (touching ground ~= +0.012)")
                # net contact force finally says which belts carry load and
                # which are merely near the ground
                fs = []
                for k, (tag, _) in enumerate(SPEC.FLIPPERS):
                    try:
                        f = np.asarray(bottom_view[k].get_net_contact_forces())[0]
                        fs.append(f"{tag}=|{np.linalg.norm(f):6.1f}|N "
                                  f"fx={f[0]:+7.1f} fz={f[2]:+7.1f}")
                    except Exception as exc:  # noqa: BLE001
                        fs.append(f"{tag}=n/a({type(exc).__name__})")
                P("        bottom-run contact force: " + "   ".join(fs))
                for idx in (1, 2, 3):
                    row = []
                    tot = 0.0
                    for tag, _ in SPEC.FLIPPERS:
                        try:
                            f = np.asarray(seg_views[(tag, idx)].get_net_contact_forces())[0]
                            tot += float(f[2])
                            row.append(f"{tag}={float(f[2]):7.1f}")
                        except Exception:  # noqa: BLE001
                            row.append(f"{tag}=n/a")
                    P(f"        Fz {SEG_LABEL[idx]:9s}: " + "  ".join(row)
                      + f"   total={tot:7.1f} N")
        else:
            slip = 1.0 - v / abs(SPEED) if SPEED else float("nan")
            P(f"t={i*dt:5.2f}s [{phase:5s}] v_body={v:6.3f} m/s  slip={slip:6.3f}  "
              f"pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f})")
            qq = view.get_joint_positions()
            if qq is not None:
                seg = [n for n in names if n.endswith("_straight_segment1_joint")]
                vals = [float(qq[0, names.index(n)]) for n in seg]
                lim = handles[0].geometry.pitch
                P("        bottom-run joint pos: "
                  + "  ".join(f"{n.split('_')[0]}={x:+.4f}" for n, x in zip(seg, vals))
                  + f"   (limit +/-{lim:.4f} m)")
    if i % log_every == 0:
        prev = pos
        prev_yaw = yaw

# Drive targets persist once the loop ends, so without this the belts keep
# turning for as long as the viewer is open.
STOP = {f"{t}_track": 0.0 for t, _ in SPEC.FLIPPERS}


def hold_still():
    driver.set_track_speeds(STOP)
    view.set_joint_position_targets(ARM_TARGET, joint_indices=ARM_DOFS)


hold_still()
for _ in range(int(0.5 * HZ)):  # let the sprockets spin down
    hold_still()
    sim.step(render=False)
P("done - belts stopped")

if not HEADLESS:
    P("run finished - close the Isaac Sim window to exit")
    while app.is_running():
        hold_still()
        sim.step(render=True)

app.close()
