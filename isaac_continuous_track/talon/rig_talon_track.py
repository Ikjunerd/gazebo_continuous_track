r"""Re-rig the Talon so its four flippers run real continuous tracks.

    C:\isaac-sim-5.1.0\python.bat isaac_continuous_track\talon\rig_talon_track.py
    ... --src <in.usd> --dst <out.usd> --elements 36

Reads ``talon_p_v4_0c.usd`` and writes a new asset; the source is never touched.

What the source asset is
------------------------
The Talon has no track geometry at all.  What touches the ground is, per
flipper, one large wheel + one small wheel + 13 rollers -- 60 point contacts for
the whole 83 kg machine.  Driving it directly slips 74-88 % and will not turn in
place.  The previous workaround (``tools/rig_talon_track.py`` in
multi_robot_control) laid flat pads along the contact line and gave them
``PhysxSurfaceVelocityAPI``: cheap and stable, but the belt is not simulated, so
grousers cannot bite into terrain and turning needed a 1/0.135 fudge factor.

What this script does instead
-----------------------------
Each flipper gets a real continuous track from
:mod:`isaac_continuous_track` -- four segment links sliding by at most one
grouser pitch and rewound past it, with grousers as actual colliders.

The flipper belt is *tapered*, not an oval: a 0.179 m drive sprocket at the
pivot and a smaller idler at the tip.  ``make_two_pulley_track_cfg`` builds that
shape.  The idler radius is derived rather than measured -- see IDLER_R below.

Removed on the way (the user confirmed these are shape-only):
  * the 13 roller links per flipper and their joints
  * the gear joints that chained rollers to the drive wheel
  * roller / wheel colliders, which would otherwise fight the belt

The wheels themselves are kept as visuals and are spun by the driver, so the
robot still looks right.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if _PKG_PARENT not in sys.path:
    sys.path.append(_PKG_PARENT)


def parse_args(argv):
    here = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default=os.path.join(here, "talon_p_v4_0c.usd"))
    p.add_argument("--dst", default=os.path.join(here, "talon_p_v4_0c_ctrack.usd"))
    p.add_argument("--elements", type=int, default=36, help="grousers per flipper belt")
    p.add_argument(
        "--grouser-height",
        type=float,
        default=0.01,
        help="how far a grouser stands off the belt surface. Keep it above "
        "--arc-margin, or the arc pulley discs swallow the grousers and the "
        "track meets the ground on bare cylinders at each end.",
    )
    p.add_argument("--belt-mass", type=float, default=3.0, help="per flipper belt, kg")
    p.add_argument("--keep-rollers", action="store_true", help="leave the roller links in place")
    p.add_argument(
        "--drive-arc-margin",
        type=float,
        default=0.015,
        help="how far the DRIVE pulley disc (arc_segment1, at the pivot) stands "
        "proud of the belt path (m). The disc is one prim doing both visual and "
        "collision, so this moves both. Keep it below --grouser-height or the "
        "disc swallows the grousers on that arc.",
    )
    p.add_argument(
        "--idler-arc-margin",
        type=float,
        default=0.01,
        help="same, for the IDLER pulley disc (arc_segment0, at the tip).",
    )
    p.add_argument(
        "--keep-idler-wheel",
        action="store_true",
        help="keep the small wheel link and its joint. By default they go: the "
        "belt's own idler arc replaces them, so once the wheel is hidden and "
        "its collider is off it is a dead body and a dead DOF.",
    )
    p.add_argument(
        "--hide-pulley-visuals",
        action="store_true",
        help="stop rendering the belt's own arc-segment cylinders, leaving only "
        "the straight runs and the grousers. They keep colliding either way -- "
        "they are the shape the belt rolls on at each end.",
    )
    p.add_argument(
        "--keep-original-visuals",
        action="store_true",
        help="keep rendering the original wheel meshes; by default they are "
        "hidden because they sit inside the new belt and show through it",
    )
    return p.parse_args(argv)


ARGS = parse_args(sys.argv[1:])

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True})

from pxr import Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

from isaac_continuous_track import build_track  # noqa: E402
from isaac_continuous_track.talon.talon_spec import (  # noqa: E402
    BODY,
    DRIVE_R,
    FLIPPERS,
    IDLER_OFFSET,
    IDLER_R,
    ROLLER_COUNT,
    flipper_paths,
    make_flipper_track_cfg,
)

BODY_PATH = f"/World/{BODY}"


def log(*a):
    print("[rig]", *a, flush=True)


def _rename_prim(stage, old_path: str, new_name: str) -> bool:
    """Rename a prim in place, keeping every attribute and relationship.

    The four flippers each ship a joint literally called ``large_wheel_joint``.
    PhysX merges the whole Talon into ONE articulation, so those duplicates
    collide and Isaac silently disambiguates them as ``large_wheel_joint``,
    ``large_wheel_joint_0``, ``_1``, ``_2`` -- in an order that is an
    implementation detail.  Looking a sprocket up by name then returns the wrong
    DOF and every flipper ends up driving the same wheel.  Giving each joint a
    unique name removes the ambiguity at the source.
    """
    layer = stage.GetRootLayer()
    old = Sdf.Path(old_path)
    new = old.GetParentPath().AppendChild(new_name)
    if not layer.GetPrimAtPath(old):
        return False
    Sdf.CopySpec(layer, old, layer, new)
    stage.RemovePrim(old)
    return True


JOINT_BODY_RELS = ("physics:body0", "physics:body1")


def _dangling_targets(stage):
    """Joints whose body0/body1 points at a prim that is not on the stage."""
    out = []
    for prim in stage.Traverse():
        for rel_name in JOINT_BODY_RELS:
            rel = prim.GetRelationship(rel_name)
            if not rel:
                continue
            for target in rel.GetTargets():
                if not stage.GetPrimAtPath(target):
                    out.append((prim.GetPath(), rel_name, target.pathString))
    return out


def _prune_dangling(stage):
    """Delete every joint left pointing at a removed body.

    Removing the roller links orphans the gear joints that drove them, and those
    live in their own scope rather than under the flipper, so they cannot be
    found by prim path.  Deciding on the dangling reference itself catches them
    wherever they are.
    """
    removed = 0
    while True:
        stale = _dangling_targets(stage)
        if not stale:
            return removed
        for path, _rel, _target in {(p, r, t) for p, r, t in stale}:
            if stage.GetPrimAtPath(path):
                stage.RemovePrim(path)
                removed += 1


def main() -> int:
    if not os.path.isfile(ARGS.src):
        log(f"source not found: {ARGS.src}")
        return 1

    stage = Usd.Stage.Open(ARGS.src)
    log(f"opened {ARGS.src}")

    removed_prims = 0
    disabled_colliders = 0
    renamed_joints = 0
    hidden_links = 0
    hidden_pulleys = 0
    handles = []

    for tag, _frag in FLIPPERS:
        p = flipper_paths("/World", tag)
        flip, vflip = p["flip"], p["vflip"]
        if not stage.GetPrimAtPath(vflip):
            log(f"  {tag}: {vflip} missing, skipped")
            continue

        # -- strip the roller chain ----------------------------------------
        if not ARGS.keep_rollers:
            for i in range(1, ROLLER_COUNT + 1):
                for path in (f"{flip}/roller_{i}_link", f"{flip}/joints/roller_{i}_joint"):
                    if stage.GetPrimAtPath(path):
                        stage.RemovePrim(Sdf.Path(path))
                        removed_prims += 1
            # The gear joints that chained the rollers do NOT live under the
            # flipper: they sit in their own scope at <body>/GearJoint/<tag>/.
            # Matching on the prim path misses them and leaves joints pointing
            # at prims that no longer exist, which corrupts the articulation
            # silently.  _prune_dangling() sweeps the whole stage instead.

        # -- drop the idler wheel ---------------------------------------------
        # The belt's own idler arc segment now provides that shape and its
        # collision.  Leaving the original behind costs a rigid body and a DOF
        # for nothing.  _prune_dangling() sweeps up whatever referenced it.
        if not ARGS.keep_idler_wheel:
            for path in (p["small_wheel"], p["idler_joint"]):
                if stage.GetPrimAtPath(path):
                    stage.RemovePrim(Sdf.Path(path))
                    removed_prims += 1

        # -- unique joint names, see _rename_prim() --------------------------
        for key, base in (("drive_joint", "large_wheel_joint"),
                          ("idler_joint", "small_wheel_joint")):
            if _rename_prim(stage, p[key], f"{tag}_{base}"):
                renamed_joints += 1

        # -- the original wheels: no collision, and by default no render -----
        # Their colliders would fight the belt, and their meshes sit *inside*
        # the new belt where they show through it.  The belt's own arc segments
        # are full cylinders of the same radii, so nothing is lost visually.
        # vflip_link is the flipper frame plate.  Its collider still touches the
        # ground, so it adds yaw resistance and drag while contributing no
        # traction -- the belt is what should carry the machine.
        # arm_link is the pivot arm; its collider spans the whole flipper
        # (bbox x 0.08..0.80) and holds the machine up, leaving the belt in the
        # air with no load and therefore no traction.  Measured: the twelve belt
        # segments together carried 2 N of an 814 N vehicle before this went.
        for key in ("vflip", "large_wheel", "small_wheel", "arm_link"):
            link = stage.GetPrimAtPath(p[key])
            if not link:
                continue
            for prim in Usd.PrimRange(link):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    prim.GetAttribute("physics:collisionEnabled").Set(False)
                    disabled_colliders += 1

        # -- hide the flipper's original appearance --------------------------
        # Every link carries TWO subtrees: "collisions" and "visuals".  The
        # "visuals" one is an *instance*, so Usd.Stage.Traverse() walks straight
        # past it -- it is invisible to a naive audit yet very much rendered,
        # and vflip_link/visuals is where the old belt and sprocket shapes live.
        # Hide each subtree explicitly rather than relying on the parent link.
        # The flipper arm link (FL_link and friends) spans the whole track too --
        # its bbox runs x 0.08..0.80 -- so it overlaps the new belt as well.
        # Everything in the track area goes; only base_link keeps its skin.
        if not ARGS.keep_original_visuals:
            for path in (p["vflip"], p["large_wheel"], p["small_wheel"], p["arm_link"]):
                for sub in ("visuals", "collisions"):
                    prim = stage.GetPrimAtPath(f"{path}/{sub}")
                    if prim:
                        UsdGeom.Imageable(prim).MakeInvisible()
                        hidden_links += 1

        # -- build the belt --------------------------------------------------
        cfg = make_flipper_track_cfg(
            "/World",
            tag,
            elements_per_round=ARGS.elements,
            grouser_height=ARGS.grouser_height,
            drive_arc_margin=ARGS.drive_arc_margin,
            idler_arc_margin=ARGS.idler_arc_margin,
            belt_mass=ARGS.belt_mass,
        )
        handle = build_track(
            stage,
            cfg,
            body_paths=[vflip, p["large_wheel"], p["small_wheel"]],
            collision_groups_scope=f"{BODY_PATH}/CollisionGroups",
        )
        if ARGS.hide_pulley_visuals:
            for seg in handle.segments:
                if seg.cfg.joint_type != "revolute":
                    continue
                prim = stage.GetPrimAtPath(f"{seg.link_path}/belt")
                if prim:
                    # render-only; the collider stays live
                    UsdGeom.Imageable(prim).MakeInvisible()
                    hidden_pulleys += 1

        handles.append((tag, handle))
        g = handle.geometry
        log(
            f"  {tag}: perimeter {g.perimeter:.4f} m  pitch {g.pitch * 100:.2f} cm  "
            f"grousers {handle.element_count}"
        )

    if not handles:
        log("no flipper was rigged - check the prim paths")
        return 1

    # -- exactly one articulation root ---------------------------------------
    # The asset declares ArticulationRootAPI five times: on base_link and on
    # each flipper's vflip_link.  PhysX wants exactly one per articulation, and
    # the flippers are tied to base_link by fixed joints, so they are all one
    # articulation anyway.  With the extra roots in place the merged articulation
    # translates but will not rotate at all -- a locked track can be dragged
    # sideways and the body still refuses to yaw.  Keep base_link, drop the rest.
    dropped_roots = 0
    for tag, _frag in FLIPPERS:
        prim = stage.GetPrimAtPath(flipper_paths("/World", tag)["vflip"])
        if prim and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            dropped_roots += 1
    log(f"dropped {dropped_roots} extra articulation roots (base_link keeps the only one)")

    removed_prims += _prune_dangling(stage)

    stale = _dangling_targets(stage)
    if stale:
        log(f"ABORT: {len(stale)} joint targets still dangle, e.g. {stale[:3]}")
        return 1

    stage.GetRootLayer().Export(ARGS.dst)
    log(f"removed {removed_prims} prims, disabled {disabled_colliders} wheel colliders, "
        f"renamed {renamed_joints} joints, hid {hidden_links} original visual subtrees "
        f"and {hidden_pulleys} pulley visuals")
    log("no dangling joint targets remain")
    log(f"wrote {ARGS.dst}")
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
