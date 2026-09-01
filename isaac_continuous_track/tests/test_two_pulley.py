"""Checks on the tapered (two-pulley) belt. Plain Python -- no Isaac Sim needed.

    python -m isaac_continuous_track.tests.test_two_pulley

The Talon flipper is the motivating case: a 0.179 m drive sprocket at the pivot
and a 0.109 m idler at the tip, so the belt is a trapezoid rather than an oval.
The invariant that matters is the same as for the oval -- grousers land at exact
multiples of the pitch all the way round and the pattern closes -- because that
is what makes the rewind invisible.
"""

from __future__ import annotations

import math
import sys

import numpy as np

from ..math_utils import Pose
from ..track_config import (
    GrouserCfg,
    make_oval_track_cfg,
    make_two_pulley_track_cfg,
    two_pulley_geometry,
)
from ..track_geometry import fill_segment_length, iter_grouser_placements
from ..talon import talon_spec as SPEC

# The geometry lives in one place -- talon_spec -- so editing a radius there is
# picked up here without touching this file.
DRIVE_R = SPEC.DRIVE_R
IDLER_R = SPEC.IDLER_R
IDLER_OFFSET = SPEC.IDLER_OFFSET
WIDTH = SPEC.BELT_WIDTH
ELEMENTS = 36

# The 13 rollers of the stock asset all sit at z = -0.11349 with radius 0.06553,
# so its belt bottom run is dead flat on their underside.
STOCK_ROLLER_UNDERSIDE = -0.11349 - 0.06553
STOCK_DRIVE_R = 0.17903


def check(name: str, cond: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    return bool(cond)


def make_cfg(elements: int = ELEMENTS, grouser_pos=None):
    return make_two_pulley_track_cfg(
        name="flip",
        chassis_path="/W/vflip_link",
        sprocket_joint_path="/W/joints/large_wheel_joint",
        drive_radius=DRIVE_R,
        idler_radius=IDLER_R,
        idler_offset=IDLER_OFFSET,
        width=WIDTH,
        mass=3.0,
        elements_per_round=elements,
        grouser=GrouserCfg(size=(0.012, WIDTH, 0.012), pos=grouser_pos),
    )


def test_closed_form() -> bool:
    """The analytic belt geometry."""
    g = two_pulley_geometry(DRIVE_R, IDLER_R, IDLER_OFFSET)
    ok = True
    d = math.hypot(*IDLER_OFFSET)
    ok &= check("centre distance", abs(g["d"] - d) < 1e-12, f"{g['d']:.5f} m")
    ok &= check(
        "alpha equals the centre-line tilt, i.e. the bottom run is flat",
        abs(g["alpha"] + g["theta"]) < 1e-9,
        f"alpha {math.degrees(g['alpha']):.4f} deg, tilt {math.degrees(g['theta']):.4f} deg",
    )
    ok &= check(
        "sin(alpha) == (R - r) / d",
        abs(math.sin(g["alpha"]) - (DRIVE_R - IDLER_R) / d) < 1e-12,
        f"alpha = {math.degrees(g['alpha']):.4f} deg",
    )
    ok &= check(
        "tangent length",
        abs(g["tangent"] - math.sqrt(d * d - (DRIVE_R - IDLER_R) ** 2)) < 1e-12,
        f"{g['tangent']:.5f} m",
    )
    ok &= check(
        "wrap angles sum to 2*pi",
        abs(g["wrap_drive"] + g["wrap_idler"] - 2.0 * math.pi) < 1e-12,
        f"{math.degrees(g['wrap_drive']):.3f} + {math.degrees(g['wrap_idler']):.3f} deg",
    )
    # cross-check the closed form against the length the segment builder derives
    # independently, via dist_point_to_line on each arc
    geom = fill_segment_length(make_cfg())
    ok &= check(
        "closed-form perimeter equals the sum of the built segment lengths",
        abs(g["perimeter"] - geom.perimeter) < 1e-9,
        f"{g['perimeter']:.5f} m vs {geom.perimeter:.5f} m",
    )
    return ok


def test_degenerates_to_oval() -> bool:
    """R == r must reproduce the symmetric oval the xacro macro built."""
    L, radius = 0.5, 0.1
    common = dict(
        chassis_path="/W/chassis",
        sprocket_joint_path="/W/j",
        width=0.12,
        mass=2.0,
        elements_per_round=32,
    )
    tapered = make_two_pulley_track_cfg(
        name="t", drive_radius=radius, idler_radius=radius, idler_offset=(L, 0.0), **common
    )
    oval = make_oval_track_cfg(
        name="t", pitch_diameter=2 * radius, length=L, radius=radius, **common
    )
    gt = fill_segment_length(tapered)
    go = fill_segment_length(oval)

    ok = check(
        "perimeter identical to the symmetric oval",
        abs(gt.perimeter - go.perimeter) < 1e-12,
        f"{gt.perimeter:.6f} vs {go.perimeter:.6f}",
    )
    ok &= check(
        "per-segment joint_to_track identical",
        all(
            abs(a.joint_to_track - b.joint_to_track) < 1e-12
            for a, b in zip(gt.segments, go.segments)
        ),
    )
    ok &= check(
        "per-segment length identical",
        all(abs(a.length - b.length) < 1e-12 for a, b in zip(gt.segments, go.segments)),
    )
    ok &= check(
        "same segment ordering: straight / arc / straight / arc",
        [s.joint_type for s in tapered.segments] == [s.joint_type for s in oval.segments],
    )
    return ok


def _distance_to_belt(point) -> float:
    """Gap between a point and the belt outline, in the track plane."""
    x, _, z = point
    g = two_pulley_geometry(DRIVE_R, IDLER_R, IDLER_OFFSET)
    cx, cz = IDLER_OFFSET
    best = min(abs(math.hypot(x, z) - DRIVE_R), abs(math.hypot(x - cx, z - cz) - IDLER_R))
    for phi in (g["phi_top"], g["phi_bottom"]):
        n = (math.cos(phi), math.sin(phi))
        a = (DRIVE_R * n[0], DRIVE_R * n[1])
        b = (cx + IDLER_R * n[0], cz + IDLER_R * n[1])
        abx, abz = b[0] - a[0], b[1] - a[1]
        t = ((x - a[0]) * abx + (z - a[1]) * abz) / (abx * abx + abz * abz)
        if 0.0 <= t <= 1.0:
            best = min(best, math.hypot(x - (a[0] + t * abx), z - (a[1] + t * abz)))
    return best


def test_grousers_lie_on_the_belt() -> bool:
    """Every element must sit exactly on the belt outline, across all four segments."""
    cfg = make_cfg(grouser_pos=(0.0, 0.0, 0.0))
    geom = fill_segment_length(cfg)
    zero = [Pose.from_rpy(s.zero_pos, s.zero_rpy) for s in cfg.segments]

    worst = 0.0
    per_segment = [0] * 4
    for p in iter_grouser_placements(cfg, geom):
        pos = (zero[p.segment_index] * p.pose).pos
        worst = max(worst, _distance_to_belt(pos))
        per_segment[p.segment_index] += 1

    ok = check("all elements sit on the belt outline", worst < 1e-9, f"worst gap {worst:.3e} m")
    ok &= check(
        "all four segments carry elements",
        all(c > 0 for c in per_segment),
        f"per segment {per_segment}, lengths {[round(s.length, 3) for s in geom.segments]}",
    )
    ok &= check(
        "perimeter is the four segment lengths, and pitch divides it evenly",
        abs(geom.perimeter - sum(sg.length for sg in geom.segments)) < 1e-12
        and abs(geom.pitch * ELEMENTS - geom.perimeter) < 1e-12,
        f"{geom.perimeter:.5f} m, pitch {geom.pitch * 100:.2f} cm",
    )
    return ok


def test_spacing_is_uniform() -> bool:
    """Consecutive elements are one pitch apart measured along the belt."""
    cfg = make_cfg(grouser_pos=(0.0, 0.0, 0.0))
    geom = fill_segment_length(cfg)
    zero = [Pose.from_rpy(s.zero_pos, s.zero_rpy) for s in cfg.segments]
    pts = np.array(
        [(zero[p.segment_index] * p.pose).pos for p in iter_grouser_placements(cfg, geom)]
    )
    pitch = geom.pitch

    # chord between neighbours: exactly the pitch on a straight run, and the
    # chord of a `pitch` arc on a pulley -- never longer than the pitch
    chords = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    min_chord = 2.0 * IDLER_R * math.sin(pitch / (2.0 * IDLER_R))  # tightest pulley
    ok = check(
        "neighbour spacing lies between the tightest arc chord and one pitch",
        bool(np.all(chords <= pitch + 1e-9) and np.all(chords >= min_chord - 1e-9)),
        f"chords {chords.min():.5f}..{chords.max():.5f}, allowed {min_chord:.5f}..{pitch:.5f}",
    )

    closing = float(np.linalg.norm(pts[0] - pts[-1]))
    ok &= check(
        "the pattern closes at one pitch",
        min_chord - 1e-9 <= closing <= pitch + 1e-9,
        f"closing gap {closing:.5f} m vs pitch {pitch:.5f} m",
    )
    return ok


def test_ground_line() -> bool:
    """The bottom run must land on the Talon's measured roller contact line."""
    g = two_pulley_geometry(DRIVE_R, IDLER_R, IDLER_OFFSET)
    phi = g["phi_bottom"]
    at_drive = DRIVE_R * math.sin(phi)
    at_idler = IDLER_OFFSET[1] + IDLER_R * math.sin(phi)

    # value-independent invariant: a flat bottom run sits exactly one drive
    # radius below the pulley centre, at both ends
    ok = check(
        "bottom run sits at -DRIVE_R at the drive end",
        abs(at_drive + DRIVE_R) < 1e-9,
        f"{at_drive:.5f} vs -DRIVE_R = {-DRIVE_R:.5f} m",
    )
    ok &= check(
        "bottom run sits at the same height at the idler end",
        abs(at_idler - at_drive) < 1e-9,
        f"{at_idler:.5f} m",
    )
    # informational: does the current spec still match the stock Talon asset?
    stock = abs(DRIVE_R - STOCK_DRIVE_R) < 1e-9
    if stock:
        ok &= check(
            "and that line is the stock asset's roller underside",
            abs(at_drive - STOCK_ROLLER_UNDERSIDE) < 2e-5,
            f"{at_drive:.5f} vs measured {STOCK_ROLLER_UNDERSIDE:.5f} m",
        )
    else:
        print(f"  ....  geometry differs from the stock asset "
              f"(DRIVE_R {DRIVE_R:.5f} vs {STOCK_DRIVE_R:.5f}); "
              "roller-underside check skipped")
    tilt = math.degrees(math.atan2(at_idler - at_drive, IDLER_OFFSET[0]))
    ok &= check(
        "so the bottom run is essentially flat",
        abs(at_drive - at_idler) < 1e-9,
        f"tilt {tilt:+.3f} deg",
    )
    return ok


def main() -> int:
    tests = [
        ("closed-form belt geometry", test_closed_form),
        ("degenerates to the symmetric oval", test_degenerates_to_oval),
        ("grousers lie on the belt", test_grousers_lie_on_the_belt),
        ("uniform spacing", test_spacing_is_uniform),
        ("ground contact line vs the measured Talon", test_ground_line),
    ]
    failed = []
    for name, fn in tests:
        print(f"\n{name}")
        if not fn():
            failed.append(name)
    print("\n" + "-" * 62)
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print(f"all {len(tests)} groups passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
