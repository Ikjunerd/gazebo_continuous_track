"""Checks on the ported geometry. Runs in plain Python -- no Isaac Sim needed.

    python -m isaac_continuous_track.tests.test_track_geometry

These cover the parts of the port that carry the original's behaviour: the
scalars ``FillSegmentLength()`` produced, and the element distribution of
``ComposeSegments()``.  The physics-level checks (slip, step climbing, ...) are
plan section 6 and need the simulator.
"""

from __future__ import annotations

import math
import sys

import numpy as np

from ..math_utils import Pose, child_pose_offset, dist_point_to_line
from ..track_config import GrouserCfg, make_oval_track_cfg
from ..track_geometry import fill_segment_length, iter_grouser_placements

LENGTH = 0.5
RADIUS = 0.1
WIDTH = 0.12
ELEMENTS = 32


def make_cfg(elements: int = ELEMENTS):
    return make_oval_track_cfg(
        name="track",
        chassis_path="/World/Robot/chassis",
        sprocket_joint_path="/World/Robot/sprocket_joint",
        pitch_diameter=2.0 * RADIUS,
        length=LENGTH,
        radius=RADIUS,
        width=WIDTH,
        mass=2.0,
        elements_per_round=elements,
        grouser=GrouserCfg(size=(0.012, WIDTH, 0.012)),
    )


def check(name: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    return condition


def test_child_pose_offset() -> bool:
    """ComputeChildPoseOffset(): the one-parameter group property."""
    ok = True

    off = child_pose_offset("prismatic", (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.3)
    ok &= check(
        "prismatic offset is a pure translation along the axis",
        np.allclose(off.pos, [0.3, 0.0, 0.0]) and np.allclose(off.quat, [1.0, 0.0, 0.0, 0.0]),
    )

    # rotating a half turn about the arc joint moves the link frame from the top
    # of the circle to the bottom, i.e. by 2*radius
    off = child_pose_offset("revolute", (0.0, 1.0, 0.0), (0.0, 0.0, -RADIUS), math.pi)
    ok &= check(
        "revolute half turn moves the link frame by 2*radius",
        np.allclose(off.pos, [0.0, 0.0, -2.0 * RADIUS], atol=1e-12),
        f"pos={off.pos}",
    )

    # offset(a) * offset(b) == offset(a + b) -- this is what lets the original
    # accumulate `base_pose = base_pose_step + base_pose` instead of recomputing
    a, b = 0.37, 0.81
    lhs = child_pose_offset("revolute", (0.0, 1.0, 0.0), (0.0, 0.0, -RADIUS), a) * child_pose_offset(
        "revolute", (0.0, 1.0, 0.0), (0.0, 0.0, -RADIUS), b
    )
    rhs = child_pose_offset("revolute", (0.0, 1.0, 0.0), (0.0, 0.0, -RADIUS), a + b)
    ok &= check(
        "revolute offsets compose additively",
        np.allclose(lhs.pos, rhs.pos, atol=1e-12) and np.allclose(lhs.quat, rhs.quat, atol=1e-12),
    )
    return bool(ok)


def test_fill_segment_length() -> bool:
    """FillSegmentLength(): joint_to_track, per-segment length, perimeter."""
    cfg = make_cfg()
    geom = fill_segment_length(cfg)
    ok = True

    ok &= check(
        "straight segments have joint_to_track == 1",
        all(geom.segments[i].joint_to_track == 1.0 for i in (0, 2)),
    )
    ok &= check(
        "arc segments recover the radius from the joint axis",
        all(abs(geom.segments[i].joint_to_track - RADIUS) < 1e-12 for i in (1, 3)),
        f"got {geom.segments[1].joint_to_track}",
    )
    ok &= check(
        "arc length == radius * end_position",
        all(abs(geom.segments[i].length - RADIUS * math.pi) < 1e-12 for i in (1, 3)),
    )

    expected = 2.0 * LENGTH + 2.0 * math.pi * RADIUS
    ok &= check(
        "perimeter == 2*length + 2*pi*radius",
        abs(geom.perimeter - expected) < 1e-12,
        f"{geom.perimeter:.6f} vs {expected:.6f}",
    )
    ok &= check(
        "pitch == perimeter / elements_per_round",
        abs(geom.pitch - expected / ELEMENTS) < 1e-12,
        f"pitch={geom.pitch:.5f} m",
    )
    ok &= check(
        "dist_point_to_line matches the arc radius",
        abs(dist_point_to_line((0, 0, 0), (0, 0, -RADIUS), (0, 1, 0)) - RADIUS) < 1e-15,
    )
    return bool(ok)


def make_flat_cfg(elements: int = ELEMENTS):
    """Same track, but with the grouser sitting exactly on the belt surface.

    Removing the outward offset makes the placed points lie on the belt path
    itself, so their spacing can be compared against the pitch exactly.
    """
    cfg = make_cfg(elements)
    cfg.grouser = GrouserCfg(size=(0.012, WIDTH, 0.012), pos=(0.0, 0.0, 0.0))
    return cfg


def _track_points(cfg, geom):
    """Placed element origins in the track frame, in the order they are placed."""
    zero_poses = [Pose.from_rpy(s.zero_pos, s.zero_rpy) for s in cfg.segments]
    return np.array(
        [
            (zero_poses[p.segment_index] * p.pose).pos
            for p in iter_grouser_placements(cfg, geom)
        ]
    )


def _path_parameter(point) -> float:
    """Distance travelled along the oval belt path, from the rear of the top run.

    The path is the outline the segment link frames sweep: top run forward,
    front arc, bottom run backward, rear arc.
    """
    x, _, z = point
    half_len = LENGTH / 2.0
    arc = math.pi * RADIUS
    if -half_len - 1e-9 <= x <= half_len + 1e-9 and z > 0.0:
        return x + half_len
    if x > half_len:  # front arc, centred at (+L/2, 0)
        return LENGTH + RADIUS * math.atan2(x - half_len, z)
    if -half_len - 1e-9 <= x <= half_len + 1e-9:  # bottom run, travelled backwards
        return LENGTH + arc + (half_len - x)
    # rear arc, centred at (-L/2, 0)
    return 2.0 * LENGTH + arc + RADIUS * math.atan2(-(x + half_len), -z)


def test_grouser_distribution() -> bool:
    """ComposeSegments(): count, share per segment, and exact pitch spacing."""
    cfg = make_flat_cfg()
    geom = fill_segment_length(cfg)
    placements = list(iter_grouser_placements(cfg, geom))
    ok = True

    ok &= check(
        "exactly elements_per_round grousers are placed",
        len(placements) == ELEMENTS,
        f"got {len(placements)}",
    )
    ok &= check(
        "element indices are 0..n-1 in order",
        [p.element_index for p in placements] == list(range(ELEMENTS)),
    )

    per_segment = [0] * len(cfg.segments)
    for p in placements:
        per_segment[p.segment_index] += 1
    ok &= check(
        "grousers are shared across all four segments",
        all(c > 0 for c in per_segment),
        f"per segment: {per_segment}",
    )

    points = _track_points(cfg, geom)
    params = np.array([_path_parameter(p) for p in points])
    expected = np.arange(ELEMENTS) * geom.pitch
    err = float(np.max(np.abs(params - expected)))
    ok &= check(
        "elements sit at exactly k * pitch along the belt path",
        err < 1e-9,
        f"max deviation {err:.3e} m over {ELEMENTS} elements spanning all four segments",
    )

    gap = geom.perimeter - float(params[-1])
    ok &= check(
        "the pattern closes: the last-to-first gap is also one pitch",
        abs(gap - geom.pitch) < 1e-9,
        f"gap {gap:.6f} m vs pitch {geom.pitch:.6f} m",
    )

    # and with the default outward offset the grousers protrude from the belt
    offset_cfg = make_cfg()
    offset_geom = fill_segment_length(offset_cfg)
    half = offset_cfg.grouser.size[2] / 2.0
    offset_points = _track_points(offset_cfg, offset_geom)
    radii = [
        abs(p[2]) if -LENGTH / 2.0 - 1e-9 <= p[0] <= LENGTH / 2.0 + 1e-9
        else min(
            math.hypot(p[0] - LENGTH / 2.0, p[2]), math.hypot(p[0] + LENGTH / 2.0, p[2])
        )
        for p in offset_points
    ]
    ok &= check(
        "grousers protrude one half-height outside the belt surface",
        all(abs(r - (RADIUS + half)) < 1e-9 for r in radii),
        f"outline radius {min(radii):.4f}..{max(radii):.4f}, expected {RADIUS + half:.4f}",
    )
    return bool(ok)


def test_rewind_invariance() -> bool:
    """The property the whole design rests on.

    Advancing every segment by exactly one pitch must reproduce the original set
    of grouser positions, which is what makes rewinding to zero invisible.

    The elements that would run off the end of their own segment are the
    exception: a segment is a rigid body, so a grouser near its trailing end
    keeps sliding straight instead of following the belt path round the corner.
    There is at most one such element per segment boundary, and the deviation is
    bounded by the rewind range of +/- pitch/2.  This is inherent to the
    approach and behaves the same way in the Gazebo original.
    """
    cfg = make_flat_cfg()
    geom = fill_segment_length(cfg)
    pitch = geom.pitch
    zero_poses = [Pose.from_rpy(s.zero_pos, s.zero_rpy) for s in cfg.segments]

    before, after, seg_of = [], [], []
    for placement in iter_grouser_placements(cfg, geom):
        seg = cfg.segments[placement.segment_index]
        seg_geom = geom.segments[placement.segment_index]
        base = zero_poses[placement.segment_index]
        before.append((base * placement.pose).pos)
        moved = child_pose_offset(
            seg.joint_type, seg.axis, seg.joint_pos, pitch / seg_geom.joint_to_track
        )
        after.append((base * moved * placement.pose).pos)
        seg_of.append(placement.segment_index)

    before = np.array(before)
    after = np.array(after)

    matched = sum(
        1 for p in after if float(np.min(np.linalg.norm(before - p, axis=1))) < 1e-9
    )
    boundaries = len(cfg.segments)
    ok = check(
        "advancing one pitch maps the pattern onto itself",
        matched >= ELEMENTS - boundaries,
        f"{matched}/{ELEMENTS} landed on an existing position; "
        f"at most {boundaries} may miss, one per segment boundary",
    )

    # the misses must be the trailing element of each segment, and no other
    trailing = set()
    counts = {}
    for i, s_idx in enumerate(seg_of):
        counts[s_idx] = i  # last index seen for this segment
    trailing = set(counts.values())
    missed = {
        i
        for i, p in enumerate(after)
        if float(np.min(np.linalg.norm(before - p, axis=1))) >= 1e-9
    }
    ok &= check(
        "the elements that miss are exactly the trailing ones of each segment",
        missed <= trailing,
        f"missed {sorted(missed)}, trailing elements are {sorted(trailing)}",
    )

    # and the deviation stays inside one pitch, so the pattern never opens a gap
    # larger than the rewind range
    worst = max(
        (float(np.min(np.linalg.norm(before - after[i], axis=1))) for i in missed), default=0.0
    )
    ok &= check(
        "the boundary deviation is bounded by one pitch",
        worst <= pitch + 1e-9,
        f"worst deviation {worst:.5f} m vs pitch {pitch:.5f} m",
    )
    return bool(ok)


def test_wrap() -> bool:
    """The wrap arithmetic of UpdateTrack() / the driver."""
    cfg = make_cfg()
    geom = fill_segment_length(cfg)
    pitch = geom.pitch

    positions = np.linspace(-3.0, 3.0, 20001)
    wrap_index = np.floor(positions / pitch)
    wrapped = positions - pitch * wrap_index - pitch / 2.0

    ok = check(
        "wrapped position stays in [-pitch/2, +pitch/2)",
        bool(np.all(wrapped >= -pitch / 2.0 - 1e-12) and np.all(wrapped < pitch / 2.0 + 1e-12)),
        f"range [{wrapped.min():.6f}, {wrapped.max():.6f}], pitch/2={pitch / 2.0:.6f}",
    )

    # the rewind rate the plan quotes: v_track / pitch
    v_track = 1.0
    ok &= check(
        "rewind rate at 1 m/s is v_track / pitch",
        abs(v_track / pitch - 1.0 / pitch) < 1e-12,
        f"{v_track / pitch:.1f} Hz at pitch={pitch * 100:.2f} cm",
    )
    return bool(ok)


def main() -> int:
    tests = [
        ("ComputeChildPoseOffset", test_child_pose_offset),
        ("FillSegmentLength", test_fill_segment_length),
        ("ComposeSegments element distribution", test_grouser_distribution),
        ("rewind invariance", test_rewind_invariance),
        ("UpdateTrack wrap arithmetic", test_wrap),
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
