"""The ported track algorithm, with no USD or Isaac dependency.

Keeping ``FillSegmentLength()`` and the element-distribution loop of
``ComposeSegments()`` in a pxr-free module means the part of the port that
actually carries the original's behaviour can be exercised in plain Python --
see ``tests/test_track_geometry.py``.  :mod:`isaac_continuous_track.track_builder`
consumes what is computed here and turns it into prims.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List

from .math_utils import Pose, child_pose_offset, dist_point_to_line
from .track_config import SegmentCfg, TrackCfg

__all__ = [
    "SegmentGeometry",
    "TrackGeometry",
    "GrouserPlacement",
    "SegmentHandle",
    "TrackHandle",
    "fill_segment_length",
    "iter_grouser_placements",
]

_EPS = 1e-9


@dataclass
class SegmentGeometry:
    """Per-segment scalars of ``Track::Belt::Segment``."""

    joint_to_track: float  # 1.0 if translational, rotation radius if rotational
    length: float  # length of the segment along the track


@dataclass
class TrackGeometry:
    """Scalars of ``Track::Belt``."""

    segments: List[SegmentGeometry]
    perimeter: float
    elements_per_round: int

    @property
    def pitch(self) -> float:
        """``len_per_element``: spacing between two adjacent grousers."""
        return self.perimeter / self.elements_per_round


@dataclass
class GrouserPlacement:
    """One pattern element, ready to be authored on its segment link."""

    segment_index: int
    element_index: int
    len_traveled: float  # distance from the start of the segment
    pose: Pose  # in the segment link frame at joint position 0


def fill_segment_length(cfg: TrackCfg) -> TrackGeometry:
    """Port of ``ContinuousTrack::FillSegmentLength()`` (hpp:256).

    The original measured the rotation radius in world coordinates as the
    distance from the child link origin to the joint axis line.  Distance is
    frame invariant, so the same quantity is computed here in the child link
    frame, where the child origin is the origin and the joint axis line passes
    through ``seg.joint_pos`` along ``seg.axis``.
    """
    segments: List[SegmentGeometry] = []
    perimeter = 0.0

    for seg in cfg.segments:
        if seg.joint_type == "revolute":
            radius = dist_point_to_line((0.0, 0.0, 0.0), seg.joint_pos, seg.axis)
            if radius < _EPS:
                raise ValueError(
                    f"segment {seg.name!r}: the joint axis passes through the child link "
                    "origin, so the segment has zero radius"
                )
            geom = SegmentGeometry(joint_to_track=radius, length=radius * seg.end_position)
        elif seg.joint_type == "prismatic":
            geom = SegmentGeometry(joint_to_track=1.0, length=seg.end_position)
        else:  # pragma: no cover - TrackCfg.validate() rejects this earlier
            raise ValueError(f"unexpected joint type {seg.joint_type!r}")

        segments.append(geom)
        perimeter += geom.length

    return TrackGeometry(
        segments=segments, perimeter=perimeter, elements_per_round=cfg.elements_per_round
    )


def iter_grouser_placements(
    cfg: TrackCfg, geometry: TrackGeometry
) -> Iterator[GrouserPlacement]:
    """Port of the element-distribution loop in ``ComposeSegments()`` (hpp:106).

    ``len_step`` / ``len_left`` / ``len_traveled`` keep the exact meaning they
    had in the original: ``len_left`` is how much of the current segment is
    still available, ``len_traveled`` how far along it the next element goes,
    and both carry over into the following segment so that the spacing is
    uniform across segment boundaries.

    The one simplification is that a single grouser shape is used, so the
    variant loop and its ``elem_id`` bookkeeping are gone (plan section 3.2).
    """
    element_local = Pose.from_rpy(cfg.grouser.resolved_pos(), cfg.grouser.rpy)

    len_step = geometry.pitch
    len_left = 0.0
    len_traveled = 0.0
    elem_count = 0

    for segment_index, (seg, seg_geom) in enumerate(zip(cfg.segments, geometry.segments)):
        len_left += seg_geom.length

        while len_left >= 0.0 and elem_count < cfg.elements_per_round:
            # ComputeChildPoseOffset(joint, 0, len_traveled / joint_to_track):
            # where the child link would sit if the joint had moved that far
            base_pose = child_pose_offset(
                seg.joint_type, seg.axis, seg.joint_pos, len_traveled / seg_geom.joint_to_track
            )
            # matches `pose_elem->Set(pose_offset + base_pose)` in the original
            yield GrouserPlacement(
                segment_index=segment_index,
                element_index=elem_count,
                len_traveled=len_traveled,
                pose=base_pose * element_local,
            )

            len_left -= len_step
            len_traveled += len_step
            elem_count += 1

        len_traveled -= seg_geom.length

    if elem_count != cfg.elements_per_round:  # pragma: no cover - guards bad configs
        raise RuntimeError(
            f"placed {elem_count} grousers but elements_per_round is "
            f"{cfg.elements_per_round}; check the segment end_positions"
        )


@dataclass
class SegmentHandle:
    cfg: SegmentCfg
    geometry: SegmentGeometry
    link_path: str
    joint_path: str

    @property
    def joint_name(self) -> str:
        return self.joint_path.rsplit("/", 1)[-1]


@dataclass
class TrackHandle:
    """Everything :class:`~isaac_continuous_track.track_driver.TrackDriver` needs."""

    name: str
    cfg: TrackCfg
    geometry: TrackGeometry
    segments: List[SegmentHandle]
    scope_path: str
    element_count: int
    belt_group_path: str
    body_group_path: str

    @property
    def sprocket_joint_name(self) -> str:
        return self.cfg.sprocket.joint_name

    @property
    def sprocket_joint_to_track(self) -> float:
        return self.cfg.sprocket.joint_to_track


def segment_zero_poses(cfg: TrackCfg) -> List[Pose]:
    """Segment link poses at joint position 0, in the track origin frame."""
    return [Pose.from_rpy(seg.zero_pos, seg.zero_rpy) for seg in cfg.segments]
