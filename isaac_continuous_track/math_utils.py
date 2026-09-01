"""Minimal pose/quaternion math used by the track builder.

This module is a port of the ignition::math pieces the original Gazebo plugin
relied on, so that the geometric behaviour of the port can be checked against
the C++ source line by line:

  * ``Pose3d::operator*``          -> :meth:`Pose.__mul__`
  * ``Vector3d::DistToLine()``     -> :func:`dist_point_to_line`
  * ``ComputeChildPoseOffset()``   -> :func:`child_pose_offset`
    (gazebo_continuous_track.hpp:333)

Quaternions are stored as ``(w, x, y, z)`` and Euler angles use the SDF/URDF
convention ``R = Rz(yaw) * Ry(pitch) * Rx(roll)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = [
    "Pose",
    "quat_identity",
    "quat_from_axis_angle",
    "quat_from_rpy",
    "quat_mul",
    "quat_rotate",
    "quat_inverse",
    "dist_point_to_line",
    "child_pose_offset",
]


def _vec3(v: Sequence[float]) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64).reshape(3)
    return a


def quat_identity() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0])


def quat_from_axis_angle(axis: Sequence[float], angle: float) -> np.ndarray:
    a = _vec3(axis)
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        return quat_identity()
    a = a / n
    h = 0.5 * angle
    s = math.sin(h)
    return np.array([math.cos(h), a[0] * s, a[1] * s, a[2] * s])


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """SDF <pose> rotation: Rz(yaw) * Ry(pitch) * Rx(roll)."""
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a * b`` (rotation ``a`` applied after ``b``)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_inverse(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    return np.array([w, -x, -y, -z]) / n


def quat_rotate(q: np.ndarray, v: Sequence[float]) -> np.ndarray:
    w, x, y, z = q
    u = np.array([x, y, z])
    vv = _vec3(v)
    return vv + 2.0 * np.cross(u, np.cross(u, vv) + w * vv)


@dataclass
class Pose:
    """Rigid transform. ``Pose(a) * Pose(b)`` expresses ``b`` in ``a``'s parent."""

    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quat: np.ndarray = field(default_factory=quat_identity)

    def __post_init__(self) -> None:
        self.pos = _vec3(self.pos)
        self.quat = np.asarray(self.quat, dtype=np.float64).reshape(4)

    @staticmethod
    def from_rpy(pos: Sequence[float], rpy: Sequence[float]) -> "Pose":
        r, p, y = rpy
        return Pose(_vec3(pos), quat_from_rpy(r, p, y))

    def __mul__(self, other: "Pose") -> "Pose":
        return Pose(self.pos + quat_rotate(self.quat, other.pos), quat_mul(self.quat, other.quat))

    def inverse(self) -> "Pose":
        qi = quat_inverse(self.quat)
        return Pose(-quat_rotate(qi, self.pos), qi)


def dist_point_to_line(
    point: Sequence[float], line_point: Sequence[float], line_dir: Sequence[float]
) -> float:
    """Port of ``ignition::math::Vector3d::DistToLine()``.

    Used by :func:`~isaac_continuous_track.track_builder.fill_segment_length` to
    recover the rotation radius of an arc segment, exactly as
    ``FillSegmentLength()`` does (gazebo_continuous_track.hpp:256).
    """
    p = _vec3(point)
    a = _vec3(line_point)
    d = _vec3(line_dir)
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        return float(np.linalg.norm(p - a))
    return float(np.linalg.norm(np.cross(p - a, d / n)))


def child_pose_offset(
    joint_type: str, axis: Sequence[float], anchor: Sequence[float], position: float
) -> Pose:
    """Pose change of a joint's child link when the joint moves from 0 to ``position``.

    Port of ``ContinuousTrack::ComputeChildPoseOffset()``, which evaluates
    ``ChildLinkPose(to) - ChildLinkPose(from)`` and therefore yields a transform
    expressed in the child link frame at joint position 0.

    ``axis`` and ``anchor`` are the joint axis and the joint origin, both given
    in the child link frame (SDF ``<axis>`` with ``use_parent_model_frame`` 0,
    and SDF ``<joint><pose>``).
    """
    a = _vec3(axis)
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        raise ValueError("joint axis must be non-zero")
    a = a / n

    if joint_type == "prismatic":
        return Pose(a * position, quat_identity())

    if joint_type == "revolute":
        rot = quat_from_axis_angle(a, position)
        anc = _vec3(anchor)
        # rotation about the line through `anchor` along `axis`
        return Pose(anc - quat_rotate(rot, anc), rot)

    raise ValueError(f"unsupported joint type: {joint_type!r}")
