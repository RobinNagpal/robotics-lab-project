"""Poses and the ALOHA arm's exact inverse kinematics, ported from AutoBio to plain numpy.

AutoBio's ``kinematics.py`` (``Pose``, ``slerp``, ``AlohaAnalyticalIK``) and ``aloha_analytical_ik.py``
import JAX, SciPy and SymPy for other solvers; the analytical one needs only numpy, so it is copied here
step by step with the same numbers. "Analytical" means it solves the arm's angles by formula (the ViperX
300s has a spherical wrist: its last three joints meet at one point), not by searching.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Pose:
    pos: np.ndarray
    quat: np.ndarray  # w, x, y, z


def quatcompose(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quatinv(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quatapply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, xyz = q[0], np.asarray(q[1:])
    t = 2 * np.cross(xyz, v)
    return np.asarray(v) + w * t + np.cross(xyz, t)


def slerp(q0: np.ndarray, q1: np.ndarray, amount: float = 0.5) -> np.ndarray:
    """AutoBio's slerp, unchanged: the shorter way round, linear when nearly equal."""
    amount = np.clip(amount, 0, 1)
    dot = np.dot(q0, q1)
    if dot < 0.0:
        q0 = -q0
        dot = -dot
    if dot > 0.9995:
        qr = q0 + amount * (q1 - q0)
        return qr / np.linalg.norm(qr)
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * amount
    sin_theta = np.sin(theta)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    qr = s0 * q0 + s1 * q1
    return qr / np.linalg.norm(qr)


# --- aloha_analytical_ik.py -----------------------------------------------------------------------------


def _close_to_zero(x: float, tol: float = 1e-8) -> bool:
    return -tol < x < tol


def _site_pose_to_wrist_pos(site_pos, site_quat):
    # The gripper site (left/gripper) to the wrist centre (left/wrist), along the hand.
    local_axis = np.array([0.9998872305243953, 0.0, -0.015017530897414616])
    length = 0.1997665275665571
    return site_pos - quatapply(site_quat, local_axis) * length


def _solve_linear_trig(A, B, C):
    # A sin(x) + B cos(x) + C = 0
    R = np.sqrt(A**2 + B**2)
    theta = np.arctan2(B, A)
    t = -C / R
    if np.abs(t) > 1:
        return None
    inv = np.arcsin(t)
    return np.array((inv - theta, np.pi - inv - theta))


def _solve_planar_2link(u, v):
    a, b = 0.3, 0.05955
    A = -2 * a * u + 2 * b * v
    B = -2 * a * v - 2 * b * u
    C = b**2 + u**2 + v**2
    q1 = _solve_linear_trig(A, B, C)
    if q1 is None:
        return None
    cos_q12 = (u - a * np.sin(q1) - b * np.cos(q1)) / a
    sin_q12 = -(v - a * np.cos(q1) + b * np.sin(q1)) / a
    q2 = np.arctan2(sin_q12, cos_q12) - q1
    return np.stack((q1, q2), axis=-1)


def _leftpad(arr, v):
    v = np.atleast_1d(v)
    m, n = arr.shape
    res = np.empty((m, n + len(v)))
    res[:, : len(v)] = v
    res[:, len(v) :] = arr
    return res


def _solve_base(wrist_pos):
    c = 0.12705
    x, y, z = wrist_pos
    v = z - c
    if _close_to_zero(x) and _close_to_zero(y):
        raise NotImplementedError  # wrist straight above the base: the waist angle is undefined
    sln = []
    q0 = np.arctan2(y, x)
    u = np.sqrt(x**2 + y**2)
    q12 = _solve_planar_2link(u, v)
    if q12 is not None:
        sln.append(_leftpad(q12, q0))
    q12 = _solve_planar_2link(-u, v)
    if q12 is not None:
        sln.append(_leftpad(q12, q0 + np.pi))
    if not sln:
        return None
    return np.concatenate(sln, axis=0)


def _target_quat_in_wrist_frame(target_quat, q_base):
    q0, q1, q2 = q_base[..., 0], q_base[..., 1], q_base[..., 2]
    c0, s0 = np.cos(q0 / 2), np.sin(q0 / 2)
    c12, s12 = np.cos(q1 / 2 + q2 / 2), np.sin(q1 / 2 + q2 / 2)
    wrist_quat = np.stack((c0 * c12, -s0 * s12, c0 * s12, s0 * c12), axis=-1)
    return np.array([quatcompose(quatinv(q), target_quat) for q in wrist_quat])


def _solve_wrist(target_quat):
    w, x, y, z = target_quat
    c4_sqr = w**2 + x**2
    p35 = np.arctan2(x, w)
    if _close_to_zero(1 - c4_sqr):
        return np.array(((0.0, 0.0, p35 * 2),))  # gimbal lock
    q4 = np.arccos(np.clip(2 * c4_sqr - 1, -1.0, 1.0))
    m35 = np.arctan2(z, y)
    return np.array(((p35 + m35, q4, p35 - m35), (p35 + m35 + np.pi, -q4, p35 - m35 - np.pi)))


_BOUNDS = np.array(
    [
        [-3.14158, 3.14158],
        [-1.85005, 1.25664],
        [-1.76278, 1.6057],
        [-3.14158, 3.14158],
        [-1.8675, 2.23402],
        [-6.28, 6.28],
    ]
)


def aloha_analytical_ik(target_pos, target_quat):
    """Every joint solution (rows of 6 angles) putting the gripper site at the pose, in the arm's base
    frame, within the joint limits."""
    q_base = _solve_base(_site_pose_to_wrist_pos(target_pos, target_quat))
    if q_base is None:
        return None
    local = _target_quat_in_wrist_frame(target_quat, q_base)
    sln = np.concatenate([_leftpad(_solve_wrist(quat), q) for q, quat in zip(q_base, local, strict=True)])
    sln = (sln + np.pi) % (2 * np.pi) - np.pi
    return sln[np.all((sln >= _BOUNDS[:, 0]) & (sln <= _BOUNDS[:, 1]), axis=1)]


class AlohaAnalyticalIK:
    """AutoBio's class: a pose in the world is turned into the arm's base frame, solved, and the solution
    nearest the last one is kept (so a path of poses gives a smooth path of angles)."""

    def __init__(self, pose: Pose):
        self.pos = np.asarray(pose.pos, dtype=float).copy()
        self.quat = np.asarray(pose.quat, dtype=float).copy()
        self.initial_qpos = np.zeros(6)

    def solve(self, target_pos: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
        target_pos = quatapply(quatinv(self.quat), np.asarray(target_pos) - self.pos)
        target_quat = quatcompose(quatinv(self.quat), np.asarray(target_quat))
        try:
            solutions = aloha_analytical_ik(target_pos, target_quat)
        except NotImplementedError as e:
            raise ValueError(f"IK edge case at {target_pos.tolist()}") from e
        if solutions is None or len(solutions) == 0:
            raise ValueError(f"No IK solution at {target_pos.tolist()}, {target_quat.tolist()}")
        solution = solutions[np.argmin(np.linalg.norm(solutions - self.initial_qpos, axis=1))]
        self.initial_qpos = solution
        return solution
