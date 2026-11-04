"""Moving a held tube, not the gripper: the tube kept upright, the gripper at whatever angle that needs.

After the pour a full tube has turned in the fingers, up to about 19 degrees (its weight, far out from the
pads, pushes them apart while it is tipped). Moved with the fingers level, as Step 1b moves tubes, it
would lean that much and could not go down into a rack cell 30 mm wide. So every move here is planned
for the tube: where its bottom goes and that its axis stays vertical. The gripper's pose follows from
where the tube sits in the fingers now (measured), then the arm's joints by inverse kinematics, every
5 mm along the way, as in Step 1b (carry.motion) and Step 2 (pour.tilt).
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
from carry import motion
from lab_scene import scene as lab

PIECE = 0.005  # m between solved points along a line
TURN_PIECE = 0.05  # rad between solved points when the tube turns
MAX_JUMP = 0.5  # rad, the most any joint may change between two points 5 mm apart (as grasp.side)
REACHED = 0.003  # m (and 0.05 x rad): further than this, the arm cannot get there


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def log_rot(R: np.ndarray) -> np.ndarray:
    """The rotation as one turn about an axis: axis x angle."""
    angle = math.acos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if angle < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v / (2 * math.sin(angle)) * angle


def exp_rot(v: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3)
    k = v / angle
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def upright(R: np.ndarray) -> np.ndarray:
    """The tube's frame turned the shortest way to stand straight up."""
    z = R[:, 2]
    axis = np.cross(z, [0.0, 0.0, 1.0])
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return R.copy()
    angle = math.atan2(s, z[2])
    return exp_rot(axis / s * angle) @ R


def lean(data: mujoco.MjData, tube: str) -> float:
    """How far the tube's axis is from vertical, degrees."""
    return math.degrees(math.acos(np.clip(data.body(tube).xmat.reshape(3, 3)[2, 2], -1, 1)))


class Mover:
    """Joint paths for one arm that take its tube along straight lines, upright."""

    def __init__(self, model: mujoco.MjModel, arm: str):
        self.model, self.arm = model, arm
        self.site = model.site(lab.pinch_site(arm)).id
        self.joints = lab.arm_joint_ids(model, arm)
        self.dofs = [model.joint(f"{arm}/{n}").dofadr[0] for n in lab.ARM_JOINTS]
        self.limits = np.array([model.jnt_range[model.joint(f"{arm}/{n}").id] for n in lab.ARM_JOINTS])

    def grip(self, data: mujoco.MjData, tube: str) -> tuple[np.ndarray, np.ndarray]:
        """Where the pinch point sits in the tube's frame now (position, rotation)."""
        body = data.body(tube)
        R = body.xmat.reshape(3, 3)
        return R.T @ (data.site_xpos[self.site] - body.xpos), R.T @ data.site_xmat[self.site].reshape(3, 3)

    def solve(self, work: mujoco.MjData, p: np.ndarray, R: np.ndarray, iterations: int = 80):
        """The joints that put the pinch site at ``p`` turned ``R``, from the joints in ``work``."""
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        quat, rv = np.zeros(4), np.zeros(3)
        for _ in range(iterations):
            mujoco.mj_kinematics(self.model, work)
            mujoco.mj_comPos(self.model, work)
            mujoco.mju_mat2Quat(quat, (R @ work.site_xmat[self.site].reshape(3, 3).T).flatten())
            mujoco.mju_quat2Vel(rv, quat, 1.0)
            err = np.concatenate([p - work.site_xpos[self.site], 0.05 * rv])
            if np.linalg.norm(err) < 2e-5:
                break
            mujoco.mj_jacSite(self.model, work, jacp, jacr, self.site)
            J = np.vstack([jacp, 0.05 * jacr])[:, self.dofs]
            step = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), err)
            work.qpos[self.joints] += step * min(1.0, 0.2 / max(np.abs(step).max(), 1e-12))
        mujoco.mj_kinematics(self.model, work)
        q = work.qpos[self.joints].copy()
        angle = math.acos(np.clip((np.trace(R @ work.site_xmat[self.site].reshape(3, 3).T) - 1) / 2, -1, 1))
        miss = float(np.linalg.norm(p - work.site_xpos[self.site])) + 0.05 * angle
        if np.any(q < self.limits[:, 0]) or np.any(q > self.limits[:, 1]):
            miss = math.inf
        return q, miss

    def line(
        self,
        data: mujoco.MjData,
        grip: tuple[np.ndarray, np.ndarray],
        b0: np.ndarray,
        R0: np.ndarray,
        b1: np.ndarray,
        R1: np.ndarray,
    ) -> list[np.ndarray]:
        """Joints every 5 mm (and every 3 degrees of turn) taking the tube's bottom from b0 to b1 and its
        frame from R0 to R1. The arm starts as it is in ``data``. Raises motion.Blocked if it cannot."""
        pos, rot = grip
        turn = log_rot(R1 @ R0.T)
        steps = max(
            1, math.ceil(np.linalg.norm(b1 - b0) / PIECE), math.ceil(np.linalg.norm(turn) / TURN_PIECE)
        )
        work = mujoco.MjData(self.model)
        work.qpos[:] = data.qpos
        out = []
        for i in range(1, steps + 1):
            s = i / steps
            R = exp_rot(turn * s) @ R0
            b = b0 + (b1 - b0) * s
            before = work.qpos[self.joints].copy()
            q, miss = self.solve(work, b + R @ pos, R @ rot)
            if miss > REACHED:
                raise motion.Blocked(f"{self.arm} cannot hold its tube with the bottom at {np.round(b, 3)}")
            if np.abs(q - before).max() > MAX_JUMP:
                raise motion.Blocked(
                    f"{self.arm} would have to swing a joint by {np.abs(q - before).max():.1f} rad"
                )
            out.append(q)
        return out

    def hand_line(self, data: mujoco.MjData, p1: np.ndarray) -> list[np.ndarray]:
        """Empty-handed: the pinch point in a straight line to ``p1``, the gripper keeping its turn."""
        R = data.site_xmat[self.site].reshape(3, 3).copy()
        p0 = data.site_xpos[self.site].copy()
        steps = max(1, math.ceil(np.linalg.norm(p1 - p0) / PIECE))
        work = mujoco.MjData(self.model)
        work.qpos[:] = data.qpos
        out = []
        for i in range(1, steps + 1):
            before = work.qpos[self.joints].copy()
            q, miss = self.solve(work, p0 + (p1 - p0) * i / steps, R)
            if miss > REACHED or np.abs(q - before).max() > MAX_JUMP:
                raise motion.Blocked(f"{self.arm} cannot move its open hand to {np.round(p1, 3)}")
            out.append(q)
        return out
