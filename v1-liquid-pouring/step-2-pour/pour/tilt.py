"""Where Arm B's joints go to hold the source tipped by some angle with its lip at a given point.

The source is tipped about the y axis (the line across the row of racks), towards the receiver,
which stands on its -x side. The lip is the point of the source's rim nearest the receiver; tipped
over, it is the rim's lowest point, where the balls leave. The tube is assumed to stay in the
fingers as Step 1b left it (the pads' friction 1.2 holds it: docs/solutions/step-2-pour-approaches.md).
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
from lab_scene import scene as lab

ARM = "arm1"  # Arm B, the source arm (carry.episode.SOURCE_ARM)
# Where the lip goes for the pour: this far from the receiver's axis towards the source, and this
# far above its rim. Nearer the centre the balls overshoot; lower, the two grippers collide.
LIP_OUT, LIP_UP = 0.015, 0.020


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


class Tilter:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, source: str, height: float, radius: float):
        self.model = model
        self.site = model.site(lab.pinch_site(ARM)).id
        self.joints = lab.arm_joint_ids(model, ARM)
        self.dofs = [model.joint(f"{ARM}/{n}").dofadr[0] for n in lab.ARM_JOINTS]
        self.limits = np.array([model.jnt_range[model.joint(f"{ARM}/{n}").id] for n in lab.ARM_JOINTS])
        body = data.body(source)
        self.R0 = body.xmat.reshape(3, 3).copy()  # the tube as held, upright
        site_R = data.site_xmat[self.site].reshape(3, 3)
        # The pinch point in the tube's frame: fixed while the tube stays put in the fingers.
        self.site_pos = self.R0.T @ (data.site_xpos[self.site] - body.xpos)
        self.site_R = self.R0.T @ site_R
        self.start_R = self.site_R.copy()
        # The rim point facing the receiver (-x), in the tube's frame.
        towards = self.R0.T @ np.array([-1.0, 0.0, 0.0])
        towards[2] = 0.0
        towards /= np.linalg.norm(towards)
        self.lip_local = radius * towards + np.array([0.0, 0.0, height])
        self.work = mujoco.MjData(model)

    def regrip(self, data: mujoco.MjData, source: str, share: float = 1.0) -> None:
        """Take where the tube sits in the fingers now (``share`` of the way from the last estimate).

        Tipped, a full large tube turns in the fingers by up to about 10 degrees: its weight, far out
        from the pads, pushes them apart. Aiming with the tube's real place in the fingers keeps the
        tube itself, not the gripper, on the planned tilt and the lip in place.
        """
        body = data.body(source)
        R = body.xmat.reshape(3, 3)
        site_R = data.site_xmat[self.site].reshape(3, 3)
        pos = R.T @ (data.site_xpos[self.site] - body.xpos)
        rot = R.T @ site_R
        self.site_pos = self.site_pos + share * (pos - self.site_pos)
        blended = self.site_R + share * (rot - self.site_R)
        u, _, vt = np.linalg.svd(blended)  # back to a rotation
        self.site_R = u @ vt

    def turned(self, data: mujoco.MjData, source: str) -> float:
        """How far the tube has turned in the fingers since Step 1b left it, degrees."""
        body = data.body(source)
        rot = body.xmat.reshape(3, 3).T @ data.site_xmat[self.site].reshape(3, 3)
        return math.degrees(math.acos(np.clip((np.trace(rot @ self.start_R.T) - 1) / 2, -1, 1)))

    def lip(self, data: mujoco.MjData, source: str) -> np.ndarray:
        body = data.body(source)
        return body.xpos + body.xmat.reshape(3, 3) @ self.lip_local

    def tube_pose(self, tilt_deg: float, lip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        R = rot_y(-math.radians(tilt_deg)) @ self.R0
        return lip - R @ self.lip_local, R

    def solve(self, data: mujoco.MjData, tilt_deg: float, lip: np.ndarray, start: np.ndarray, iterations=60):
        """Arm B's joint angles for the tilt and lip, searched from ``start``. Returns (q, miss in m)."""
        p, R = self.tube_pose(tilt_deg, lip)
        goal_R = R @ self.site_R
        goal_p = p + R @ self.site_pos
        w = self.work
        w.qpos[:] = data.qpos
        w.qpos[self.joints] = start
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        quat, rv = np.zeros(4), np.zeros(3)
        for _ in range(iterations):
            mujoco.mj_kinematics(self.model, w)
            mujoco.mj_comPos(self.model, w)
            mujoco.mju_mat2Quat(quat, (goal_R @ w.site_xmat[self.site].reshape(3, 3).T).flatten())
            mujoco.mju_quat2Vel(rv, quat, 1.0)
            err = np.concatenate([goal_p - w.site_xpos[self.site], 0.05 * rv])
            if np.linalg.norm(err) < 2e-5:
                break
            mujoco.mj_jacSite(self.model, w, jacp, jacr, self.site)
            J = np.vstack([jacp, 0.05 * jacr])[:, self.dofs]
            step = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), err)
            w.qpos[self.joints] += step * min(1.0, 0.2 / max(np.abs(step).max(), 1e-12))
        mujoco.mj_kinematics(self.model, w)
        q = w.qpos[self.joints].copy()
        miss = float(np.linalg.norm(goal_p - w.site_xpos[self.site]))
        miss += 0.05 * math.acos(
            np.clip((np.trace(goal_R @ w.site_xmat[self.site].reshape(3, 3).T) - 1) / 2, -1, 1)
        )
        if np.any(q < self.limits[:, 0]) or np.any(q > self.limits[:, 1]):
            miss = math.inf
        return q, miss

    def clashes(
        self,
        data: mujoco.MjData,
        q: np.ndarray,
        tilt_deg: float,
        lip: np.ndarray,
        source: str,
        others: tuple[str, ...],
    ) -> list[tuple[str, str]]:
        """What Arm B or the source would touch among ``others`` (name prefixes), set there, no physics."""
        w = mujoco.MjData(self.model)
        w.qpos[:] = data.qpos
        w.qpos[self.joints] = q
        p, R = self.tube_pose(tilt_deg, lip)
        adr = self.model.jnt_qposadr[self.model.body(source).jntadr[0]]
        w.qpos[adr : adr + 3] = p
        mujoco.mju_mat2Quat(w.qpos[adr + 3 : adr + 7], R.flatten())
        mujoco.mj_forward(self.model, w)
        out = []
        for c in w.contact[: w.ncon]:
            a = self.model.body(self.model.geom_bodyid[c.geom1]).name
            b = self.model.body(self.model.geom_bodyid[c.geom2]).name
            for x, y in ((a, b), (b, a)):
                mine = x.startswith(ARM) or x == source
                if mine and y.startswith(others) and c.dist < -0.0005:
                    out.append((x, y))
        return out
