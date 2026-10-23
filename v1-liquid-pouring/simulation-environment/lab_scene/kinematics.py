"""Inverse kinematics for the pinch point between the fingers, by damped least squares.

Only a helper for tests and demos: it moves the gripper to a point, pointing
straight down.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .scene import PINCH_SITE, arm_joint_ids

DOWN = np.array([0.0, 0.0, -1.0])


def solve(
    model: mujoco.MjModel, data: mujoco.MjData, target: np.ndarray, iterations: int = 200
) -> np.ndarray:
    """Arm joint angles that put the pinch point at ``target`` with the fingers pointing down."""
    work = mujoco.MjData(model)
    work.qpos[:] = data.qpos
    joints = arm_joint_ids(model)
    site = model.site(PINCH_SITE).id
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    for _ in range(iterations):
        mujoco.mj_kinematics(model, work)
        mujoco.mj_comPos(model, work)
        pos = work.site_xpos[site]
        # The site's z axis runs along the fingers.
        axis = work.site_xmat[site].reshape(3, 3)[:, 2]
        error = np.concatenate([target - pos, np.cross(axis, DOWN)])
        if np.linalg.norm(error) < 1e-5:
            break
        mujoco.mj_jacSite(model, work, jacp, jacr, site)
        jac = np.vstack([jacp, jacr])[:, joints]
        step = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), error)
        work.qpos[joints] += step
    return work.qpos[joints].copy()
