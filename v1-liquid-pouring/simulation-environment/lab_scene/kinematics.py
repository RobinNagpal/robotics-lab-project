"""Inverse kinematics for the pinch point between the fingers, by damped least squares.

Only a helper for tests and demos: it moves one arm's gripper to a point,
with the fingers pointing a given way (straight down unless told otherwise).
The turn of the gripper about that direction is left free.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from .scene import ARM_JOINTS, arm_actuator_ids, arm_joint_ids, pinch_site

DOWN = np.array([0.0, 0.0, -1.0])
# Closer than this counts as reached.
TOLERANCE = 1e-4
# The largest change in joint angles per search step, radians. Bigger steps
# near an awkward pose send the search spinning through many full turns.
MAX_STEP = 0.2


def solve(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target: np.ndarray,
    arm: str = "arm1",
    iterations: int = 200,
    pointing: np.ndarray = DOWN,
) -> np.ndarray:
    """Joint angles for ``arm`` that put its pinch point at ``target``, fingers along ``pointing``.

    Starts from the arm's current angles. If that does not reach, it starts
    again with the shoulder already turned towards the target: from a pose
    facing the other way the search can stall halfway round.
    """
    joints = arm_joint_ids(model, arm)
    start = data.qpos[joints].copy()
    q, miss = _search(model, data, target, arm, start, iterations, pointing)
    if miss > TOLERANCE:
        site = data.site(pinch_site(arm)).xpos
        base = data.body(f"{arm}/base").xpos
        turn = math.atan2(*(target - base)[1::-1]) - math.atan2(*(site - base)[1::-1])
        turned = start.copy()
        turned[0] += turn
        q2, miss2 = _search(model, data, target, arm, turned, iterations, pointing)
        if miss2 < miss:
            q = q2
    return q


def _search(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target: np.ndarray,
    arm: str,
    start: np.ndarray,
    iterations: int,
    pointing: np.ndarray,
) -> tuple[np.ndarray, float]:
    pointing = pointing / np.linalg.norm(pointing)
    work = mujoco.MjData(model)
    work.qpos[:] = data.qpos
    joints = arm_joint_ids(model, arm)
    dofs = [model.joint(f"{arm}/{name}").dofadr[0] for name in ARM_JOINTS]
    work.qpos[joints] = start
    site = model.site(pinch_site(arm)).id
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    for _ in range(iterations):
        mujoco.mj_kinematics(model, work)
        mujoco.mj_comPos(model, work)
        # The site's z axis runs along the fingers.
        axis = work.site_xmat[site].reshape(3, 3)[:, 2]
        error = np.concatenate([target - work.site_xpos[site], np.cross(axis, pointing)])
        if np.linalg.norm(error) < 1e-5:
            break
        mujoco.mj_jacSite(model, work, jacp, jacr, site)
        jac = np.vstack([jacp, jacr])[:, dofs]
        step = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), error)
        work.qpos[joints] += step * min(1.0, MAX_STEP / np.abs(step).max())
    # The same pose with each joint the fewest whole turns from where it started.
    q = work.qpos[joints]
    q -= 2 * np.pi * np.round((q - start) / (2 * np.pi))
    work.qpos[joints] = q
    mujoco.mj_kinematics(model, work)
    axis = work.site_xmat[site].reshape(3, 3)[:, 2]
    # Pointing the opposite way also zeroes the cross product; that is a miss.
    miss = float(np.linalg.norm(target - work.site_xpos[site])) + float(np.linalg.norm(axis - pointing))
    limits = np.array([model.jnt_range[model.joint(f"{arm}/{name}").id] for name in ARM_JOINTS])
    if np.any(q < limits[:, 0]) or np.any(q > limits[:, 1]):
        miss = np.inf
    return q.copy(), miss


def move_to(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target: np.ndarray,
    arm: str = "arm1",
    seconds: float = 1.0,
    settle: float = 0.3,
    pointing: np.ndarray = DOWN,
) -> None:
    """Move ``arm``'s pinch point to ``target`` by sliding its joint targets there over ``seconds``.

    Jumping the targets would swing the arm the shortest way in joint
    angles, which can sweep it low through whatever stands in between.
    """
    actuators = arm_actuator_ids(model, arm)
    start = data.ctrl[actuators].copy()
    goal = solve(model, data, target, arm, pointing=pointing)
    steps = int(seconds / model.opt.timestep)
    for i in range(1, steps + 1):
        data.ctrl[actuators] = start + (goal - start) * i / steps
        mujoco.mj_step(model, data)
    for _ in range(int(settle / model.opt.timestep)):
        mujoco.mj_step(model, data)
