"""Run one grasp with full physics: the real arm goes down, closes, and lifts.

This is the final word on a grasp, and slow (about 10 s each), so it is
used on a sample of the test tubes only. It also checks that the fast check
(scenes.Checker) agrees with what really happens.

A grasp succeeds when the tube comes up at least 8 cm, stays within about
18 degrees of upright, and the arm touched nothing else on the way: the same
thing the fast check claims to foresee.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import mujoco
import numpy as np
from detect_classify import random_scene
from lab_scene import kinematics, liquid
from lab_scene import scene as lab

from . import rules, scenes

ROSE = 0.08
UPRIGHT = 0.95  # cosine of the tube's lean
HIGH = 0.35  # the arm comes over the tube from this high


def _open_angle(data: mujoco.MjData, arm: str) -> float:
    """The direction the arm's fingers open along, on the table, radians. The pinch site's y axis."""
    y = data.site(lab.pinch_site(arm)).xmat.reshape(3, 3)[:, 1]
    return math.atan2(y[1], y[0])


def _turn_error(a: float, b: float) -> float:
    """The smallest difference between two finger turns; half a turn apart is the same grasp."""
    return (a - b + math.pi / 2) % math.pi - math.pi / 2


def solve(model: mujoco.MjModel, data: mujoco.MjData, target: np.ndarray, arm: str, yaw: float) -> np.ndarray:
    """Joint angles with the pinch point at ``target``, fingers down and opening along ``yaw``.

    lab_scene.kinematics leaves the turn about the fingers free. The last
    wrist joint turns the gripper about that same line, so the turn is set
    afterwards by turning that joint alone, the shorter way.
    """
    q = kinematics.solve(model, data, target, arm)
    joints = lab.arm_joint_ids(model, arm)
    work = mujoco.MjData(model)
    work.qpos[:] = data.qpos
    work.qpos[joints] = q
    limits = model.jnt_range[model.joint(f"{arm}/{lab.ARM_JOINTS[5]}").id]
    best = None
    for change in np.arange(-2 * math.pi, 2 * math.pi + 1e-9, math.pi / 2):
        for sign in (1, -1):
            work.qpos[joints[5]] = q[5] + change
            mujoco.mj_kinematics(model, work)
            error = _turn_error(_open_angle(work, arm), yaw)
            fix = q[5] + change + sign * error
            work.qpos[joints[5]] = fix
            mujoco.mj_kinematics(model, work)
            left = abs(_turn_error(_open_angle(work, arm), yaw))
            if limits[0] <= fix <= limits[1] and left < 1e-3:
                cost = abs(fix - data.qpos[joints[5]])
                if best is None or cost < best[0]:
                    best = (cost, fix)
    if best is not None:
        q = q.copy()
        q[5] = best[1]
    return q


class Watch:
    """Steps the simulation and notes every object, other than the target, that the arm touches."""

    EVERY = 5  # steps between looks at the contacts

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, target: str):
        self.model, self.data = model, data
        names = [model.body(model.geom_bodyid[g]).name for g in range(model.ngeom)]
        self.is_arm = np.array([n.startswith(lab.ARM_NAMES) for n in names])
        # The table, the target and the liquid may be touched; anything else counts.
        allowed = {"world", target}
        self.counts = np.array([not (n in allowed or n.startswith(liquid.BALL_PREFIX)) for n in names])
        self.names = names
        self.touched: set[str] = set()
        self.steps = 0

    def step(self) -> None:
        mujoco.mj_step(self.model, self.data)
        self.steps += 1
        if self.steps % self.EVERY:
            return
        for c in self.data.contact[: self.data.ncon]:
            for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if self.is_arm[a] and self.counts[b] and not self.is_arm[b]:
                    self.touched.add(self.names[b])

    def wait(self, seconds: float) -> None:
        for _ in range(int(seconds / self.model.opt.timestep)):
            self.step()

    def move(self, target: np.ndarray, arm: str, yaw: float, seconds: float, settle: float = 0.3) -> None:
        """Slide the joint targets to the pose over ``seconds``, as kinematics.move_to does."""
        actuators = lab.arm_actuator_ids(self.model, arm)
        start = self.data.ctrl[actuators].copy()
        goal = solve(self.model, self.data, target, arm, yaw)
        steps = int(seconds / self.model.opt.timestep)
        for i in range(1, steps + 1):
            self.data.ctrl[actuators] = start + (goal - start) * i / steps
            self.step()
        self.wait(settle)


def squeeze_force(model, data, tube: str) -> float:
    """The total force (N) the finger pads press on the tube with."""
    body = model.body(tube).id
    force, f6 = 0.0, np.zeros(6)
    for i, c in enumerate(data.contact[: data.ncon]):
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        names = model.geom(c.geom1).name + model.geom(c.geom2).name
        if body in (b1, b2) and "pad" in names:
            mujoco.mj_contactForce(model, data, i, f6)
            force += abs(f6[0])
    return force


def run(
    seed: int,
    target: str,
    xy: np.ndarray,
    yaw: float,
    top: float,
    squeeze: float = lab.GRIP_SQUEEZE,
    liquid_in: bool = False,
    seen: Callable[[str, mujoco.MjModel, mujoco.MjData], None] | None = None,
) -> dict:
    """Grasp ``target`` in the scene of ``seed`` with its pinch point at ``xy``, turned ``yaw``.

    ``squeeze`` is how much narrower than the tube the fingers are told to close.
    With ``liquid_in`` the target holds its liquid, for a true weight (the
    other tubes stay empty, see scenes.py).
    ``seen``, if given, is called after each stage (above, down, closed, lifted),
    for pictures.
    """
    seen = seen or (lambda *_: None)
    model, data, scene = scenes.build(seed, target if liquid_in else None)
    watch = Watch(model, data, target)
    tube = next(t for t in scenes.tubes(scene) if t.name == target)
    container = random_scene.BY_NAME[tube.kind]
    arm = min(lab.ARM_NAMES, key=lambda a: np.linalg.norm(data.body(f"{a}/base").xpos[:2] - xy))
    start = {i.name: data.body(i.name).xpos.copy() for i in scene.instances}
    balls = len(liquid.balls_in(model, data, container, target))
    low = rules.grip_height(top)
    x, y = xy
    try:
        watch.move(np.array([x, y, max(HIGH, top + 0.1)]), arm, yaw, 1.5)
        watch.move(np.array([x, y, top + rules.APPROACH]), arm, yaw, 0.7)
        seen("above", model, data)
        watch.move(np.array([x, y, low]), arm, yaw, 1.0)
        seen("down", model, data)
        touched_going_down = sorted(watch.touched)
        data.ctrl[lab.gripper_actuator_id(model, arm)] = lab.grip_command(2 * rules.TUBE_RADIUS, squeeze)
        watch.wait(0.8)
        seen("closed", model, data)
        touched_closing = sorted(watch.touched - set(touched_going_down))
        force = squeeze_force(model, data, target)
        pinch0 = data.site(lab.pinch_site(arm)).xpos[2]
        tube0 = data.body(target).xpos[2]
        watch.move(np.array([x, y, low + rules.LIFT]), arm, yaw, 1.0)
        seen("lifted", model, data)
        finite = bool(np.isfinite(data.qpos).all())
    except Exception:  # a solver blow-up can surface as a MuJoCo error
        finite = False
    if not finite:
        return {"success": False, "blew_up": True, "touched": sorted(watch.touched)}
    rose = float(data.body(target).xpos[2] - start[target][2])
    upright = float(data.body(target).xmat[8])
    moved = max(float(np.linalg.norm(data.body(k).xpos - p)) for k, p in start.items() if k != target)
    lifted = rose > ROSE and upright > UPRIGHT
    return {
        # Taken cleanly: up, upright, and nothing else touched on the way.
        "success": lifted and not watch.touched,
        "lifted": lifted,
        "touched": sorted(watch.touched),
        "touched_going_down": touched_going_down,
        "touched_closing": touched_closing,
        "blew_up": moved > 1.0,
        "rose": rose,
        "upright": upright,
        "moved": moved,
        "slip": float(
            (data.site(lab.pinch_site(arm)).xpos[2] - pinch0) - (data.body(target).xpos[2] - tube0)
        ),
        "squeeze": force,
        "spilled": balls - len(liquid.balls_in(model, data, container, target)),
        "balls": balls,
        "weight": float(model.body_subtreemass[model.body(target).id]) + balls * _ball_mass(model),
        "arm": arm,
    }


def _ball_mass(model: mujoco.MjModel) -> float:
    for i in range(model.nbody):
        if model.body(i).name.startswith(liquid.BALL_PREFIX):
            return float(model.body_mass[i])
    return 0.0
