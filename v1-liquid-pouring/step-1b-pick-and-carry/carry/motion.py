"""Moving an arm along straight lines, checked for collisions before it moves.

In Step 1 the arm went from home to above the tube by sliding its joint
angles straight to the goal. The gripper then swings along whatever curve
the joints make, and once it swept through a rack. Here every move is a
straight line in space: the line is cut into 5 mm pieces, each piece's end
is solved for joint angles, and the arm follows those one after the other.

Before an arm moves, the whole line is checked with collision tests only
(no physics): the arm, and the tube it holds, at every piece's end. If
anything would be touched, the move is not made.

A move is a Python generator: it sets the arm's joint targets for one
simulation step, then yields. The coordinator (episode.py) steps both arms'
generators and the simulation together, so the arms really move at the same
time.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import mujoco
import numpy as np
from grasp import scenes, side, trial
from lab_scene import liquid
from lab_scene import scene as lab

PIECE = 0.005  # m between solved points along a line
REACHED = 0.004  # a solved point further than this from its target means the arm cannot get there
JOINT_SPEED = 0.8  # rad/s, the fastest any joint is asked to turn
# The Robotiq finger joints (driver, coupler, spring link, follower; right, then left) as measured
# holding a full tube while lifting it: the passive linkage settles away from its unloaded closed
# pose, and the couplers swing further out.
LOADED = np.array([0.60, -0.10, 0.53, -0.37, 0.60, -0.10, 0.53, -0.38])
FINGER_ROOM = 0.02  # m kept between the closing fingers and anything but the target
MAX_JUMP = 0.15  # rad, the most any joint may change between two points 5 mm apart
ARRIVED, CATCH_UP = 0.01, 2.0  # rad: close enough to the targets; s: the most to wait for that


class Blocked(RuntimeError):
    """A planned move would touch something, or the arm cannot reach."""


@dataclass
class Held:
    """A tube in an arm's fingers: its name and where it sits relative to the pinch point."""

    name: str
    pos: np.ndarray  # in the pinch site's frame
    mat: np.ndarray


def smooth(s: float) -> float:
    """0 to 1 with gentle speeding up and slowing down, so held liquid is not jolted."""
    return s * s * (3 - 2 * s)


class Planner:
    """Solves and checks lines for one arm, on a scratch copy of the simulation state."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        names = [model.body(model.geom_bodyid[g]).name for g in range(model.ngeom)]
        self.names = names
        self.is_ball = np.array([n.startswith(liquid.BALL_PREFIX) for n in names])
        self.arm_of = {a: np.array([n.startswith(f"{a}/") for n in names]) for a in lab.ARM_NAMES}
        # How far closed the fingers already are on the way in (0: fully open), and the sideways
        # shove tried while closing. Side grasps version 3 (Coordinator) come in with the fingers
        # opened only 35 mm and need no shove (step-1-grasp/docs/side/version2.md).
        self.way_in_closing = 0.0
        self.shove = side.SideChecker.SHOVE

    def solve_line(
        self, data: mujoco.MjData, arm: str, start: np.ndarray, end: np.ndarray, yaw0: float, yaw1: float
    ) -> list[np.ndarray]:
        """Joint angles every 5 mm from ``start`` to ``end``, turning from ``yaw0`` to ``yaw1``."""
        work = mujoco.MjData(self.model)
        work.qpos[:] = data.qpos
        joints = lab.arm_joint_ids(self.model, arm)
        steps = max(1, math.ceil(np.linalg.norm(end - start) / PIECE))
        out = []
        for i in range(1, steps + 1):
            s = i / steps
            target = start + (end - start) * s
            q = trial.solve(self.model, work, target, arm, yaw0 + _turn(yaw0, yaw1) * s)
            _no_jump(arm, work.qpos[joints], q)
            work.qpos[joints] = q
            mujoco.mj_kinematics(self.model, work)
            if np.linalg.norm(work.site(lab.pinch_site(arm)).xpos - target) > REACHED:
                raise Blocked(f"{arm} cannot reach {np.round(target, 3)}")
            out.append(q)
        return out

    def side_line(
        self,
        data: mujoco.MjData,
        arm: str,
        start: np.ndarray,
        end: np.ndarray,
        dir0: float,
        dir1: float,
        long_way: bool = False,
    ) -> list[np.ndarray]:
        """As solve_line, with the fingers level, coming in from a direction that turns from dir0 to dir1."""
        turn = _round(dir0, dir1)
        if long_way and abs(turn) > 1e-6:
            turn -= math.copysign(2 * math.pi, turn)
        work = mujoco.MjData(self.model)
        work.qpos[:] = data.qpos
        joints = lab.arm_joint_ids(self.model, arm)
        steps = max(
            1, math.ceil(np.linalg.norm(end - start) / PIECE), math.ceil(abs(_round(dir0, dir1)) / 0.05)
        )
        out = []
        for i in range(1, steps + 1):
            s = i / steps
            try:
                q = side.solve(
                    self.model, work, start + (end - start) * s, arm, dir0 + _round(dir0, dir1) * s
                )
            except side.Unreachable as e:
                raise Blocked(str(e)) from e
            _no_jump(arm, work.qpos[joints], q)
            work.qpos[joints] = q
            out.append(q)
        return out

    def turn_level(
        self, data: mujoco.MjData, arm: str, point: np.ndarray, direction: float
    ) -> list[np.ndarray]:
        """From fingers down to fingers level, in place: the joints slide to an elbow-up pose."""
        joints = lab.arm_joint_ids(self.model, arm)
        try:
            goal = side.solve(self.model, data, point, arm, direction, elbow_up=True)
        except side.Unreachable as e:
            raise Blocked(str(e)) from e
        now = data.qpos[joints].copy()
        steps = max(2, math.ceil(np.abs(goal - now).max() / 0.05))
        return [now + (goal - now) * i / steps for i in range(1, steps + 1)]

    def check_closing(
        self, data: mujoco.MjData, arm: str, q: np.ndarray, direction: float, target: str
    ) -> None:
        """Raise Blocked if the arm's fingers, closing at pose ``q``, would touch anything but ``target``.

        As the Robotiq fingers close, the linkage swings outwards; and the first
        finger to reach the tube shoves the arm a little along the finger line. The
        fingers are tried 25 to 100% closed, also shifted 3 mm either way, as in the
        side-grasp network's labels (grasp/side.py), and in the pose they settle
        into holding a full tube (LOADED), which swings the couplers further out.
        """
        work = mujoco.MjData(self.model)
        work.qpos[:] = data.qpos
        joints = lab.arm_joint_ids(self.model, arm)
        fingers = np.array(
            [
                self.model.jnt_qposadr[j]
                for j in range(self.model.njnt)
                if self.model.joint(j).name.startswith(f"{arm}/gripper/")
            ]
        )
        closed = scenes._closed_fingers()
        poses = [max(f, self.way_in_closing) * closed for f in side.SideChecker.CLOSING] + [LOADED]
        own = self.arm_of[arm]
        skip = np.array([n in (target, "world") for n in self.names]) | self.is_ball
        site = self.model.site(lab.pinch_site(arm)).id
        # Touching nothing at all (racks included)...
        self._closing(work, joints, fingers, poses, own, skip, site, arm, q, direction)
        # ...and staying FINGER_ROOM from every other tube: the passive linkage swings further
        # out under a full tube's load, and more as the tube comes out of its cell. (Not from the
        # racks: a short tube's fingers close only 2 cm above the rack's top plate.)
        racks = np.array([n in lab.RACKS for n in self.names])
        hand = np.array([n.startswith(f"{arm}/gripper/") for n in self.names])
        saved = self.model.geom_margin.copy()
        self.model.geom_margin[hand] = FINGER_ROOM
        try:
            self._closing(work, joints, fingers, poses, own, skip | racks, site, arm, q, direction)
        finally:
            self.model.geom_margin[:] = saved

    def _closing(self, work, joints, fingers, poses, own, skip, site, arm, q, direction) -> None:
        for shove in sorted({0.0, -self.shove, self.shove}, key=abs):
            work.qpos[joints] = q
            if shove:
                mujoco.mj_kinematics(self.model, work)
                opening = work.site_xmat[site].reshape(3, 3)[:, 1]
                pinch = work.site_xpos[site] + shove * opening
                try:
                    work.qpos[joints] = side.solve(self.model, work, pinch, arm, direction)
                except side.Unreachable:
                    continue
            for pose in poses:
                work.qpos[fingers] = pose
                mujoco.mj_kinematics(self.model, work)
                mujoco.mj_collision(self.model, work)
                for c in work.contact[: work.ncon]:
                    for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                        if own[a] and not own[b] and not skip[b]:
                            raise Blocked(f"{arm} would touch {self.names[b]} while closing")

    def check(
        self,
        data: mujoco.MjData,
        arm: str,
        path: list[np.ndarray],
        held: Held | None,
        ignore: set[str] = frozenset(),
        margin: float = 0.0,
    ) -> None:
        """Raise Blocked if the arm, or its tube, would come within ``margin`` of anything on ``path``."""
        if margin:
            saved = self.model.geom_margin.copy()
            self.model.geom_margin[self.arm_of[arm]] = margin
            try:
                # The arm stands on the table: within the margin of it from the start.
                self.check(data, arm, path, held, set(ignore) | {"world"})
            finally:
                self.model.geom_margin[:] = saved
            return
        work = mujoco.MjData(self.model)
        work.qpos[:] = data.qpos
        joints = lab.arm_joint_ids(self.model, arm)
        own = self.arm_of[arm].copy()
        tube_adr = None
        tube = np.zeros(self.model.ngeom, bool)
        if held is not None:
            body = self.model.body(held.name)
            tube = np.array([self.model.geom_bodyid[g] == body.id for g in range(self.model.ngeom)])
            own |= tube
            tube_adr = self.model.jnt_qposadr[body.jntadr[0]]
        # The held tube may touch its own gripper, never the rest of its own arm.
        links = self.arm_of[arm] & ~np.array([n.startswith(f"{arm}/gripper/") for n in self.names])
        skip = np.array([n in ignore for n in self.names]) | self.is_ball
        for q in path:
            work.qpos[joints] = q
            if held is not None:
                mujoco.mj_kinematics(self.model, work)
                site = work.site(lab.pinch_site(arm))
                mat = site.xmat.reshape(3, 3)
                quat = np.zeros(4)
                mujoco.mju_mat2Quat(quat, (mat @ held.mat).ravel())
                work.qpos[tube_adr : tube_adr + 3] = site.xpos + mat @ held.pos
                work.qpos[tube_adr + 3 : tube_adr + 7] = quat
            mujoco.mj_kinematics(self.model, work)
            if side.folds_onto_itself(self.model, work, arm):
                raise Blocked(f"{arm} would fold onto itself or touch the table")
            mujoco.mj_collision(self.model, work)
            for c in work.contact[: work.ncon]:
                for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                    if own[a] and not own[b] and not skip[b]:
                        part = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, a) or self.names[a]
                        raise Blocked(f"{arm} would touch {self.names[b]} (with {part})")
                    if tube[a] and links[b]:
                        raise Blocked(f"{arm} would hit its own {self.names[b]} with the tube it holds")


def _no_jump(arm: str, before: np.ndarray, after: np.ndarray) -> None:
    """Refuse a path on which a joint would jump: near a singular pose, a few millimetres of the
    fingers can need half a turn of a wrist, and the arm cannot follow (it falls far behind)."""
    if np.abs(after - before).max() > MAX_JUMP:
        raise Blocked(f"{arm} would have to swing a joint by {np.abs(after - before).max():.1f} rad")


def _round(a: float, b: float) -> float:
    """The shorter way from direction a to direction b, all the way round."""
    return (b - a + math.pi) % (2 * math.pi) - math.pi


def _turn(a: float, b: float) -> float:
    """The shorter way from turn a to turn b; half a turn apart is the same grasp."""
    return (b - a + math.pi / 2) % math.pi - math.pi / 2


def follow(
    model: mujoco.MjModel, data: mujoco.MjData, arm: str, path: list[np.ndarray], speed: float
) -> Iterator:
    """Slide the arm's joint targets along ``path`` at about ``speed`` m/s, gently, then wait to arrive.

    The time allowed is the longer of: the path's length at ``speed`` (points are
    5 mm apart), and its largest joint change at JOINT_SPEED (a turn of the
    wrist covers little distance but a lot of angle). Afterwards the arm is
    given time to catch up with its targets, so the next move starts from
    where the arm really is.
    """
    actuators = lab.arm_actuator_ids(model, arm)
    joints = lab.arm_joint_ids(model, arm)
    points = [data.ctrl[actuators].copy(), *path]
    travel = np.abs(np.diff(np.array(points), axis=0)).sum(axis=0).max()
    seconds = max(0.2, (len(points) - 1) * PIECE / speed, travel / JOINT_SPEED) * 1.5  # gentle start and end
    steps = int(seconds / model.opt.timestep)
    for i in range(1, steps + 1):
        x = smooth(i / steps) * (len(points) - 1)
        k = min(int(x), len(points) - 2)
        f = x - k
        data.ctrl[actuators] = points[k] + (points[k + 1] - points[k]) * f
        yield
    for _ in range(int(CATCH_UP / model.opt.timestep)):
        if np.abs(data.qpos[joints] - data.ctrl[actuators]).max() < ARRIVED:
            break
        yield


def wait(model: mujoco.MjModel, seconds: float) -> Iterator:
    for _ in range(int(seconds / model.opt.timestep)):
        yield


def pinch(data: mujoco.MjData, arm: str) -> np.ndarray:
    return data.site(lab.pinch_site(arm)).xpos.copy()


def open_angle(data: mujoco.MjData, arm: str) -> float:
    y = data.site(lab.pinch_site(arm)).xmat.reshape(3, 3)[:, 1]
    return math.atan2(y[1], y[0])


def grab(data: mujoco.MjData, arm: str, tube: str) -> Held:
    """Where ``tube`` sits in the arm's fingers now."""
    site = data.site(lab.pinch_site(arm))
    mat = site.xmat.reshape(3, 3)
    body = data.body(tube)
    return Held(tube, mat.T @ (body.xpos - site.xpos), mat.T @ body.xmat.reshape(3, 3))
