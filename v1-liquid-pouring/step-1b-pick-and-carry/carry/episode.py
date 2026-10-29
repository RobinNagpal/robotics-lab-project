"""One run of Step 1b: find the tubes, choose a pair, both arms pick, and meet at the pouring point.

The coordinator is the only thing that decides. The arms never talk to each
other; each runs a script (a Python generator, see motion.py), and the
coordinator steps both scripts and the simulation together. The scripts only
share a few flags, which stand for the messages an arm would send back:

    arm A (receiver)                     arm B (source)
    ----------------                     --------------
    wait at home                         over, turn the fingers level,
                                         down beside the tube, in, close, up
                                         "picked"; park it behind the source rack
    plan the pick again  <-------------  "waiting"
    pick the same way, "picked"
    carry to the pouring point
    hold still, "ready"  ------------->  carry to the waiting spot beside the middle
    keep holding                         move in beside the receiver, "in place"

    python -m carry.episode --seed 0 --size 0.15
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import torch
from grasp import pick, rules, side
from grasp.net import GraspNet
from lab_scene import liquid
from lab_scene import scene as lab
from lab_scene.settings import LAYOUT

from . import look, motion
from .perceive import Perception, Table, Tube

ROOT = Path(__file__).resolve().parents[2]
# Step 1's side-grasp network, and the cut-off chosen for it on its validation pictures.
GRASP_RUN = ROOT / "step-1-grasp" / "runs" / "side_grconv"
# Each arm takes the tube on its own side: arm 1 rests on the +y side, next to the source
# rack, and arm 2 on the -y side, next to the empty rack. (The other way round, each arm's
# routes crossed the other's side of the table.)
RECEIVER_ARM, SOURCE_ARM = "arm2", "arm1"
# At the pouring point the two tubes stand side by side along the line between
# the arms, and the grippers hold them from opposite sides across it: the
# receiver from -y, the source from +y. (From each arm's own side, the gripper
# would sit between the tube and the arm's base, too close for the arm to bend.)
# Tilting the source towards the receiver is then a roll of its wrist.
RECEIVER_SIDE, SOURCE_SIDE = 1.5 * math.pi, 0.5 * math.pi

# The receiver's rim is held here: above the middle of the table, as in
# lab_scene.settings.LAYOUT.pour_point, but 35 cm up instead of 25. Held from
# the side, each gripper's wrist reaches out over a rack; at 25 cm it would be
# at the height of the tall tubes' rims.
POUR = np.array([LAYOUT.pour_point[0], LAYOUT.pour_point[1], 0.35])
# The source waits this far beyond its furthest meeting place, away from the receiver and up.
WAIT_OFFSET = np.array([0.05, 0.0, 0.08])
# Where the source arm holds its tube while the receiver arm works: on the source arm's own
# side (+y), behind the source rack, out of the middle of the table.
PARKS = ((0.30, 0.40), (0.38, 0.45), (0.22, 0.32))
# How far apart the two tubes' axes may be when they meet (m), and the margin
# added to the closest clear distance.
MIN_DISTANCE, MAX_DISTANCE, MARGIN = 0.03, 0.15, 0.01
# The source's rim is held this much higher than the receiver's.
SOURCE_ABOVE = 0.02
# Carrying, a tube's bottom stays this far above the tallest rim on the table.
CLEARANCE = 0.04
# Moving empty-handed, the pinch point stays this far above the tallest rim.
EMPTY_CLEARANCE = 0.12
# m/s: moving empty-handed, carrying a tube, and the last few cm near a tube.
FAST, CARRY, SLOW = 0.25, 0.12, 0.05
# The arm's own check of a grasp keeps this far from everything.
CHECK_MARGIN = 0.005
SETTLE = 0.5  # s held still before reporting
# When only the other arm is in the way: look again this often (s), for at most this long.
RECHECK, MAX_WAIT = 0.2, 8.0
# Held from the side, a tube's axis sits this far behind the pinch point, towards the gripper:
# the fingers go side.DEPTH (11.5 mm) past the axis, and the tube settles a few millimetres further
# back as they close, more when it is full. Measured over 30 runs: 14 mm empty, 16 mm full.
HELD_BEHIND = 0.015
TIME_LIMIT = 120.0  # s of simulated time


@dataclass
class SideGrasp:
    xy: np.ndarray  # the tube's axis
    direction: float  # where the gripper comes from, radians (grasp/side.py)
    top: float  # rim height, m
    score: float
    plan: list = field(default_factory=list)  # the checked way in (Coordinator.plan_pick)
    others: list = field(default_factory=list)  # the network's other clear directions, best first


@dataclass
class Choice:
    receiver: Tube
    source: Tube
    receiver_grasp: SideGrasp
    source_grasp: SideGrasp
    tried: list[float] = field(default_factory=list)  # sizes tried before this one


class Coordinator:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, perception: Perception, net: GraspNet):
        self.model, self.data = model, data
        self.perception, self.net = perception, net
        self.planner = motion.Planner(model)
        self.flags: dict[str, float] = {}  # message -> time it was sent
        self.log: list[tuple[float, str, str]] = []
        self.heard: Callable[[str, str], None] | None = None  # called on every message, for pictures
        # How far each arm's fingers were from their target as they closed.
        self.at_close: dict[str, list[float]] = {}
        self.holding: dict[str, str] = {}  # arm -> the tube it holds
        self.closest_log: list[tuple[int, str]] = []  # why each closer meeting distance was refused
        self.timeline: list[tuple[float, str, str]] = []  # when each arm closed and lifted
        self.threshold = _threshold()

    def say(self, arm: str, message: str) -> None:
        self.flags[message] = self.data.time
        self.log.append((round(self.data.time, 2), arm, message))
        if self.heard is not None:
            self.heard(arm, message)

    # ---------------------------------------------------------------- choosing

    def grasp_of(self, camera: look.Camera, tube: Tube, arm: str) -> SideGrasp | None:
        """The side-grasp network's best direction that also passes the arm's own check, or None.

        The network ranks the 24 directions at the tube's centre. Each clear one,
        best first, is checked once more with the whole arm (not only the
        gripper and its wrist): turning the fingers level above the tube, coming
        down beside it and in, with a 5 mm margin, and without the arm folding
        onto itself. The first that passes is used.
        """
        heights, calib = camera.heights(tube.xy, tube.top + side.LOOK_HEIGHT)
        device = next(self.net.parameters()).device.type
        scores = pick.model_scores(self.net, heights[None], device)[0]
        best, (u, v, _) = pick.learned(scores, heights, calib, threshold=self.threshold)
        if best is None:
            return None
        ranked = [
            k
            for k in sorted(range(side.BINS), key=lambda k: -scores[k, v, u])
            if scores[k, v, u] >= self.threshold
        ]
        for i, k in enumerate(ranked):
            grasp = SideGrasp(best.xy, float(side.direction(k)), best.top, float(scores[k, v, u]))
            if self.way_in_clear(arm, grasp, tube):
                grasp.others[:] = [float(side.direction(j)) for j in ranked[i + 1 :]]
                return grasp
        return None

    def replan(self, arm: str, grasp: SideGrasp, tube: Tube) -> SideGrasp:
        """Plan the pick again from the scene as it is now, trying the network's directions in order."""
        for direction in [grasp.direction, *grasp.others]:
            option = SideGrasp(grasp.xy, direction, grasp.top, grasp.score)
            if self.way_in_clear(arm, option, tube):
                return option
        raise motion.Blocked(f"{arm} has no clear way to its tube any more")

    def way_in_clear(self, arm: str, grasp: SideGrasp, tube: Tube) -> bool:
        """Plan the whole pick from where the arm is now, check it, and keep the plan in the grasp."""
        try:
            target = self.truth_name(tube.xy)
            grasp.plan[:] = self.plan_pick(arm, grasp, target)
            for path, _, ignore in grasp.plan:
                self.planner.check(self.data, arm, path, None, ignore, margin=CHECK_MARGIN)
            self.planner.check_closing(self.data, arm, grasp.plan[-1][0][-1], grasp.direction, target)
        except motion.Blocked:
            grasp.plan.clear()
            return False
        return True

    def plan_pick(self, arm: str, grasp: SideGrasp, target: str) -> list[tuple[list, float, set]]:
        """Joint paths, with speeds and what each may touch: up and across with the fingers down,
        turn them level, down beside the tube, in. Executed exactly as planned (pick)."""
        a = grasp.direction
        path = side.way_in(grasp.xy, a, grasp.top)
        out, pinch = path[0], path[-1]
        joints = lab.arm_joint_ids(self.model, arm)
        work = mujoco.MjData(self.model)
        work.qpos[:] = self.data.qpos
        here = motion.pinch(self.data, arm)
        yaw = motion.open_angle(self.data, arm)
        plan = []

        def add(points, speed, ignore=frozenset()):
            plan.append((points, speed, set(ignore)))
            work.qpos[joints] = points[-1]

        up = np.array([here[0], here[1], self.transfer])
        above = np.array([out[0], out[1], self.transfer])
        add(self.planner.solve_line(work, arm, here, up, yaw, yaw), FAST)
        add(self.planner.solve_line(work, arm, up, above, yaw, yaw), FAST)
        add(self.planner.turn_level(work, arm, above, a), FAST)
        add(self.planner.side_line(work, arm, above, out + [0, 0, rules.APPROACH], a, a), FAST)
        add(self.planner.side_line(work, arm, out + [0, 0, rules.APPROACH], out, a, a), SLOW)
        add(self.planner.side_line(work, arm, out, pinch, a, a), SLOW, {target})
        return plan

    def choose(self, table: Table, sizes: list[float]) -> Choice | None:
        """The first size in ``sizes`` with a receiver and a source that can both be taken."""
        self.table = table
        self.transfer = max(side.TRANSFER, table.tallest_rim + EMPTY_CLEARANCE)
        camera = look.Camera(self.model, self.data)
        tried = []
        try:
            for size in sizes:
                receivers = [t for t in table.in_rack("empty_rack") if t.size == size]
                sources = [t for t in table.in_rack("source_rack") if t.size == size and t.liquid]
                for r in receivers:
                    for s in sources:
                        gr = self.grasp_of(camera, r, RECEIVER_ARM)
                        gs = self.grasp_of(camera, s, SOURCE_ARM) if gr is not None else None
                        if gr is not None and gs is not None:
                            return Choice(r, s, gr, gs, tried)
                tried.append(size)
        finally:
            camera.close()
        return None

    # ---------------------------------------------------------------- moving

    def line(
        self, arm: str, end: np.ndarray, yaw: float, speed: float, held=None, ignore=frozenset()
    ) -> Iterator:
        """Plan, check, then follow a straight line from where the arm is now, fingers down."""
        start = motion.pinch(self.data, arm)
        yaw0 = motion.open_angle(self.data, arm)
        path = self.planner.solve_line(self.data, arm, start, end, yaw0, yaw)
        yield from self.checked(arm, path, held, ignore)
        yield from motion.follow(self.model, self.data, arm, path, speed)

    def level(
        self, arm: str, end: np.ndarray, d0: float, d1: float, speed: float, held=None, ignore=frozenset()
    ) -> Iterator:
        """Plan, check, then follow a straight line with the fingers level, turning from d0 to d1.

        Turning the gripper round its tube while moving can need a huge wrist swing
        somewhere on the way, so several ways are tried, and the first that the arm
        can follow and that touches nothing is used: turning the shorter way round
        or the longer way, while moving; moving first, then turning on the spot; or
        turning on the spot first, then moving.
        """
        waited = 0.0
        while True:
            paths, problem = self._level_paths(arm, end, d0, d1, held, ignore)
            if paths is not None:
                break
            if not self._other_arm_in_the_way(arm, problem) or waited >= MAX_WAIT:
                raise problem
            yield from motion.wait(self.model, RECHECK)  # the other arm is in the way: let it move
            waited += RECHECK
        for path in paths:
            yield from self.checked(arm, path, held, ignore)
            yield from motion.follow(self.model, self.data, arm, path, speed)

    def _level_paths(self, arm, end, d0, d1, held, ignore):
        """The first of the ways (see level) the arm can follow without touching anything."""
        start = motion.pinch(self.data, arm)
        joints = lab.arm_joint_ids(self.model, arm)
        ways = [
            [(start, end, d0, d1, False)],
            [(start, end, d0, d1, True)],
            [(start, end, d0, d0, False), (end, end, d0, d1, False)],
            [(start, end, d0, d0, False), (end, end, d0, d1, True)],
            [(start, start, d0, d1, False), (start, end, d1, d1, False)],
            [(start, start, d0, d1, True), (start, end, d1, d1, False)],
        ]
        problem = None
        for way in ways:
            work = mujoco.MjData(self.model)
            work.qpos[:] = self.data.qpos
            paths = []
            try:
                for a, b, e0, e1, long_way in way:
                    if np.allclose(a, b) and abs(motion._round(e0, e1)) < 1e-6:
                        continue
                    path = self.planner.side_line(work, arm, a, b, e0, e1, long_way=long_way)
                    self.planner.check(work, arm, path, held, set(ignore))
                    paths.append(path)
                    work.qpos[joints] = path[-1]
            except motion.Blocked as e:
                if problem is None or self._other_arm_in_the_way(arm, e):
                    problem = e
                continue
            return paths, None
        return None, problem

    def _other_arm_in_the_way(self, arm: str, problem: Exception) -> bool:
        other = next(a for a in lab.ARM_NAMES if a != arm)
        theirs = self.holding.get(other)
        return f" {other}/" in str(problem) or (theirs is not None and f" {theirs} (" in str(problem))

    def checked(self, arm: str, path: list[np.ndarray], held=None, ignore=frozenset()) -> Iterator:
        """Check the path; if only the other arm is in the way, wait for it and check again.

        Anything else in the way stops the run. The other arm is still moving, so
        waiting is the coordinator's way of letting the arms take turns where their
        paths cross.
        """
        waited = 0.0
        while True:
            try:
                self.planner.check(self.data, arm, path, held, set(ignore))
                return
            except motion.Blocked as e:
                if not self._other_arm_in_the_way(arm, e) or waited >= MAX_WAIT:
                    raise
            yield from motion.wait(self.model, RECHECK)
            waited += RECHECK

    def pick(self, arm: str, grasp: SideGrasp, name: str, carry: float) -> Iterator:
        """Follow the planned way in, close, and lift. Returns what is held."""
        a = grasp.direction
        pinch = side.way_in(grasp.xy, a, grasp.top)[-1]  # side.pinch_at: past the axis by side.DEPTH
        for path, speed, ignore in grasp.plan:
            yield from self.checked(arm, path, None, ignore)
            yield from motion.follow(self.model, self.data, arm, path, speed)
        self.at_close[arm] = (motion.pinch(self.data, arm) - pinch).tolist()
        self.timeline.append((round(self.data.time, 2), arm, "closing"))
        self.data.ctrl[lab.gripper_actuator_id(self.model, arm)] = lab.grip_command(2 * rules.TUBE_RADIUS)
        yield from motion.wait(self.model, 0.8)
        held = motion.grab(self.data, arm, name)
        self.holding[arm] = name
        self.timeline.append((round(self.data.time, 2), arm, "lifting"))
        rack = self.rack_of(name)
        yield from self.level(arm, pinch + [0, 0, side.LIFT], a, a, SLOW, held, ignore={rack})
        yield from self.level(arm, np.array([pinch[0], pinch[1], carry]), a, a, CARRY, held)
        return held

    def rack_of(self, tube: str) -> str:
        xy = self.data.body(tube).xpos[:2]
        return min(lab.RACKS, key=lambda r: np.linalg.norm(self.data.body(r).xpos[:2] - xy))

    def receiver_script(self, c: Choice, name: str) -> Iterator:
        arm = RECEIVER_ARM
        # One arm moves at a time (see source_script): Arm A waits at home until Arm B has its
        # source parked at the waiting spot, then plans its pick again against the scene as it is.
        while "waiting" not in self.flags:
            yield
        g = self.replan(arm, c.receiver_grasp, c.receiver)
        offset = side.grip_height(g.top) - g.top  # pinch below the rim
        carry = self.carry_height(g)
        held = yield from self.pick(arm, g, name, carry)
        self.say(arm, "picked")
        # Held, the tube's axis sits HELD_BEHIND the pinch point (towards the gripper).
        hold = POUR + [0, 0, offset] - HELD_BEHIND * side.along(RECEIVER_SIDE)
        yield from self.level(
            arm, np.array([hold[0], hold[1], carry]), g.direction, RECEIVER_SIDE, CARRY, held
        )
        yield from self.level(arm, hold, RECEIVER_SIDE, RECEIVER_SIDE, SLOW, held)
        yield from motion.wait(self.model, SETTLE)
        self.say(arm, "ready")
        while True:
            yield

    def source_script(self, c: Choice, name: str) -> Iterator:
        arm = SOURCE_ARM
        # One arm moves at a time: held from the side, both arms reach far across the table,
        # and a move checked at its start can meet the other arm moving into its way. Arm B goes
        # first, while Arm A is still at home, and waits holding the source up over its own
        # rack, out of the middle; then Arm A picks and carries; then Arm B comes in.
        g = c.source_grasp
        offset = side.grip_height(g.top) - g.top
        carry = self.carry_height(g)
        held = yield from self.pick(arm, g, name, carry)
        self.say(arm, "picked")
        # Out of the middle: back towards its own base, behind the source rack, gripper outwards.
        facing = None
        for spot in PARKS:
            for turn_to in (SOURCE_SIDE, g.direction):  # turned outwards if the arm can, else as it is
                try:
                    yield from self.level(arm, np.array([*spot, carry]), g.direction, turn_to, CARRY, held)
                except motion.Blocked:
                    continue
                facing = turn_to
                break
            if facing is not None:
                break
        if facing is None:
            raise motion.Blocked(f"{arm} has nowhere to park the source")
        self.say(arm, "waiting")
        while "ready" not in self.flags:
            yield
        past = -HELD_BEHIND * side.along(SOURCE_SIDE)  # the tube's axis sits behind the pinch point
        waiting = POUR + [MAX_DISTANCE, 0, SOURCE_ABOVE + offset] + WAIT_OFFSET + past
        high = np.array([waiting[0], waiting[1], max(carry, waiting[2])])
        yield from self.level(arm, high, facing, SOURCE_SIDE, CARRY, held)
        yield from self.level(arm, waiting, SOURCE_SIDE, SOURCE_SIDE, SLOW, held)
        # The receiver is now held still: find how close the source can come.
        self.meet_distance = self.closest(arm, held, offset)
        meet = POUR + [self.meet_distance, 0, SOURCE_ABOVE + offset] + past
        yield from self.level(arm, meet, SOURCE_SIDE, SOURCE_SIDE, SLOW, held)
        yield from motion.wait(self.model, SETTLE)
        self.say(arm, "in place")
        while True:
            yield

    def closest(self, arm: str, held: motion.Held, offset: float) -> float:
        """The smallest distance between the two tubes' axes at which the source can come in clear.

        Tried with collision tests only, from 3 cm up, then 1 cm is added to keep a margin.
        """
        start = motion.pinch(self.data, arm)
        for d in np.arange(MIN_DISTANCE, MAX_DISTANCE + 1e-9, 0.005):
            meet = POUR + [d, 0, SOURCE_ABOVE + offset] - HELD_BEHIND * side.along(SOURCE_SIDE)
            try:
                path = self.planner.side_line(self.data, arm, start, meet, SOURCE_SIDE, SOURCE_SIDE)
                self.planner.check(self.data, arm, path, held)
            except motion.Blocked as e:
                self.closest_log.append((round(float(d) * 1000), str(e)))
                continue
            return float(min(d + MARGIN, MAX_DISTANCE))
        raise motion.Blocked("the source cannot come near the receiver")

    def carry_height(self, grasp: SideGrasp) -> float:
        """The pinch height that keeps the held tube's bottom above every rim on the table."""
        size = min((0.1, 0.15, 0.2), key=lambda s: abs(grasp.top - 0.006 - s))
        below = side.grip_height(grasp.top) - (grasp.top - size)  # pinch above the tube's bottom
        return self.table.tallest_rim + CLEARANCE + below

    # ---------------------------------------------------------------- the run

    def run(self, sizes: list[float]) -> dict:
        self.meet_distance = MAX_DISTANCE
        self.table = self.perception.table(self.model, self.data)
        choice = self.choose(self.table, sizes)
        if choice is None:
            return {"chosen": False, "log": self.log}
        names = {
            "receiver": self.truth_name(choice.receiver.xy),
            "source": self.truth_name(choice.source.xy),
        }
        watch = Watch(self.model, self.data, names)
        scripts = {
            RECEIVER_ARM: self.receiver_script(choice, names["receiver"]),
            SOURCE_ARM: self.source_script(choice, names["source"]),
        }
        balls = {k: self.balls(n) for k, n in names.items()}
        failure = None
        start = time.time()
        while self.data.time < TIME_LIMIT and "in place" not in self.flags:
            try:
                for script in scripts.values():
                    next(script)
            except motion.Blocked as e:
                failure = str(e)
                break
            watch.step()
            if not np.isfinite(self.data.qpos).all():
                failure = "simulation blew up"
                break
        return self.grade(choice, names, balls, watch, failure, time.time() - start)

    def truth_name(self, xy: np.ndarray) -> str:
        """The simulator's name for the tube Step 0 saw at ``xy``. For running and grading only."""
        tubes = [n for n in self.containers]
        return min(tubes, key=lambda n: np.linalg.norm(self.data.body(n).xpos[:2] - xy))

    def balls(self, name: str) -> int:
        return len(liquid.balls_in(self.model, self.data, self.containers[name], name))

    def grade(self, choice, names, balls, watch, failure, seconds) -> dict:
        d = self.data
        rim = {k: _rim(d, n, self.containers[n].height) for k, n in names.items()}
        lean = {k: math.degrees(math.acos(min(1.0, d.body(n).xmat[8]))) for k, n in names.items()}
        meet = POUR + [self.meet_distance, 0, SOURCE_ABOVE]
        sizes = {k: self.containers[n].height for k, n in names.items()}
        return {
            "chosen": True,
            "sizes_tried_first": choice.tried,
            "size": choice.receiver.size,
            "pair_matches": abs(sizes["receiver"] - sizes["source"]) < 1e-6,
            "picked_from_right_racks": names["receiver"].startswith("receiver")
            and names["source"].startswith("source"),
            "failure": failure,
            "done": "in place" in self.flags,
            "receiver_error_mm": float(np.linalg.norm(rim["receiver"] - POUR) * 1000),
            "source_error_mm": float(np.linalg.norm(rim["source"] - meet) * 1000),
            "source_offset_mm": np.round((rim["source"] - meet) * 1000, 1).tolist(),
            "receiver_offset_mm": np.round((rim["receiver"] - POUR) * 1000, 1).tolist(),
            "meet_distance_mm": round(self.meet_distance * 1000, 1),
            "closer_refused": self.closest_log[-3:],
            "timeline": self.timeline,
            "lean_deg": lean,
            "balls_lost": balls["source"] - self.balls(names["source"]),
            "balls_start": balls["source"],
            "touched": {a: sorted(t) for a, t in watch.touched.items()},
            "touches": sorted(watch.parts, key=lambda p: p[3])[:5],
            "fingers_off_at_close_mm": {
                a: np.round(np.array(v) * 1000, 1).tolist() for a, v in self.at_close.items()
            },
            "log": self.log,
            "sim_seconds": round(d.time, 2),
            "wall_seconds": round(seconds, 1),
        }


def _rim(data: mujoco.MjData, name: str, height: float) -> np.ndarray:
    body = data.body(name)
    return body.xpos + body.xmat.reshape(3, 3)[:, 2] * height


class Watch:
    """Notes everything each arm touches other than its own tube and the liquid."""

    EVERY = 5

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, names: dict[str, str]):
        self.model, self.data = model, data
        geom_names = [model.body(model.geom_bodyid[g]).name for g in range(model.ngeom)]
        self.names = geom_names
        self.arm_of = {a: np.array([n.startswith(f"{a}/") for n in geom_names]) for a in lab.ARM_NAMES}
        own = {RECEIVER_ARM: names["receiver"], SOURCE_ARM: names["source"]}
        self.allowed = {
            a: np.array([n == own[a] or n.startswith(liquid.BALL_PREFIX) for n in geom_names]) for a in own
        }
        self.touched = {a: set() for a in lab.ARM_NAMES}
        self.parts: set[tuple] = set()  # (arm, the arm's part, what it touched, when), first few
        self.steps = 0

    def step(self) -> None:
        mujoco.mj_step(self.model, self.data)
        self.steps += 1
        if self.steps % self.EVERY:
            return
        for c in self.data.contact[: self.data.ncon]:
            for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                for arm, mine in self.arm_of.items():
                    if mine[a] and not mine[b] and not self.allowed[arm][b]:
                        self.touched[arm].add(self.names[b])
                        self.parts.add((arm, self.names[a], self.names[b], round(float(self.data.time), 2)))


def load_net(weights: Path = GRASP_RUN / "best.pt") -> GraspNet:
    net = GraspNet(side.BINS)
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    return net.eval()


def _threshold(run: Path = GRASP_RUN) -> float:
    """The side-grasp network's cut-off, as chosen on its validation pictures (grasp/evaluate_side.py)."""
    summary = run / "summary.json"
    return float(json.loads(summary.read_text())["threshold"]) if summary.exists() else pick.THRESHOLD


def run_episode(
    seed: int,
    sizes: list[float],
    perception: Perception,
    net: GraspNet,
    heard: Callable | None = None,
) -> dict:
    """``heard(coordinator, arm, message)``, if given, is called on every message."""
    model, truth = lab.build(seed)
    data = mujoco.MjData(model)
    lab.reset(model, data)
    lab.settle(model, data, 0.5)
    coordinator = Coordinator(model, data, perception, net)
    coordinator.containers = truth.containers
    if heard is not None:
        coordinator.heard = lambda arm, message: heard(coordinator, arm, message)
    result = coordinator.run(sizes)
    result["seed"] = seed
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=float, default=0.15)
    args = parser.parse_args()
    sizes = [args.size] + [s for s in (0.2, 0.15, 0.1) if s != args.size]
    result = run_episode(
        args.seed,
        sizes,
        Perception(),
        load_net(),
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
