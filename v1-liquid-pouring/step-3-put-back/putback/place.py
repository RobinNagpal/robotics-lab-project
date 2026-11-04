"""Steps 3 and 4: Arm B puts the source back in its own cell, then Arm A puts the filled receiver in the
destination rack.

Where everything goes comes from Step 0, as Step 1b saw the table before anything moved: each rack's
place, and where the source stood. The cells follow from the rack's place (the rack is a known part:
2 rows of 5 cells, 36 mm apart; lab_scene.racks.slot_positions). One arm moves at a time, as in Step 1b.

For each tube, in order:

    straighten   turn the tube upright in the fingers' hold (it has turned up to ~19 degrees in them)
    up           straight up until its bottom clears every rim on the table by 4 cm
    across       over the cell, at that height
    down         slowly into the cell, until the bottom is 4 mm above the cell's floor
    let go       open the fingers to 35 mm (as before a pick)
    back off     the open hand straight back, away from the tube, then up
    home         the arm's joints back to where they started

Every move is planned before the arm moves (inverse kinematics every 5 mm, putback.upright) and checked
with collision tests only, no physics. The whole sequence is planned ahead for a cell and a turn of the
gripper round the tube; the first that passes is used. Each move is planned again from where the tube
really is just before it is made.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import mujoco
import numpy as np
from carry import motion
from grasp import side
from lab_scene import liquid
from lab_scene import scene as lab
from lab_scene.racks import slot_positions
from lab_scene.settings import HOME, RACK

from .upright import Mover, exp_rot, log_rot, rot_z, upright

CLEARANCE = 0.04  # m: carried, the tube's bottom stays this far above the tallest rim on the table
GAP = 0.004  # m: let go with the bottom this far above the cell's floor
STRAIGHTEN_RISE = 0.05  # m: lifted this much while it is turned upright
# ...and moved this far sideways (m): where the arm can hold it upright without its wrist resting on its
# forearm depends on where the tube is. Tried in order: not at all, then towards the arm's base first.
SHIFT = 0.10
# While the arm changes its pose to straighten the tube, the tube may lean this much (degrees). Balls
# only start to leave a tube tipped past about 81 degrees.
MAX_LEAN = 45.0
BACK_OFF = side.OUT + side.DEPTH  # m: the open hand backs away this far, as far as it came in
HAND_UP = 0.10  # m: then up
CARRY, SLOW = 0.10, 0.04  # m/s
LET_GO = 0.8  # s for the fingers to open
# Turns of the gripper round the tube tried (degrees from how it holds the tube now), smallest first.
TURNS = (0, *[s * a for a in range(15, 180, 15) for s in (1, -1)], 180)
# How upright the tube is carried: the share of the way from hanging as the level fingers hold it to
# upright (see PutBack.carried). Tried in this order.
SHARES = (0.5, 0.25)
ABOVE_RACK = 0.01  # m: the tube is made upright with its bottom this far above the rack's top
# After the pour the source first moves away like this (m: x, y, up), keeping the gripper's angle: the
# first that neither rests the wrist harder on the forearm nor presses into anything is used.
UNTIP_AWAY = [
    (dx, dy, up)
    for up in (0.06, 0.04, 0.03, 0.0)
    for dx, dy in ((0, 0), (0, -0.03), (0, 0.03), (-0.03, 0), (0.03, 0), (0, 0.05), (0, -0.05))
    if (dx, dy, up) != (0, 0, 0.0)
]
TOUCH_SLACK = 0.0002  # m: leaving a contact, it may press this much deeper (numerical noise)
# The first move may start with the arm pressing on itself this much more than it already does (m).
FOLD_SLACK = 0.0005


@dataclass
class Move:
    kind: str  # "tube" (carrying), "open" (let go), "hand" (empty-handed) or "joints" (home)
    target: tuple = ()
    speed: float = CARRY
    ignore: frozenset = frozenset()
    # The pour can leave Arm B's wrist resting on its forearm (holding the turned tube upright needs
    # the gripper tipped up). The first move may start so, if it never presses harder and ends clear.
    unfold: bool = False


@dataclass
class Plan:
    cell: int
    bottom: np.ndarray  # where the tube's bottom goes in the cell
    turn: float  # degrees
    moves: list[Move] = field(default_factory=list)
    share: float = 0.0  # how upright it is carried (SHARES)
    lift: tuple = ()  # where the first move takes the tube's bottom, from where it was (m)
    tried: int = 0  # ways tried before this one
    seconds: float = 0.0  # planning time


def self_contact(model: mujoco.MjModel, data: mujoco.MjData, arm: str) -> float:
    """How deep the arm presses into itself (gripper on the rest of the arm) or the table, m."""
    mujoco.mj_collision(model, data)
    depth = 0.0
    for c in data.contact[: data.ncon]:
        n1 = model.body(model.geom_bodyid[c.geom1]).name
        n2 = model.body(model.geom_bodyid[c.geom2]).name
        mine1, mine2 = n1.startswith(f"{arm}/"), n2.startswith(f"{arm}/")
        folded = mine1 and mine2 and (("gripper/" in n1) != ("gripper/" in n2))
        table = (mine1 and n2 == "world" and not n1.endswith(side.STANDING)) or (
            mine2 and n1 == "world" and not n2.endswith(side.STANDING)
        )
        if folded or table:
            depth = max(depth, -c.dist)
    return depth


def cells(rack_xy: np.ndarray) -> list[np.ndarray]:
    return [np.array(p) for p in slot_positions(tuple(rack_xy))]


class PutBack:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        tallest_rim: float,
        step: Callable[[], None],
    ):
        self.model, self.data = model, data
        self.planner = motion.Planner(model)
        self.tallest = tallest_rim
        self.step = step  # one simulation step (with whatever watches it)
        self.log: list[tuple[float, str, str]] = []

    def say(self, arm: str, message: str) -> None:
        self.log.append((round(self.data.time, 2), arm, message))

    # ---------------------------------------------------------------- planning

    def lifts(self, arm: str, tube: str) -> list[np.ndarray]:
        """Where the straightening move may take the tube's bottom, best first."""
        b = self.data.body(tube).xpos.copy()
        base = self.data.body(f"{arm}/base").xpos[:2] - b[:2]
        base /= np.linalg.norm(base)
        shifts = [np.array([dx, dy]) for dx in (-SHIFT, 0, SHIFT) for dy in (-SHIFT, 0, SHIFT)]
        shifts.sort(key=lambda v: (np.linalg.norm(v) > 0, -float(v @ base)))
        return [b + [v[0], v[1], STRAIGHTEN_RISE] for v in shifts]

    def carried(self, arm: str, tube: str, share: float) -> np.ndarray:
        """How the tube is turned while carried: ``share`` of the way from where it would hang with the
        fingers level (as Step 1b carries) to upright. Upright all the way, the wrist can come to rest
        on the forearm; it is only made fully upright just above its cell."""
        site = self.data.site_xmat[self.model.site(lab.pinch_site(arm)).id].reshape(3, 3)
        x = site[:, 0]  # level fingers have this axis pointing straight down (as Step 1b holds a tube)
        axis = np.cross(x, [0.0, 0.0, -1.0])
        s = np.linalg.norm(axis)
        to_level = exp_rot(axis / s * math.atan2(s, -x[2])) if s > 1e-9 else np.eye(3)
        hanging = to_level @ self.data.body(tube).xmat.reshape(3, 3)
        return exp_rot(log_rot(upright(hanging) @ hanging.T) * share) @ hanging

    def start_moves(self, arm: str, tube: str, turn: float, lift: np.ndarray, share: float) -> list[Move]:
        """Away from where the pour left it, to the carrying height: the same whatever the cell."""
        R = rot_z(math.radians(turn)) @ self.carried(arm, tube, share)
        carry = max(self.tallest + CLEARANCE, lift[2])
        # One move of the joints to a clean pose holding the tube as it will be carried, a little up and
        # aside (the arm may have to change its pose a lot), then straight up.
        out = [Move("reach", (lift, R), SLOW, unfold=True)]
        if carry > lift[2] + 1e-4:
            out.append(Move("tube", (np.array([lift[0], lift[1], carry]), R), CARRY))
        return out

    def end_moves(self, tube: str, rack: str, bottom: np.ndarray, at: tuple) -> list[Move]:
        """From the carrying height (the tube's bottom and turn ``at``) into the cell, and away."""
        b, R = at
        above_rack = bottom + [0, 0, RACK.height + ABOVE_RACK]
        return [
            Move("tube", (np.array([bottom[0], bottom[1], b[2]]), R), CARRY),
            Move("tube", (above_rack, R), SLOW),
            Move("tube", (above_rack, upright(R)), SLOW),  # straight up, just above its cell
            Move("tube", (bottom + [0, 0, GAP], upright(R)), SLOW, frozenset({rack})),
            Move("open"),
            Move("hand", ("back",), SLOW, frozenset({tube})),
            Move("hand", ("up",), CARRY, frozenset({tube})),
            Move("joints", (np.array(HOME),), CARRY),
        ]

    def path(
        self, arm: str, tube: str, move: Move, work: mujoco.MjData, mover: Mover, grip, now
    ) -> tuple[list[np.ndarray], motion.Held | None]:
        """The joint path for one move from the arm's pose in ``work``; ``now`` is the tube's (bottom, R)."""
        if move.kind == "tube":
            b1, R1 = move.target
            path = mover.line(work, grip, now[0], now[1], b1, R1)
            held = self._held(grip)
            return path, held
        if move.kind == "reach":
            b1, R1 = move.target
            q0 = work.qpos[mover.joints].copy()
            q1 = self.reach(work, mover, grip, b1, R1)
            n = max(2, math.ceil(np.abs(q1 - q0).max() / 0.02))
            return [q0 + (q1 - q0) * i / n for i in range(1, n + 1)], self._held(grip)
        if move.kind == "hand":
            p = work.site_xpos[mover.site].copy()
            if move.target[0] == "back":
                # Back the way the fingers point, level: the pinch point is past the tube's axis, so
                # going "away from the tube" through the pinch point would push the palm into it.
                fingers = work.site_xmat[mover.site].reshape(3, 3)[:, 2]
                away = -fingers[:2] / max(np.linalg.norm(fingers[:2]), 1e-9)
                end = p + BACK_OFF * np.array([away[0], away[1], 0.0])
            else:
                end = p + [0, 0, HAND_UP]
            return mover.hand_line(work, end), None
        if move.kind == "joints":
            q0 = work.qpos[mover.joints].copy()
            q1 = move.target[0]
            n = max(2, math.ceil(np.abs(q1 - q0).max() / 0.05))
            return [q0 + (q1 - q0) * i / n for i in range(1, n + 1)], None
        return [], None

    def reach(self, work: mujoco.MjData, mover: Mover, grip, b: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Joints holding the tube with its bottom at ``b``, turned ``R``, not folded: of the poses found
        from several starting guesses, the nearest to the arm's pose now."""
        pos, rot = grip
        q0 = work.qpos[mover.joints].copy()
        guesses = [q0, np.array(HOME)]  # the arm's own pose first: when that works, it is the nearest
        for flip in ((0, 0, 0), (math.pi, 0, math.pi), (-math.pi, 0, -math.pi)):
            g = q0.copy()
            g[3] += flip[0]
            g[4] = -g[4] if flip[0] else g[4]
            g[5] += flip[2]
            guesses.append(g)
        best = None
        for g in guesses:
            trial = mujoco.MjData(self.model)
            trial.qpos[:] = work.qpos
            trial.qpos[mover.joints] = g
            q, miss = mover.solve(trial, b + R @ pos, R @ rot, iterations=300)
            if miss > 0.003:
                continue
            trial.qpos[mover.joints] = q
            mujoco.mj_kinematics(self.model, trial)
            if self_contact(self.model, trial, mover.arm) > 0:
                continue
            cost = np.abs(q - q0).max()
            if g is guesses[0]:
                return q
            if best is None or cost < best[0]:
                best = (cost, q)
        if best is None:
            raise motion.Blocked(f"{mover.arm} finds no clean pose holding its tube upright there")
        return best[1]

    def check(self, data: mujoco.MjData, arm: str, path: list, held: motion.Held | None, move: Move) -> None:
        """As carry.motion.Planner.check, except that a move marked ``unfold`` may start with the arm
        resting on itself, if it never presses harder than at the start and ends clear."""
        if not move.unfold:
            self.planner.check(data, arm, path, held, set(move.ignore))
            return
        work = mujoco.MjData(self.model)
        work.qpos[:] = data.qpos
        mujoco.mj_kinematics(self.model, work)
        allowed = self_contact(self.model, work, arm) + FOLD_SLACK
        joints = lab.arm_joint_ids(self.model, arm)
        for i, q in enumerate(path):
            work.qpos[joints] = q
            mujoco.mj_kinematics(self.model, work)
            depth = self_contact(self.model, work, arm)
            if depth > allowed or (i == len(path) - 1 and depth > 0):
                raise motion.Blocked(f"{arm} would fold onto itself ({depth * 1000:.1f} mm)")
            if held is not None:
                site = self.model.site(lab.pinch_site(arm)).id
                tube_z = work.site_xmat[site].reshape(3, 3) @ held.mat[:, 2]
                if math.degrees(math.acos(np.clip(tube_z[2], -1, 1))) > MAX_LEAN:
                    raise motion.Blocked(f"{arm} would tip its tube past {MAX_LEAN:.0f} degrees")
        # Everything else as usual, without the arm's own folding (just checked).
        saved = side.folds_onto_itself
        motion.side.folds_onto_itself = lambda *a: False
        try:
            self.planner.check(data, arm, path, held, set(move.ignore))
        finally:
            motion.side.folds_onto_itself = saved

    @staticmethod
    def _held(grip) -> motion.Held:
        """The tube relative to the pinch point, from where the pinch point sits in the tube's frame."""
        pos, rot = grip
        return motion.Held("", -rot.T @ pos, rot.T)

    def plan(self, arm: str, tube: str, options: list[tuple[int, np.ndarray, str]]) -> Plan:
        """The first cell, and way of holding the tube (how upright, where it is lifted to, the gripper's
        turn round it), whose whole sequence the arm can make without touching anything.

        The start (away from the pour, up to the carrying height) does not depend on the cell, so each
        start is planned once and kept.
        """
        t0 = time.time()
        mover = Mover(self.model, arm)
        grip = mover.grip(self.data, tube)
        reasons = []
        starts: dict[tuple, tuple | None] = {}

        def follow(moves, work, now) -> tuple:
            for move in moves:
                path, held = self.path(arm, tube, move, work, mover, grip, now)
                if held is not None:
                    held.name = tube
                if path:
                    self.check(work, arm, path, held, move)
                    work.qpos[mover.joints] = path[-1]
                    mujoco.mj_kinematics(self.model, work)
                if move.kind == "tube":
                    now = move.target
            return now

        ways = [(share, lift, turn) for share in SHARES for lift in self.lifts(arm, tube) for turn in TURNS]
        for cell, bottom, rack in options:
            for i, (share, lift, turn) in enumerate(ways):
                if i not in starts:
                    work = mujoco.MjData(self.model)
                    work.qpos[:] = self.data.qpos
                    mujoco.mj_kinematics(self.model, work)
                    now = (self.data.body(tube).xpos.copy(), self.data.body(tube).xmat.reshape(3, 3).copy())
                    first = self.start_moves(arm, tube, turn, lift, share)
                    try:
                        starts[i] = (first, work.qpos.copy(), follow(first, work, now))
                    except motion.Blocked as e:
                        starts[i] = None
                        reasons.append(f"start {share}, {np.round(lift, 2)}, {turn}: {e}")
                if starts[i] is None:
                    continue
                first, qpos, at = starts[i]
                work = mujoco.MjData(self.model)
                work.qpos[:] = qpos
                mujoco.mj_kinematics(self.model, work)
                rest = self.end_moves(tube, rack, bottom, at)
                try:
                    follow(rest, work, at)
                except motion.Blocked as e:
                    reasons.append(f"cell {cell}, {share}, {np.round(lift, 2)}, turn {turn}: {e}")
                    continue
                self.reasons = reasons
                lifted = tuple(np.round(lift - self.data.body(tube).xpos, 2).tolist())
                return Plan(cell, bottom, turn, first + rest, share, lifted, len(reasons), time.time() - t0)
        self.reasons = reasons
        raise motion.Blocked(
            f"{arm} has no clear way to put {tube} down: {len(reasons)} tried, e.g. {reasons[:3]}"
        )

    # ---------------------------------------------------------------- moving

    def run(self, arm: str, tube: str, plan: Plan) -> None:
        """Make the planned moves, each planned again from where the tube and the arm are now."""
        mover = Mover(self.model, arm)
        gripper = lab.gripper_actuator_id(self.model, arm)
        for move in plan.moves:
            if move.kind == "open":
                self.data.ctrl[gripper] = lab.grip_command(side.PRESHAPE_GAP, squeeze=0.0)
                for _ in range(int(LET_GO / self.model.opt.timestep)):
                    self.step()
                self.say(arm, "let go")
                continue
            grip = mover.grip(self.data, tube)
            now = (self.data.body(tube).xpos.copy(), self.data.body(tube).xmat.reshape(3, 3).copy())
            path, held = self.path(arm, tube, move, self.data, mover, grip, now)
            if held is not None:
                held.name = tube
            if not path:
                continue
            try:
                self.check(self.data, arm, path, held, move)
            except motion.Blocked:
                if move.kind == "joints":  # home is only tidying up: stay if the way is not clear
                    self.say(arm, "stays: no clear way home")
                    return
                raise
            for _ in motion.follow(self.model, self.data, arm, path, move.speed):
                self.step()
            if move.kind == "tube":  # how far the tube's bottom ended from where it was sent (mm)
                miss = np.linalg.norm(self.data.body(tube).xpos - move.target[0]) * 1000
                self.say(arm, f"tube {np.round(move.target[0], 3).tolist()}, {miss:.1f} mm off")
        self.say(arm, "home")


def depths(model: mujoco.MjModel, data: mujoco.MjData, arm: str, tube: str) -> tuple[float, float]:
    """How deep the arm presses into itself or the table, and how deep the arm or its tube press into
    anything else (balls aside), m, in the pose ``data`` holds."""
    mujoco.mj_collision(model, data)
    fold, other = 0.0, 0.0
    for c in data.contact[: data.ncon]:
        n1 = model.body(model.geom_bodyid[c.geom1]).name
        n2 = model.body(model.geom_bodyid[c.geom2]).name
        if n1.startswith(liquid.BALL_PREFIX) or n2.startswith(liquid.BALL_PREFIX):
            continue
        mine1, mine2 = n1.startswith(f"{arm}/") or n1 == tube, n2.startswith(f"{arm}/") or n2 == tube
        if mine1 and mine2:
            arm_only = n1 != tube and n2 != tube
            if arm_only and ("gripper/" in n1) != ("gripper/" in n2):
                fold = max(fold, -c.dist)
        elif mine1 or mine2:
            theirs = n2 if mine1 else n1
            ours = n1 if mine1 else n2
            if theirs == "world":
                if not ours.endswith(side.STANDING) and ours != tube:
                    fold = max(fold, -c.dist)
                elif ours == tube:
                    other = max(other, -c.dist)
            else:
                other = max(other, -c.dist)
    return fold, other


def check_leaving(model: mujoco.MjModel, data: mujoco.MjData, arm: str, tube: str, path: list) -> None:
    """For moves that start where the pour left things: the arm may start touching something (with
    medium tubes the two grippers end the pour touching lightly) or resting on itself, but no contact
    may get deeper, and by the end nothing but the arm's own resting may touch."""
    joints = lab.arm_joint_ids(model, arm)
    held = motion.grab(data, arm, tube)
    site = model.site(lab.pinch_site(arm)).id
    adr = model.jnt_qposadr[model.body(tube).jntadr[0]]
    work = mujoco.MjData(model)
    work.qpos[:] = data.qpos
    mujoco.mj_kinematics(model, work)
    fold0, other0 = depths(model, work, arm, tube)
    quat = np.zeros(4)
    for i, q in enumerate(path):
        work.qpos[joints] = q
        mujoco.mj_kinematics(model, work)
        mat = work.site_xmat[site].reshape(3, 3)
        work.qpos[adr : adr + 3] = work.site_xpos[site] + mat @ held.pos
        mujoco.mju_mat2Quat(quat, (mat @ held.mat).ravel())
        work.qpos[adr + 3 : adr + 7] = quat
        mujoco.mj_kinematics(model, work)
        fold, other = depths(model, work, arm, tube)
        if fold > fold0 + FOLD_SLACK:
            raise motion.Blocked(f"{arm} would fold onto itself ({fold * 1000:.1f} mm)")
        if other > other0 + TOUCH_SLACK or (i == len(path) - 1 and other > 0):
            raise motion.Blocked(f"{arm} would touch something ({other * 1000:.2f} mm deep)")


def untip(sim: dict, step: Callable[[], None]) -> float:
    """Arm B lifts the source straight up, still tipped (at 70 degrees it no longer pours), then takes
    the pour's way in backwards, joint for joint, to where Step 1b left it.

    The lift keeps the gripper's angle as it is. Re-aimed from where the tube sits in the fingers now
    (it has turned up to ~19 degrees in them), the gripper would jump towards Arm A's fingers, and tipping
    the turned tube fully upright would rest the wrist on the forearm. The way in was clear and never
    did; the tube simply ends leaning by as much as it has turned in the fingers, and is made upright
    later, just above its cell. Returns when the lift ended (s of simulated time).
    """
    from pour.tilt import ARM

    model, data = sim["model"], sim["data"]
    source = sim["meta"]["source"]
    mover = Mover(model, ARM)
    problem = None
    for away in UNTIP_AWAY:  # straight up first, then up and a little aside
        try:
            up = mover.hand_line(data, data.site_xpos[mover.site] + np.array(away))
            check_leaving(model, data, ARM, source, up)
            break
        except motion.Blocked as e:
            problem = e
    else:
        raise problem
    for _ in motion.follow(model, data, ARM, up, speed=SLOW):
        step()
    lifted = data.time
    way_in = sim["move_in"]
    # Join the way in where it is highest (the lip comes in from above), then follow it back.
    work = mujoco.MjData(model)
    work.qpos[:] = data.qpos
    heights = []
    for q in way_in:
        work.qpos[mover.joints] = q
        mujoco.mj_kinematics(model, work)
        heights.append(work.site_xpos[mover.site][2])
    k = int(np.argmax(heights))
    q0 = data.qpos[mover.joints].copy()
    n = max(2, math.ceil(np.abs(way_in[k] - q0).max() / 0.02))
    path = [q0 + (way_in[k] - q0) * i / n for i in range(1, n + 1)] + list(reversed(way_in[:k]))
    check_leaving(model, data, ARM, source, path)
    for _ in motion.follow(model, data, ARM, path, speed=0.08):
        step()
    return lifted


def tube_tilt(data: mujoco.MjData, name: str) -> float:
    return math.degrees(math.acos(np.clip(data.body(name).xmat.reshape(3, 3)[2, 2], -1, 1)))
