"""Step 3's moves: what Step 1's arm (step-1-weigh/motion.py) could not do yet.

- **Any turn of the hand.** Step 1's arm only moves with its fingers pointing straight down. Pouring needs
  the hand turned every way, so a move here is a straight line for the pinch point while the hand turns
  evenly from its start turn to its end turn (``plan_pose``). As in Step 1, every 5 mm (or every 3 degrees
  of turn) is solved for joint angles and the whole move is checked for collisions before the arm moves.
- **Open tubes.** A balance tube has no cap: it is held from above on its top, the pinch point 6 mm under
  the rim, squeezed 1 mm (the grip plan's physics check: 4 mm throws the tube out of the fingers).
- **The side grip** for the water tubes: the fingers come in level, from the side away from the balance tube,
  35 mm under the rim, so the opening stays free to pour (grasp-plan/docs/results.md, sections 3 and 4).

A held tube sits in the fingers where the planned grip put it (``HeldAt``): the planner's copy of the scene
carries it along with the hand to check for collisions. Nothing here reads where the tube really is.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import mujoco
import numpy as np
from centrifuge_scene import scene
from centrifuge_scene.settings import TUBE
from motion import GRIP_SECONDS, MARGIN, MAX_JUMP, PIECE, SAFE, Arm, Blocked

UP = np.array([0.0, 0.0, 1.0])
TURN_PIECE = math.radians(3)  # rad of hand turn between solved points
REST = 0.001  # a tube's tip rests this far above the surface it stands on (as in Step 1)
DROP = 0.002  # let go of a tube this far above where it rests (as in Step 1)

# Open tubes (the balance tubes), held from above: pinch point 6 mm under the rim. The 38 mm pads reach
# 19 mm either side of it, so they close on the top 15 mm of the 29 mm body (below the 28 mm threads).
# Tested on 8 seeds against a grip 25 mm under the rim: into the holder, the tube leans at most 1.7 degrees
# (25 mm: up to 4.8); a full tube put back in its cell at most 2.0 degrees (25 mm: up to 4.3).
TOP_DEPTH = 0.006
HOLD_TOP = 2 * TUBE.radius
# Water tubes, held from the side: pinch point 40 mm under the rim, on the 29 mm body. Shallower, the
# fingers touch the balance tube when the water tube is tipped to pour; deeper, the hand hits the rack's top
# plate when it picks the tube (docs/results.md).
SIDE_DEPTH = 0.040
HOLD_BODY = 2 * TUBE.radius
# How far past a tube's width the fingers are told to close on an open tube (grasp-plan, section 7).
OPEN_SQUEEZE = 0.001


def rot_about(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """The angle of the turn from rotation ``a`` to rotation ``b``, rad."""
    return math.acos(float(np.clip((np.trace(a.T @ b) - 1) / 2, -1.0, 1.0)))


def turn_part(a: np.ndarray, b: np.ndarray, s: float) -> np.ndarray:
    """The rotation ``s`` of the way (0 to 1) along the shortest turn from ``a`` to ``b``."""
    qa, qb = np.zeros(4), np.zeros(4)
    mujoco.mju_mat2Quat(qa, a.flatten())
    mujoco.mju_mat2Quat(qb, b.flatten())
    if qa @ qb < 0:
        qb = -qb
    dot = float(np.clip(qa @ qb, -1.0, 1.0))
    if dot > 0.9995:
        q = qa + s * (qb - qa)
    else:
        th = math.acos(dot)
        q = (math.sin((1 - s) * th) * qa + math.sin(s * th) * qb) / math.sin(th)
    q /= np.linalg.norm(q)
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, q)
    return out.reshape(3, 3)


def side_rot(approach: np.ndarray, axis: np.ndarray = UP) -> np.ndarray:
    """The pinch point's rotation for a side grip: fingers along ``approach``, across a tube along ``axis``.

    The pinch site's z axis runs along the fingers and its y axis is the way they close (as in motion.py).
    """
    z = approach - (approach @ axis) * axis
    z /= np.linalg.norm(z)
    y = np.cross(axis, z)
    return np.column_stack([np.cross(y, z), y, z])


@dataclass
class HeldAt:
    """A tube held where the grip put it: its tip and turn, seen from the pinch point."""

    body: int
    tip_local: np.ndarray
    rot_local: np.ndarray

    @classmethod
    def planned(cls, body: int, pinch: np.ndarray, rot: np.ndarray, tip: np.ndarray, tube_rot: np.ndarray):
        return cls(body, rot.T @ (tip - pinch), rot.T @ tube_rot)

    def tube_pose(self, pinch: np.ndarray, rot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return pinch + rot @ self.tip_local, rot @ self.rot_local

    def pinch_for(self, tip: np.ndarray, tube_rot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Where the pinch point must be for the tube's tip to be at ``tip``, turned ``tube_rot``."""
        rot = tube_rot @ self.rot_local.T
        return tip - rot @ self.tip_local, rot

    def tube_qpos(self, pinch: np.ndarray, rot: np.ndarray) -> np.ndarray:
        """The tube's free-joint position (tip, then quaternion), for the planner's collision check."""
        tip, tube_rot = self.tube_pose(pinch, rot)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, tube_rot.flatten())
        return np.concatenate([tip, quat])


# The arm touching itself: the hand (wrist_3 and the gripper) against the arm's big links. Step 1's moves,
# fingers always down, never fold the arm that far; turned hands can (the gripper jammed against the upper
# arm while it was building this step).
BIG_LINKS = ("arm/base", "arm/shoulder_link", "arm/upper_arm_link", "arm/forearm_link")


def is_hand(name: str) -> bool:
    return name == "arm/wrist_3_link" or name.startswith("arm/gripper/")


def self_gap(arm: Arm, qs: Iterable[np.ndarray]) -> float:
    """The smallest gap along ``qs`` between the hand and the arm's own big links, m (up to MARGIN)."""
    m = arm.model
    work = mujoco.MjData(m)
    work.qpos[:] = arm.data.qpos
    best = MARGIN
    saved = m.geom_margin.copy()
    hand = np.array([is_hand(n) for n in arm.body_names])
    big = np.array([n in BIG_LINKS for n in arm.body_names])
    try:
        m.geom_margin[hand[m.geom_bodyid] | big[m.geom_bodyid]] = MARGIN
        for q in qs:
            work.qpos[arm.joints] = q
            mujoco.mj_kinematics(m, work)
            mujoco.mj_collision(m, work)
            for c in work.contact[: work.ncon]:
                b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
                if (hand[b1] and big[b2]) or (hand[b2] and big[b1]):
                    best = min(best, c.dist)
    finally:
        m.geom_margin[:] = saved
    return best


def hand_rot(arm: Arm) -> np.ndarray:
    return arm.data.site_xmat[arm.site].reshape(3, 3).copy()


def plan_pose(
    arm: Arm,
    end: np.ndarray,
    rot: np.ndarray,
    start: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> list[np.ndarray]:
    """Joint angles along a straight line for the pinch point to ``end``, the hand turning evenly to ``rot``.

    From where the arm is now, or from ``start`` = (pinch position, hand rotation, joint angles).
    """
    if start is None:
        a, r0, q = arm.pinch(), hand_rot(arm), arm.data.qpos[arm.joints].copy()
    else:
        a, r0, q = start
    end = np.asarray(end, dtype=float)
    steps = max(1, math.ceil(np.linalg.norm(end - a) / PIECE), math.ceil(angle_between(r0, rot) / TURN_PIECE))
    out = []
    for i in range(1, steps + 1):
        s = i / steps
        q2 = arm.solve(a + (end - a) * s, turn_part(r0, rot, s), q)
        if np.abs(q2 - q).max() > MAX_JUMP:
            raise Blocked(f"the arm would flip on the way to {np.round(end, 3)}")
        out.append(q2)
        q = q2
    return out


def move_pose(
    arm: Arm, end: np.ndarray, rot: np.ndarray, allow: Iterable[int] = (), safe: float = SAFE
) -> float:
    """Check, then make, a straight move to ``end`` turning the hand to ``rot``. Returns the smallest gap."""
    qs = plan_pose(arm, end, rot)
    gap, what = arm.clearance(qs, allow)
    if gap < safe:
        raise Blocked(f"moving to {np.round(end, 3)}: {what} would come {gap * 1000:.1f} mm close")
    own = self_gap(arm, qs)
    if own < safe:
        raise Blocked(f"moving to {np.round(end, 3)}: the hand would come {own * 1000:.1f} mm from the arm")
    arm.follow(qs)
    return gap


def move_joints(arm: Arm, q_end: np.ndarray, allow: Iterable[int] = (), safe: float = SAFE) -> float:
    """Slide the joints straight to ``q_end``, checked for collisions every 0.02 rad first. For big turns of
    the hand up at the travel height, where a straight line for the pinch point can pass the wrist's singular
    pose or a joint limit."""
    q0 = arm.data.qpos[arm.joints].copy()
    n = max(2, math.ceil(float(np.abs(q_end - q0).max()) / 0.02))
    qs = [q0 + (q_end - q0) * s for s in np.linspace(0, 1, n + 1)[1:]]
    gap, what = arm.clearance(qs, allow)
    if gap < safe:
        raise Blocked(f"turning the joints: {what} would come {gap * 1000:.1f} mm close")
    own = self_gap(arm, qs)
    if own < safe:
        raise Blocked(f"turning the joints: the hand would come {own * 1000:.1f} mm from the arm")
    arm.follow(qs)
    return gap


def reach(arm: Arm, end: np.ndarray, rot: np.ndarray, allow: Iterable[int] = ()) -> float:
    """Go to ``end`` with the hand turned ``rot``, by sliding the joints (``move_joints``) to a solution
    found from the arm's angles now or from home turned towards ``end``."""
    return move_joints(arm, arm.solve_any(np.asarray(end, dtype=float), rot), allow)


def close_on_open_tube(arm: Arm, held: HeldAt, width: float) -> None:
    """Close the fingers ``OPEN_SQUEEZE`` past ``width`` on an open tube, and carry it from now on."""
    arm.data.ctrl[arm.gripper] = scene.grip_command(width, OPEN_SQUEEZE)
    arm.wait(GRIP_SECONDS)
    arm.held = held


def top_grip_height(surface: float) -> float:
    """The pinch point's height for an open tube standing on ``surface``, held from above."""
    return surface + REST + TUBE.height - TOP_DEPTH


def side_grip_height(surface: float) -> float:
    """The pinch point's height for an open tube standing on ``surface``, held from the side."""
    return surface + REST + TUBE.height - SIDE_DEPTH


# Standing a held tube down: how far below the planned height the hand may go looking for the seat, how fast,
# and the change in the wrist force (N) that means the tube's tip has touched it.
TOUCH_EXTRA = 0.020
TOUCH_SPEED = 0.01  # m/s
TOUCH_FORCE = 1.0


def lower_until_touch(arm: Arm, rot: np.ndarray, allow: Iterable[int] = (), extra: float = TOUCH_EXTRA) -> float:
    """Lower the held tube straight down until the wrist force sensor feels it touch what it stands on, at
    most ``extra`` m below where the hand is. Returns how far it went down (m).

    A tube can slide along the fingers while it is tipped to pour (12 mm on seed 5 once the pads resisted
    twisting): let go at the planned height, it then falls that far, lands crooked in the rack's small seat
    and the simulation blows up. Feeling for the seat does not need to know where the tube sits in the hand.
    """
    data = arm.data
    start = arm.pinch()
    # Only as far down as is clear (the hand comes close to the rack's top plate).
    low = start.copy()
    for d in np.arange(0.002, extra + 1e-9, 0.002):
        try:
            q = arm.solve(start - [0, 0, d], rot, data.qpos[arm.joints].copy())
        except Blocked:
            break
        if arm.clearance([q], allow)[0] < SAFE:
            break
        low = start - [0, 0, d]

    def force() -> np.ndarray:
        return data.site("arm/wrist_ft").xmat.reshape(3, 3) @ data.sensor("arm/wrist_force").data

    still = []
    for _ in range(10):
        arm.step(10)
        still.append(force().copy())
    base = np.mean(still, axis=0)
    q = data.qpos[arm.joints].copy()
    z = start[2]
    while z > low[2]:
        if np.linalg.norm(force() - base) > TOUCH_FORCE:
            break
        z = max(low[2], z - TOUCH_SPEED * 0.01)
        q = arm.solve(np.array([*start[:2], z]), rot, q)
        data.ctrl[arm.actuators] = q
        arm.step(10)
    # Hold where the arm is (it trails its targets), so it does not press the tube on into the seat.
    data.ctrl[arm.actuators] = data.qpos[arm.joints]
    arm.wait(0.2)
    return float(start[2] - arm.pinch()[2])
