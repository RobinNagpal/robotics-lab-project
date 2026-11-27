"""Step 5, programmed: push the centrifuge's lid shut, and report the balance.

The lid rests open (98.7 degrees, held by its gas spring). The arm closes its fingers, puts the fingertips on
the lid's inside face halfway along it, and pushes it round its hinge, following the lid's swing. The latch
catches the lid within 1.7 degrees of shut. The wrist force sensor watches the push: if it rises past
``STOP_FORCE`` (the lid is shut, or something is in the way) the arm stops. Then the fingers back off and the
arm goes home. Why halfway along the lid, and why the centrifuge is turned: docs/lid-reach.md.

What the arm uses: the bench layout (the centrifuge's place, its lid's hinge, size and resting angle, like a
work cell's drawing), and the wrist force sensor. The lid's angle while pushing is not read from the
simulator: the fingertips lead and the lid's spring keeps it against them.

The report: the robot's own balance number, from the weights it read (Step 1's samples and Step 3's balance
tubes) and the holes it used, and the examiner's number from the simulator's true weights and where the tubes
really are (`grading.grade`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np
from centrifuge_scene import centrifuge
from centrifuge_scene.settings import LID_CLOSED, LID_OPEN
from motion import SAFE, Arm, Blocked
from moves import move_joints, move_pose, self_gap
from rotor import Loader as PoseFinder

# Where on the lid the fingertips push: this far from the hinge, as a part of the lid's length.
ALONG = 0.5
# The closed fingertips reach this far past the pinch point, along the fingers (measured in the model).
FINGERTIPS = 0.0177
# The fingertips start this far off the lid's face, then move in to touch it.
APPROACH = 0.03
# While pushing, the fingertips lead the lid's face by this much (a light, steady push).
LEAD = 0.002
SWING_SPEED = math.radians(30)  # rad/s of lid angle
CONTROL = 0.01  # s between joint target updates
STEPS = 80  # poses along the swing checked before moving (about 1.2 degrees apart)
STOP_FORCE = 60.0  # N: more than this and the push stops
BACK_OFF = 0.05  # m off the lid after it is shut
TURNS = [math.radians(t) for t in range(0, 360, 15)]
ENOUGH = 0.008  # m: a swing with this much room is taken without trying more starts
JUMP = 0.5  # rad: the most a joint may change between two checked poses


@dataclass
class Closing:
    latched: bool = False
    stop: str = ""
    seconds: float = 0.0
    peak_force_n: float = 0.0
    smallest_gap_mm: float = 10.0


def wrist_force(data: mujoco.MjData) -> np.ndarray:
    return (
        data.site("arm/wrist_ft").xmat.reshape(3, 3)
        @ data.sensor("arm/wrist_force").data
    )


class LidCloser:
    def __init__(self, arm: Arm, along: float = ALONG, rest: float | None = None):
        """``along``: where on the lid to push (part of its length from the hinge); ``rest``: the lid's resting
        angle, rad from shut (default: fully open, AutoBio's stop)."""
        self.arm = arm
        self.along = along
        self.rest = LID_OPEN - LID_CLOSED if rest is None else rest
        m = arm.model
        self.lid = m.body("centrifuge/lid").id
        self.adr = centrifuge.lid_qpos_adr(m)
        self.hinge = m.joint("centrifuge/lid").id
        # The lid's slab: the box among its colliding shapes (its length along the lid's z, thickness along y).
        slab = [
            g
            for g in range(m.ngeom)
            if m.geom_bodyid[g] == self.lid
            and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
            and m.geom_conaffinity[g]
        ]
        self.slab_pos = m.geom_pos[slab[0]].copy()
        self.slab_size = m.geom_size[slab[0]].copy()
        self.work = mujoco.MjData(m)

    # --- the lid's geometry (from the layout, worked out in a scratch copy) ------------------------------

    def _lid_at(self, angle: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """With the lid ``angle`` rad from shut: the point on the slab's middle line ALONG of the way from its
        hinge end, the slab's thickness direction, and the hinge axis."""
        m, w = self.arm.model, self.work
        w.qpos[:] = self.arm.data.qpos
        w.qpos[self.adr] = LID_CLOSED + angle
        mujoco.mj_kinematics(m, w)
        r_lid = w.xmat[self.lid].reshape(3, 3)
        z_local = self.slab_pos[2] - self.slab_size[2] + self.along * 2 * self.slab_size[2]
        middle = w.xpos[self.lid] + r_lid @ np.array(
            [self.slab_pos[0], self.slab_pos[1], z_local]
        )
        return middle, r_lid[:, 1].copy(), w.xaxis[self.hinge].copy()

    def push_pose(
        self, angle: float, gap: float, turn: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """The pinch point (position, rotation) with the closed fingertips ``gap`` off the lid's inside face at
        ``ALONG``, the lid ``angle`` rad from shut, the fingers pointing the way that point moves as it shuts;
        ``turn`` is the fingers' turn about that direction."""
        middle, normal, axis = self._lid_at(angle)
        a0, a1 = (angle, angle - 0.01) if angle >= 0.01 else (0.01, 0.0)
        push = self._lid_at(a1)[0] - self._lid_at(a0)[0]
        push /= np.linalg.norm(push)
        # The face the fingertips touch faces against the push: half the slab's thickness back from the middle.
        half = self.slab_size[1]
        face = middle - half * normal if normal @ push > 0 else middle + half * normal
        pinch = face - (FINGERTIPS + gap) * push
        y0 = np.cross(push, axis)
        y0 /= np.linalg.norm(y0)
        y = y0 * math.cos(turn) + np.cross(push, y0) * math.sin(turn)
        return pinch, np.column_stack([np.cross(y, push), y, push])

    def angles(self) -> np.ndarray:
        """The lid's swing from its resting angle to shut, STEPS poses."""
        return np.linspace(self.rest, 0.0, STEPS + 1)

    # --- planning ----------------------------------------------------------------------------------------

    def gap_with_lid_at(
        self, qs: list[np.ndarray], angles: list[float], fingers_may_touch: bool
    ) -> float:
        """The smallest gap along ``qs`` with the lid at the matching ``angles`` (the lid may be touched by the
        fingers while pushing)."""
        arm, data = self.arm, self.arm.data
        saved = data.qpos[self.adr]
        best = 0.010
        try:
            for q, a in zip(qs, angles, strict=True):
                data.qpos[self.adr] = LID_CLOSED + a
                g, _ = arm.clearance(
                    [q], allow=(self.lid,) if fingers_may_touch else ()
                )
                best = min(best, g)
        finally:
            data.qpos[self.adr] = saved
        return best

    def plan(self) -> tuple[list[float], list[np.ndarray], float]:
        """The joint angles along the whole swing, checked: (the fingers' turn at each pose, the poses, the
        smallest gap). The fingers' turn about the push is free and may change along the way (the lid turns
        the hand through 99 degrees; no single turn is reachable the whole way): at each lid angle the turn
        whose arm pose is nearest the last one is used. The first pose: every arm pose for every turn is tried
        as a start, and the start whose whole swing keeps the most room wins."""
        arm = self.arm
        angles = self.angles()
        loader_poses = PoseFinder(arm)
        best: tuple[float, list[float], list[np.ndarray]] | None = None
        now = arm.data.qpos[arm.joints].copy()
        for turn0 in TURNS[::2]:
            pos, rot = self.push_pose(angles[0], -LEAD, turn0)
            for q0 in loader_poses.poses_for(pos, rot, now, tries=20):
                turns, qs, q = [turn0], [q0], q0
                for a in angles[1:]:
                    options = []
                    for dt in (0, 15, -15, 30, -30):
                        t = turns[-1] + math.radians(dt)
                        pos, rot = self.push_pose(a, -LEAD, t)
                        try:
                            q2 = arm.solve(pos, rot, q)
                        except Blocked:
                            continue
                        jump = float(np.abs(q2 - q).max())
                        if jump <= JUMP:
                            options.append((jump, t, q2))
                    if not options:
                        break
                    _, t, q = min(options, key=lambda o: o[0])
                    turns.append(t)
                    qs.append(q)
                if len(qs) < len(angles):
                    continue
                gap = min(
                    self.gap_with_lid_at(qs, list(angles), True), self_gap(arm, qs)
                )
                if best is None or gap > best[0]:
                    best = (gap, turns, qs)
                if best[0] >= ENOUGH:
                    return best[1], best[2], best[0]
        if best is None or best[0] < SAFE:
            raise Blocked(
                f"no clear way to push the lid shut (best {best and round(best[0] * 1000, 1)} mm)"
            )
        return best[1], best[2], best[0]

    # --- doing it ----------------------------------------------------------------------------------------

    def close(self) -> Closing:
        arm, data = self.arm, self.arm.data
        out = Closing()
        t0 = arm.time
        turns, qs, gap = self.plan()
        out.smallest_gap_mm = round(gap * 1000, 1)
        data.ctrl[arm.gripper] = 255  # fingers shut: the fingertips push
        arm.wait(0.5)
        angles = self.angles()
        rest = angles[0]
        # To the start, fingertips APPROACH off the face (joints, sweep checked), then in to touch it.
        pos, rot = self.push_pose(rest, APPROACH, turns[0])
        move_joints(arm, arm.solve(pos, rot, qs[0]))
        move_pose(arm, self.push_pose(rest, 0.0, turns[0])[0], rot, allow=(self.lid,))
        # Push: the joints follow the planned poses as the lid's angle runs down (in between, a straight
        # line in joint angles); the fingertips lead the lid's face, its spring keeps it on them.
        still = []
        for _ in range(10):
            arm.step(5)
            still.append(wrist_force(data).copy())
        base = np.mean(still, axis=0)
        latch = centrifuge.latch_id(arm.model)
        table = np.array(qs)
        up = angles[::-1]  # np.interp needs rising angles
        angle = rest
        n = round(CONTROL / arm.model.opt.timestep)
        out.stop = "reached shut"
        while angle > 0.0:
            angle = max(0.0, angle - SWING_SPEED * CONTROL)
            q = np.array([np.interp(angle, up, table[::-1, j]) for j in range(6)])
            data.ctrl[arm.actuators] = q
            arm.step(n)
            f = float(np.linalg.norm(wrist_force(data) - base))
            out.peak_force_n = max(out.peak_force_n, f)
            if f > STOP_FORCE:
                out.stop = f"force {f:.0f} N at {math.degrees(angle):.0f} deg"
                break
            if data.eq_active[latch]:
                out.stop = (
                    f"latched (the fingers' target at {math.degrees(angle):.1f} deg)"
                )
                break
        rot = arm.data.site_xmat[arm.site].reshape(3, 3).copy()
        data.ctrl[arm.actuators] = data.qpos[arm.joints]  # hold where it is
        arm.wait(0.3)
        out.latched = bool(data.eq_active[latch])
        # Off the lid, then home.
        here = arm.pinch()
        back = here - BACK_OFF * rot[:, 2]
        try:
            move_pose(arm, back, rot, allow=(self.lid,))
        except Blocked as e:
            out.stop += f"; backing off: {e}"
        out.seconds = round(arm.time - t0, 1)
        return out
