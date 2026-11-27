"""Step 3's last part: cap the filled balance tube, as a lab would before spinning it.

    filled tube in the balance's holder
    ──► pick a spare cap off its peg on the balance's base (fingers down, Step 1's cap grip)
    ──► over the tube, down onto its rim, a half turn while it goes the last 6 mm
    ──► let go, read the balance (now with the cap) ──► the capped tube back to its cell by its cap

From then on the balance tube is held and loaded like a sample (Step 4's cap grip).

**A simplification:** the thread is not simulated. The spare cap is only drawn and moves with the hand; when
it reaches the tube's top, the tube's own cap is switched on (``tubes.put_cap_on``: drawn, touched, its 2 g
added) and the spare one is put away. The arm still has to bring the cap exactly there.

What the arm uses: the bench layout (the pegs, the holder, the cells), the cap's weight as the maker gives it
(2.0 g: Step 3 pours to the plan's weight less that), and the balance's display.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
from centrifuge_scene import scene, tubes
from centrifuge_scene.racks import floor_height
from centrifuge_scene.scale import pan_top
from centrifuge_scene.settings import TUBE
from motion import CAP_DEPTH, GRIP_SECONDS, HOLD_CAP, SAFE, Arm, Blocked, turn_between
from weigh import DROP, TRAVEL, TURNS, Weigher, grip_height

# The cap's weight, from the maker: Step 3 pours this much less than the plan's weight.
CAP_G = TUBE.cap_mass * 1000
# The cap comes down onto the rim to this far above where it sits, then turns SCREW_TURNS times 60 degrees
# while it goes the rest of the way.
SCREW_FROM = 0.006
SCREW_TURNS = 3
PARKED = np.array([0.0, 0.0, -1.0])  # where a spare cap goes once it is on a tube: under the table


class Capper:
    """Puts the spare caps on balance tubes, one per tube."""

    def __init__(self, arm: Arm):
        self.arm = arm
        self.weigher = Weigher(arm)
        self.gaps: list[float] = []
        self._follow = None

    # --- the spare cap moving with the hand -------------------------------------------------------------

    def _mocap(self, i: int) -> int:
        return int(self.arm.model.body_mocapid[self.arm.model.body(scene.SPARE_CAP.format(i)).id])

    def _hold(self, i: int) -> None:
        """From now on spare cap ``i`` moves with the hand, as it sits in the fingers now."""
        arm, data = self.arm, self.arm.data
        k = self._mocap(i)
        pinch, rot = arm.pinch(), data.site_xmat[arm.site].reshape(3, 3).copy()
        cap_rot = np.zeros(9)
        mujoco.mju_quat2Mat(cap_rot, data.mocap_quat[k])
        off_pos = rot.T @ (data.mocap_pos[k] - pinch)
        off_rot = rot.T @ cap_rot.reshape(3, 3)
        before = arm.on_step

        def follow(a: Arm) -> None:
            p = a.data.site_xpos[a.site]
            r = a.data.site_xmat[a.site].reshape(3, 3)
            a.data.mocap_pos[k] = p + r @ off_pos
            a.data.mocap_quat[k] = mat2quat(r @ off_rot)
            if before:
                before(a)

        self._follow = (before, k)
        arm.on_step = follow

    def _let_go_of_spare(self) -> None:
        before, k = self._follow
        self.arm.on_step = before
        self.arm.data.mocap_pos[k] = PARKED
        self._follow = None

    # --- the moves --------------------------------------------------------------------------------------

    def clear_turn(self, xy: np.ndarray, low: float, allow: tuple[int, ...]) -> float:
        """The clear finger turn nearest the one now, coming down over ``xy`` to ``low`` and closing there."""
        arm, w = self.arm, self.weigher
        now = arm.turn
        for t in sorted(TURNS, key=lambda t: abs(turn_between(now, t))):
            try:
                qs = w.descent(xy, low, t)
            except Blocked:
                continue
            g1, _ = arm.clearance(qs, allow=allow, fingers=w.open_fingers, holding=False)
            g2, _ = arm.clearance(qs[-1:], allow=allow, fingers=w.closed_fingers, holding=False)
            if min(g1, g2) >= SAFE:
                self.gaps.append(min(g1, g2))
                return now + turn_between(now, t)
        raise Blocked(f"no clear turn for the fingers at {np.round(xy, 3)}")

    def pick_cap(self, i: int) -> None:
        """Take spare cap ``i`` off its peg (the arm fingers-down at the travel height or home)."""
        arm = self.arm
        top = scene.spare_cap_top(i)
        xy, low = top[:2], top[2] - CAP_DEPTH
        turn = self.clear_turn(xy, low, ())
        self.gaps.append(arm.move_line(np.array([*xy, TRAVEL]), turn))
        arm.open()
        self.gaps.append(arm.move_line(np.array([*xy, low])))
        # Closed to the cap's width, no squeeze: the spare cap is only drawn (see the module's notes).
        arm.data.ctrl[arm.gripper] = scene.grip_command(HOLD_CAP, 0.0)
        arm.wait(GRIP_SECONDS)
        self._hold(i)
        self.gaps.append(arm.move_line(np.array([*xy, TRAVEL])))

    def put_on(self, tube: int, name: str) -> None:
        """Bring the held cap down onto the tube standing in the holder, turn it on, let go and rise."""
        arm, w = self.arm, self.weigher
        xy = w.holder_xy
        low = grip_height(pan_top())  # where the pinch point is for a capped tube in the holder
        allow = (tube, *w.on_balance)
        turn = self.clear_turn(xy, low, allow)
        self.gaps.append(arm.move_line(np.array([*xy, TRAVEL]), turn))
        self.gaps.append(arm.move_line(np.array([*xy, low + SCREW_FROM]), allow=allow))
        for k in range(1, SCREW_TURNS + 1):
            z = low + SCREW_FROM * (1 - k / SCREW_TURNS)
            self.gaps.append(arm.move_line(np.array([*xy, z]), arm.turn + math.radians(60), allow=allow))
        # Open first (the spare cap stays put: the hand does not move), then switch on the tube's own cap.
        # Switched on inside closed fingers, it was squeezed between the pads and flung the tube away.
        arm.open()
        self._let_go_of_spare()
        tubes.put_cap_on(arm.model, arm.data, name)
        self.gaps.append(arm.move_line(np.array([*xy, TRAVEL]), allow=(tube,)))

    def back_to_cell(self, tube: int, rack: int, xy: np.ndarray) -> None:
        """The capped tube from the holder back to its rack cell by its cap (as Step 1 puts a sample back)."""
        arm, w = self.arm, self.weigher
        t_rack, gap = w.rack_turn(xy, tube)
        self.gaps.append(gap)
        above = np.array([*w.holder_xy, TRAVEL])
        self.gaps.append(arm.move_line(np.array([*w.holder_xy, grip_height(pan_top())]), allow=(tube,)))
        arm.close_on(tube)
        self.gaps.append(arm.move_line(above, allow=w.on_balance))
        self.gaps.append(arm.move_line(np.array([*xy, TRAVEL]), t_rack))
        self.gaps.append(arm.move_line(np.array([*xy, grip_height(floor_height()) + DROP]), allow=(rack,)))
        arm.open()
        self.gaps.append(arm.move_line(np.array([*xy, TRAVEL]), allow=(tube,)))


def mat2quat(rot: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(rot).flatten())
    return q
