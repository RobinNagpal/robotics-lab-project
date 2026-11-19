"""Step 1, programmed: weigh each sample tube on the balance, and put it back in its cell.

For one tube (docs/approach.md):

    above its cell ──► down, fingers open either side of the cap ──► close ──► straight up out of the rack
    ──► across to above the balance ──► down into the holder ──► open ──► up clear
    ──► wait for a steady reading ──► read
    ──► down, close ──► up ──► back above its cell ──► down ──► open ──► up

What the arm uses: the bench layout (where the racks' cells and the balance's holder are), which cells hold
a sample (Step 0's answer), and the balance's display. It never reads a tube's weight or how much liquid it
holds from the simulator.

**Which way the fingers close** at a rack cell is chosen by trying 12 turns (every 15°) and keeping the one
with most room to the neighbouring tubes, checked for the way down with the fingers open and for the fingers
closed on the cap (their follower links swing out). The same turn puts the tube back. At the holder, nothing
stands near, so the clear turn nearest the fingers' turn is used (no needless wrist spin).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import mujoco
import numpy as np
from centrifuge_scene.racks import cell_centres, floor_height
from centrifuge_scene.scale import Balance, pan_top
from centrifuge_scene.settings import LAYOUT, TUBE

from motion import CAP_DEPTH, HOLD_CAP, MARGIN, OPEN, SAFE, Arm, Blocked, fingers_down, turn_between

# The pinch point's height while carrying: a held tube's tip then clears the holder (12.5 cm) and every cap
# (12.4 cm) by 3 cm.
TRAVEL = 0.30
# A tube's tip rests this far above the surface it stands on (the rack's floor, the balance's pan): its cone
# sits on the walls of the small seat there. Measured once in the simulation, like a work cell's setup.
REST = 0.001
# Let go of a tube this far above where it rests, so it is never pushed into its seat.
DROP = 0.002
TURNS = [math.radians(t) for t in range(0, 180, 15)]
# The display is read every READ_EVERY s. The liquid's balls never lie perfectly still, so the raw reading
# flickers by about 0.02 g; like a real balance's filter, readings are averaged over AVERAGE s. The weight is
# steady when those averages stay within 0.01 g of each other for STEADY s; give up after TIMEOUT.
READ_EVERY = 0.01
AVERAGE = 0.1
STEADY = 0.5
TIMEOUT = 8.0


@dataclass
class Weighing:
    """One sample: where it stood, what the balance read, and how it went."""

    rack: str
    cell: int
    grams: float | None = None
    steady: bool = False
    seconds: float = 0.0  # simulated time for the whole weighing
    turn_rack_deg: float | None = None
    turn_holder_deg: float | None = None
    smallest_gap_mm: float = MARGIN * 1000
    error: str = ""
    readings: list[float] = field(default_factory=list, repr=False)

    @property
    def ok(self) -> bool:
        return self.grams is not None and self.steady and not self.error


def grip_height(surface: float) -> float:
    """The pinch point's height for a capped tube standing on ``surface``."""
    return surface + REST + TUBE.capped_height - CAP_DEPTH


def tube_at(arm: Arm, xy: np.ndarray) -> int:
    """The planner's copy of the scene: the tube body standing at ``xy`` (it is what Step 0 found there)."""
    m, d = arm.model, arm.data
    best, body = 0.015, -1
    for b in range(m.nbody):
        j = m.body_jntadr[b]
        if j < 0 or m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE or arm.is_ball[b]:
            continue
        name = arm.body_names[b]
        if name.endswith("_rack"):
            continue
        dist = float(np.linalg.norm(d.xpos[b][:2] - xy))
        if dist < best:
            best, body = dist, b
    if body < 0:
        raise Blocked(f"no tube stands at {np.round(xy, 3)}")
    return body


class Weigher:
    """Weighs sample tubes one after another with the programmed arm."""

    def __init__(self, arm: Arm):
        self.arm = arm
        m = arm.model
        self.balance = Balance(m, arm.data)
        self.holder_xy = np.array(LAYOUT.scale)
        self.racks = {r: m.body(r).id for r in ("sample_rack", "supply_rack")}
        self.on_balance = (m.body("scale").id, m.body("scale/pan").id)
        self.open_fingers = arm.fingers(OPEN)
        self.closed_fingers = arm.fingers(HOLD_CAP)

    # --- choosing the fingers' turn ----------------------------------------------------------------------

    def descent(self, xy: np.ndarray, low: float, turn: float) -> list[np.ndarray]:
        """Joint angles for coming straight down from TRAVEL to ``low`` over ``xy``, fingers at ``turn``."""
        top = np.array([*xy, TRAVEL])
        q_top = self.arm.solve_any(top, fingers_down(turn))
        return [q_top, *self.arm.plan_line(np.array([*xy, low]), turn, start=(top, turn, q_top))]

    def rack_turn(self, xy: np.ndarray, tube: int) -> tuple[float, float]:
        """The turn with most room to pick the tube at ``xy`` (and put it back): (turn, smallest gap m)."""
        low = grip_height(floor_height())
        best = (-1.0, 0.0)
        for t in TURNS:
            try:
                qs = self.descent(xy, low, t)
            except Blocked:
                continue
            # The open fingers on the way down (they may touch the rack: never, but it is not a tube)...
            g1, _ = self.arm.clearance(qs, fingers=self.open_fingers, holding=False)
            # ...and closed on the cap at the bottom, where their follower links swing out.
            g2, _ = self.arm.clearance(qs[-1:], allow=(tube,), fingers=self.closed_fingers, holding=False)
            gap = min(g1, g2)
            if gap > best[0] + 1e-4:
                best = (gap, t)
        if best[0] < SAFE:
            raise Blocked(f"no clear turn for the fingers at {np.round(xy, 3)}")
        return best[1], best[0]

    def holder_turn(self) -> tuple[float, float]:
        """The clear turn at the holder nearest the fingers' turn now: (turn, smallest gap m)."""
        low = grip_height(pan_top()) + DROP
        now = self.arm.turn
        for t in sorted(TURNS, key=lambda t: abs(turn_between(now, t))):
            try:
                qs = self.descent(self.holder_xy, low, t)
            except Blocked:
                continue
            g1, _ = self.arm.clearance(qs, allow=self.on_balance)  # coming down with the tube
            g2, _ = self.arm.clearance(qs, fingers=self.open_fingers, holding=False)  # going up without it
            if min(g1, g2) >= SAFE:
                return now + turn_between(now, t), min(g1, g2)
        raise Blocked("no clear turn for the fingers at the balance")

    # --- the balance -------------------------------------------------------------------------------------

    def read_steady(self) -> tuple[float, bool, list[float]]:
        """Wait until the averaged reading stops changing for STEADY s: (grams, steady, the averages)."""
        every = int(round(READ_EVERY / self.arm.model.opt.timestep))
        per_average = int(round(AVERAGE / READ_EVERY))
        window = int(round(STEADY / AVERAGE))
        averages: list[float] = []
        for _ in range(int(TIMEOUT / AVERAGE)):
            raw = []
            for _ in range(per_average):
                self.arm.step(every)
                raw.append(self.balance.grams())
            averages.append(round(float(np.mean(raw)), 2))
            last = averages[-window:]
            if len(last) == window and max(last) - min(last) <= 0.0101:
                return round(float(np.mean(last)), 2), True, averages
        return round(float(np.mean(averages[-window:])), 2), False, averages

    # --- one tube ----------------------------------------------------------------------------------------

    def weigh(self, rack: str, cell: int) -> Weighing:
        """Weigh the sample in ``rack``'s ``cell`` and put it back. Errors are caught and reported."""
        arm = self.arm
        result = Weighing(rack, cell)
        start = arm.time
        gaps: list[float] = []
        try:
            xy = np.array(cell_centres(getattr(LAYOUT, rack))[cell])
            tube = tube_at(arm, xy)
            rack_body = self.racks[rack]
            t_rack, gap = self.rack_turn(xy, tube)
            result.turn_rack_deg = round(math.degrees(t_rack) % 180, 1)
            gaps.append(gap)
            rack_grip = np.array([*xy, grip_height(floor_height())])

            # Pick it from the rack: above it, fingers opened to 45 mm (they start fully open, 85 mm, which
            # the turn was not chosen for), then down either side of the cap.
            gaps.append(arm.move_line(np.array([*xy, TRAVEL]), t_rack))
            arm.open()
            gaps.append(arm.move_line(rack_grip))
            arm.close_on(tube)
            gaps.append(arm.move_line(np.array([*xy, TRAVEL]), allow=(rack_body,)))

            # Stand it in the balance's holder, let go, move clear, read.
            t_hold, gap = self.holder_turn()
            result.turn_holder_deg = round(math.degrees(t_hold) % 180, 1)
            gaps.append(gap)
            above = np.array([*self.holder_xy, TRAVEL])
            gaps.append(arm.move_line(above, t_hold))
            gaps.append(
                arm.move_line(
                    np.array([*self.holder_xy, grip_height(pan_top()) + DROP]), allow=self.on_balance
                )
            )
            arm.open()
            gaps.append(arm.move_line(above, allow=(tube,)))
            result.grams, result.steady, result.readings = self.read_steady()

            # Pick it up again and put it back in its cell.
            gaps.append(arm.move_line(np.array([*self.holder_xy, grip_height(pan_top())]), allow=(tube,)))
            arm.close_on(tube)
            gaps.append(arm.move_line(above, allow=self.on_balance))
            gaps.append(arm.move_line(np.array([*xy, TRAVEL]), t_rack))
            gaps.append(arm.move_line(rack_grip + [0, 0, DROP], allow=(rack_body,)))
            arm.open()
            gaps.append(arm.move_line(np.array([*xy, TRAVEL]), allow=(tube,)))
        except Blocked as e:
            result.error = str(e)
        result.seconds = arm.time - start
        result.smallest_gap_mm = round(min(gaps, default=MARGIN) * 1000, 1)
        return result

    def weigh_all(self, cells: list[tuple[str, int]]) -> list[Weighing]:
        """Tare the empty balance, weigh every sample in ``cells`` in turn, then go home."""
        self.arm.wait(0.3)
        self.balance.tare()
        out = []
        for rack, cell in cells:
            w = self.weigh(rack, cell)
            out.append(w)
            if w.error:
                break  # a refused move leaves the arm where it stopped: do not carry on blindly
        if not out or not out[-1].error:
            self.arm.home()
        return out


def run(model: mujoco.MjModel, data: mujoco.MjData, cells: list[tuple[str, int]], on_step=None):
    """Weigh the samples in ``cells``: (list of Weighing, wall-clock seconds)."""
    t0 = time.time()
    arm = Arm(model, data, on_step)
    results = Weigher(arm).weigh_all(cells)
    return results, time.time() - t0
