"""Step 3, programmed: fill an empty tube with water until it weighs what the plan says.

For one balance tube (docs/approach.md):

    empty tube: supply rack ──► balance holder ──► let go ──► read its empty weight
    water tube (side grip): supply rack ──► above the balance ──► tipped, its lip just above the balance tube
        ──► tip on until water flows ──► pour, watching the balance ──► tip back early ──► settle, read
        ──► short? a small extra pour ──► upright ──► back to its cell
    balance tube ──► back to its cell in the supply rack (Step 4 takes it from there to the rotor)

What the arm uses: the bench layout (cells, holder), which cells hold the empty tubes and the water (Step 0),
the target weight (Step 2) and the balance's display. Never how many balls are where.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from centrifuge_scene.racks import cell_centres, floor_height
from centrifuge_scene.scale import Balance, pan_top
from centrifuge_scene.settings import HOME, LAYOUT, TUBE
from motion import MARGIN, OPEN, SAFE, Arm, Blocked, fingers_down, turn_between
from weigh import TRAVEL, TURNS, tube_at

from moves import (
    DROP,
    HOLD_BODY,
    HOLD_TOP,
    REST,
    UP,
    HeldAt,
    close_on_open_tube,
    hand_rot,
    move_pose,
    plan_pose,
    reach,
    rot_about,
    self_gap,
    side_grip_height,
    side_rot,
    top_grip_height,
)

# --- where the water tube is held to pour ---------------------------------------------------------------
# The water tube's lip (the point of its rim nearest the balance tube) goes this far above the balance
# tube's rim, and this far short of its axis (towards the water tube). Nearer the axis the balls overshoot
# the far wall; lower, the two tubes touch at small tilts (grasp-plan, section 3).
LIP_UP = 0.020
LIP_BACK = 0.004
# The water tube is tipped this far on the way in, comes in from this far above its place, then straight
# down. Less tipped, the fingers come within 2 mm of the balance tube at the pour place. No water leaves
# until about 82 degrees (results/tilt_study_*.json).
MOVE_IN_TILT = 60.0
LIFT = 0.05
MAX_TILT = 135.0
CONTROL = 0.01  # s between joint target updates while tipping
# Side grips: approach directions tried at a rack cell, every 15 degrees round.
SIDE_APPROACHES = [math.radians(a) for a in range(0, 360, 15)]
SIDE_BACK = 0.10  # m: the fingers come in level from this far out
SIDE_LIFT = 0.10  # m: a water tube is lifted this far straight up out of its cell (the rack is 6 cm tall)
# The way water is poured, as seen from above (degrees from the x axis, which points away from the arm's
# base): -90 pours from the racks' side of the balance tube towards it.
POUR_YAW = -90.0
# Where the fingers come from while pouring, in degrees from the pour direction about the vertical: 0 is from
# behind (the pads then press on the tube's left and right, on the line it is tipped about, and the tube
# turns between them like a door on its hinge); 90 or -90 is from the side, the pads on its front and back.
# -90: squarely from the side, so tipping presses the tube into the pads instead of turning it between them
# (15 degrees off, the tube still turned 4 degrees in the fingers and the lip ended up 6 mm to the side).
# With this pour direction the fingers point back towards the arm's base, far from the wrist's singular pose
# (pouring away from the base instead, a hand from the side lines the fingers up with the wrist's joints).
POUR_HAND = -90.0

# --- the pour, watching the balance ---------------------------------------------------------------------
# The "water" is balls, and balls do not flow like water: nothing leaves until about 82 degrees, then they
# come in bursts of 10 to 20 (3 to 7 g) as the pile at the mouth jams and lets go (the tilt study).
# Between bursts the reading is exact; during one, every ball landing kicks it by up to 5 g. So the pour
# goes in rounds: creep up slowly until a ball lands, then either let the burst run (much still to add) or
# tip straight back (little to add), wait for the reading to settle, read, and creep again from just short
# of that angle.
APPROACH_TILT = 70.0  # deg: tipped fast up to here (the first single ball fell at 73 to 74 degrees)
FAST_RATE = 10.0  # deg/s
CREEP_RATE = 0.7  # deg/s, as v1's pour
FINE_RATE = 0.3  # deg/s when less than FINE_G is still to add: time to tip back after the first ball
FINE_G = 2.0
BACK_RATE = 40.0  # deg/s, tipping back when nothing is flowing
# To stop a flow the tilt target jumps back at once: ramped at 40 deg/s, the arm (which trails its targets by
# about 0.2 s) let whole bursts out, up to 15 balls more than wanted (results/evaluation_run1.log).
STOP_RATE = 1000.0
ARRIVED_G = (
    0.25  # g above the last settled reading: a ball has landed (its impact reads more than its weight)
)
LET_RUN_G = 4.0  # g still to add, at least, for a burst to be let run
# g: a burst that is let run is stopped when the reading comes this close to the target. Bursts carry on up
# to 1.5 g past the stop (results/evaluation_run1.json), so 2 g left too little for the fine rounds.
STOP_SHORT_G = 3.0
TIP_BACK = 10.0  # deg tipped back to stop the flow
RESTART = 2.0  # deg short of where the flow last started, the next creep starts
SETTLE_BAND, SETTLE_TIME, SETTLE_TIMEOUT = 0.02, 0.3, 4.0  # g, s, s: the reading is settled
MAX_ROUNDS = 40


@dataclass
class Fill:
    """One balance tube: where things stood, what the balance read, and how it went."""

    blank_cell: int
    water_cell: int
    target_g: float
    empty_g: float | None = None
    poured_g: float | None = None  # the settled reading when the pour ended
    final_g: float | None = None  # read again once the water tube is back in its cell
    steady: bool = False
    over: bool = False
    rounds: list[Round] = field(default_factory=list)
    seconds: float = 0.0
    smallest_gap_mm: float = MARGIN * 1000
    error: str = ""

    @property
    def off_g(self) -> float | None:
        return None if self.final_g is None else round(self.final_g - self.target_g, 2)


@dataclass
class Round:
    """One round of the pour: where the flow started, and what the settled reading was after it."""

    start_deg: float
    flow_deg: float | None
    let_run: bool
    stopped_deg: float | None
    before_g: float
    after_g: float


class Filler:
    """Fills balance tubes one after another with the programmed arm."""

    def __init__(self, arm: Arm):
        self.arm = arm
        m = arm.model
        self.balance = Balance(m, arm.data)
        self.holder_xy = np.array(LAYOUT.scale)
        self.supply = m.body("supply_rack").id
        self.on_balance = (m.body("scale").id, m.body("scale/pan").id)
        self.open_fingers = arm.fingers(OPEN)
        self.top_fingers = arm.fingers(HOLD_TOP)
        self.body_fingers = arm.fingers(HOLD_BODY)
        self.pour_dir = np.array([math.cos(math.radians(POUR_YAW)), math.sin(math.radians(POUR_YAW)), 0.0])
        self.pour_hand = POUR_HAND
        self.gaps: list[float] = []
        self.on_tick = None  # called every CONTROL s while tipping, with (filler, tilt): for recording
        self.samples: list[float] = []  # the balance read every CONTROL s during the pour

    # --- the balance -------------------------------------------------------------------------------------

    def reading(self, seconds: float = 0.1) -> float:
        """The display averaged over ``seconds`` while the simulation runs (it flickers by about 0.02 g)."""
        n = max(1, int(round(seconds / 0.01)))
        values = []
        for _ in range(n):
            self.arm.step(10)
            values.append(self.balance.grams())
        return float(np.mean(values))

    def read_steady(self, timeout: float = 8.0) -> tuple[float, bool]:
        """Wait until 0.1 s averages stay within 0.01 g for 0.5 s: (grams, steady). As Step 1."""
        averages: list[float] = []
        for _ in range(int(timeout / 0.1)):
            averages.append(round(self.reading(0.1), 2))
            last = averages[-5:]
            if len(last) == 5 and max(last) - min(last) <= 0.0101:
                return round(float(np.mean(last)), 2), True
        return round(float(np.mean(averages[-5:])), 2), False

    # --- open tubes, held from above ---------------------------------------------------------------------

    def descent(self, xy: np.ndarray, low: float, turn: float) -> list[np.ndarray]:
        top = np.array([*xy, TRAVEL])
        q_top = self.arm.solve_any(top, fingers_down(turn))
        return [q_top, *self.arm.plan_line(np.array([*xy, low]), turn, start=(top, turn, q_top))]

    def top_turn(self, xy: np.ndarray, low: float, tube: int, allow: tuple[int, ...]) -> float:
        """The turn with most room to come down over the open tube at ``xy`` and close on its top."""
        best = (-1.0, 0.0)
        for t in TURNS:
            try:
                qs = self.descent(xy, low, t)
            except Blocked:
                continue
            g1, _ = self.arm.clearance(qs, allow=allow, fingers=self.open_fingers, holding=False)
            g2, _ = self.arm.clearance(qs[-1:], allow=(tube, *allow), fingers=self.top_fingers, holding=False)
            gap = min(g1, g2)
            if gap > best[0] + 1e-4:
                best = (gap, t)
        if best[0] < SAFE:
            raise Blocked(f"no clear turn for the fingers at {np.round(xy, 3)}")
        self.gaps.append(best[0])
        return best[1]

    def pick_top(self, xy: np.ndarray, surface: float, tube: int, allow: tuple[int, ...]) -> float:
        """Pick the open tube standing at ``xy`` on ``surface`` by its top. Returns the turn used."""
        arm = self.arm
        low = top_grip_height(surface)
        turn = self.top_turn(xy, low, tube, allow)
        self.hand_down()
        self.gaps.append(arm.move_line(np.array([*xy, TRAVEL]), turn))
        arm.open()
        self.gaps.append(arm.move_line(np.array([*xy, low]), allow=allow))
        pinch, rot = arm.pinch(), hand_rot(arm)
        tip = np.array([*xy, surface + REST])
        close_on_open_tube(arm, HeldAt.planned(tube, pinch, rot, tip, np.eye(3)), HOLD_TOP)
        self.gaps.append(arm.move_line(np.array([*xy, TRAVEL]), allow=allow))
        return turn

    def hand_down(self) -> None:
        """After a side grip: back to the start pose, fingers down, so Step 1's fingers-down moves can carry
        on. (From the side grip's joint angles, turning the hand down on the way passed a joint limit.)"""
        arm = self.arm
        if hand_rot(arm)[2, 2] < -0.999:
            return
        if arm.pinch()[2] < TRAVEL - 0.005:
            self.gaps.append(move_pose(arm, np.array([*arm.pinch()[:2], TRAVEL]), hand_rot(arm)))
        self.go_home()

    def go_home(self) -> None:
        """Slide the joints to the start pose, the sweep checked for collisions first (unlike Step 1's)."""
        arm = self.arm
        q0 = arm.data.qpos[arm.joints].copy()
        qs = [q0 + (np.array(HOME) - q0) * s for s in np.linspace(0, 1, 60)[1:]]
        gap, what = arm.clearance(qs)
        if gap < SAFE or self_gap(arm, qs) < SAFE:
            raise Blocked(f"going home: {what or 'the hand and the arm'} would come too close")
        self.gaps.append(gap)
        arm.home()

    def place_top(self, xy: np.ndarray, surface: float, allow: tuple[int, ...], turn: float) -> None:
        """Stand the held open tube at ``xy`` on ``surface``, let go and rise clear."""
        arm = self.arm
        tube = arm.held.body
        self.gaps.append(arm.move_line(np.array([*xy, TRAVEL]), turn))
        self.gaps.append(arm.move_line(np.array([*xy, top_grip_height(surface) + DROP]), allow=allow))
        arm.open()
        self.gaps.append(arm.move_line(np.array([*xy, TRAVEL]), allow=(tube, *allow)))

    def holder_turn(self) -> float:
        """The clear turn at the holder nearest the fingers' turn now (as Step 1)."""
        low = top_grip_height(pan_top()) + DROP
        now = self.arm.turn
        for t in sorted(TURNS, key=lambda t: abs(turn_between(now, t))):
            try:
                qs = self.descent(self.holder_xy, low, t)
            except Blocked:
                continue
            g1, _ = self.arm.clearance(qs, allow=self.on_balance)
            g2, _ = self.arm.clearance(qs, fingers=self.open_fingers, holding=False)
            if min(g1, g2) >= SAFE:
                self.gaps.append(min(g1, g2))
                return now + turn_between(now, t)
        raise Blocked("no clear turn for the fingers at the balance")

    # --- water tubes, held from the side -----------------------------------------------------------------

    def side_poses(self, xy: np.ndarray, approach: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Pinch positions for the side pick at ``xy``: (coming in level, lifting out). One hand rotation."""
        low = side_grip_height(floor_height())
        grip = np.array([*xy, low])
        come_in = [grip - SIDE_BACK * approach + [0, 0, TRAVEL - low], grip - SIDE_BACK * approach, grip]
        lift = [grip + [0, 0, SIDE_LIFT]]
        return come_in, lift

    def side_approach(self, xy: np.ndarray, tube: int) -> np.ndarray:
        """The level approach with most room to pick the water tube at ``xy`` from the side."""
        arm = self.arm
        base = np.array([*(xy / np.linalg.norm(xy)), 0.0])
        best: tuple = ((-1.0, 0.0), None)
        for a in SIDE_APPROACHES:
            approach = rot_about(UP, a) @ base
            rot = side_rot(approach)
            come_in, lift = self.side_poses(xy, approach)
            try:
                q_top = arm.solve_any(come_in[0], rot)
                qs = [q_top]
                start = (come_in[0], rot, q_top)
                for p in come_in[1:]:
                    qs += plan_pose(arm, p, rot, start=start)
                    start = (p, rot, qs[-1])
                q_lift = plan_pose(arm, lift[0], rot, start=start)
            except Blocked:
                continue
            g1, _ = arm.clearance(qs, fingers=self.open_fingers, holding=False)
            g2, _ = arm.clearance(qs[-1:], allow=(tube,), fingers=self.body_fingers, holding=False)
            # Lifting out with the tube: as if held (the planner's copy carries it with the hand).
            arm.held = HeldAt.planned(
                tube, come_in[-1], rot, np.array([*xy, floor_height() + REST]), np.eye(3)
            )
            g3, _ = arm.clearance(q_lift, allow=(self.supply,), fingers=self.body_fingers)
            arm.held = None
            gap = min(g1, g2, g3, self_gap(arm, [*qs, *q_lift]))
            # Equally clear: the one nearest "away from the arm's base" (keeps the hand far from the arm).
            key = (round(gap, 4), -abs(math.remainder(a, 2 * math.pi)))
            if gap >= SAFE and (best[1] is None or key > best[0]):
                best = (key, approach)
        if best[1] is None:
            raise Blocked(f"no clear side approach to the water tube at {np.round(xy, 3)}")
        self.gaps.append(best[0][0])
        return best[1]

    def pick_side(self, xy: np.ndarray, tube: int) -> np.ndarray:
        """Pick the water tube at ``xy`` from the side and lift it out. Returns the approach used."""
        arm = self.arm
        approach = self.side_approach(xy, tube)
        rot = side_rot(approach)
        come_in, lift = self.side_poses(xy, approach)
        if arm.pinch()[2] < TRAVEL - 0.005:
            self.gaps.append(move_pose(arm, np.array([*arm.pinch()[:2], TRAVEL]), hand_rot(arm)))
        self.gaps.append(reach(arm, come_in[0], rot))
        arm.open()
        for p in come_in[1:]:
            self.gaps.append(move_pose(arm, p, rot))
        tip = np.array([*xy, floor_height() + REST])
        close_on_open_tube(arm, HeldAt.planned(tube, come_in[-1], rot, tip, np.eye(3)), HOLD_BODY)
        self.gaps.append(move_pose(arm, lift[0], rot, allow=(self.supply,)))
        return approach

    def place_side(self, xy: np.ndarray, approach: np.ndarray) -> None:
        """Stand the held water tube back in its cell at ``xy``, let go, and back out level."""
        arm = self.arm
        tube = arm.held.body
        rot = side_rot(approach)
        come_in, lift = self.side_poses(xy, approach)
        # To above the cell at the travel height, the joints slid to the pick's grip there, then down
        # (straight from the balance, the wrist swept through the balance tube).
        self.gaps.append(reach(arm, np.array([*lift[0][:2], TRAVEL]), rot))
        self.gaps.append(move_pose(arm, lift[0], rot))
        self.gaps.append(move_pose(arm, come_in[-1] + [0, 0, DROP], rot, allow=(self.supply,)))
        arm.open()
        self.gaps.append(move_pose(arm, come_in[1] + [0, 0, DROP], rot, allow=(tube,)))
        self.gaps.append(move_pose(arm, come_in[0], rot))

    # --- the pour ----------------------------------------------------------------------------------------

    def rim(self) -> np.ndarray:
        """The centre of the balance tube's rim, standing in the holder (from the layout)."""
        return np.array([*self.holder_xy, pan_top() + REST + TUBE.height])

    def pour_pinch(self, tilt_deg: float, lift: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """The hand's place to hold the water tube tipped ``tilt_deg`` towards the balance tube, its lip
        in place."""
        d = self.pour_dir
        u = np.cross(UP, d)
        tilt = rot_about(u, math.radians(tilt_deg))
        # The tube as it is held upright, the fingers coming in from POUR_HAND degrees off the pour direction.
        hand = rot_about(UP, math.radians(self.pour_hand)) @ d
        upright = side_rot(hand) @ self.arm.held.rot_local
        tube_rot = tilt @ upright
        lip = self.rim() + [0, 0, LIP_UP + lift] - LIP_BACK * d
        tip = lip - TUBE.height * (tilt @ UP) - TUBE.outline[-1][1] * (tilt @ d)
        return self.arm.held.pinch_for(tip, tube_rot)

    def move_in(self) -> None:
        """From the water tube lifted out of its cell to its pour place, tipped MOVE_IN_TILT degrees."""
        arm = self.arm
        # Upright, high enough that its bottom clears the balance tube's rim by 2 cm.
        up_pos, up_rot = self.pour_pinch(0.0, TUBE.height + 0.02)
        self.gaps.append(move_pose(arm, np.array([*arm.pinch()[:2], TRAVEL]), hand_rot(arm)))
        self.gaps.append(reach(arm, np.array([*up_pos[:2], TRAVEL]), up_rot))
        self.gaps.append(move_pose(arm, up_pos, up_rot))
        pos, rot = self.pour_pinch(MOVE_IN_TILT, LIFT)
        self.gaps.append(move_pose(arm, pos, rot))
        pos, rot = self.pour_pinch(MOVE_IN_TILT)
        self.gaps.append(move_pose(arm, pos, rot))

    def check_tilts(self, low: float, high: float) -> None:
        """Before tipping: the whole range of tilts at the pour place must be clear."""
        arm = self.arm
        q = arm.data.qpos[arm.joints].copy()
        qs = []
        for t in np.arange(low, high + 0.1, 2.0):
            pos, rot = self.pour_pinch(float(t))
            q = arm.solve(pos, rot, q)
            qs.append(q)
        gap, what = arm.clearance(qs)
        if gap < SAFE:
            raise Blocked(f"tipping: {what} would come {gap * 1000:.1f} mm close")
        self.gaps.append(gap)

    def tip(self, tilt: float, goal: float, rate: float, until=None) -> tuple[float, bool]:
        """Tip from ``tilt`` towards ``goal`` at ``rate`` deg/s, until ``until()`` says stop: (tilt, stopped).

        Every CONTROL s the balance is read (``self.samples``) and the joint targets are moved on.
        """
        arm = self.arm
        q = arm.data.ctrl[arm.actuators].copy()
        while True:
            if until is not None and until():
                return tilt, True
            if abs(goal - tilt) < 1e-6:
                return tilt, False
            tilt += float(np.clip(goal - tilt, -rate * CONTROL, rate * CONTROL))
            pos, rot = self.pour_pinch(tilt)
            q = arm.solve(pos, rot, q, iterations=50)
            arm.data.ctrl[arm.actuators] = q
            self.hold(tilt, 1)

    def hold(self, tilt: float, ticks: int) -> None:
        """Keep the targets where they are for ``ticks`` CONTROL periods, reading the balance each one."""
        for _ in range(ticks):
            self.arm.step(int(round(CONTROL / self.arm.model.opt.timestep)))
            self.samples.append(self.balance.grams())
            if self.on_tick:
                self.on_tick(self, tilt)

    def recent(self, seconds: float) -> list[float]:
        return self.samples[-max(1, int(round(seconds / CONTROL))) :]

    def settle(self, tilt: float) -> float:
        """Wait (holding the tilt) until the reading stays within SETTLE_BAND for SETTLE_TIME: the grams."""
        need = int(round(SETTLE_TIME / CONTROL))
        for _ in range(int(SETTLE_TIMEOUT / CONTROL)):
            self.hold(tilt, 1)
            last = self.samples[-need:]
            if len(last) == need and max(last) - min(last) <= SETTLE_BAND:
                break
        return float(np.median(self.recent(SETTLE_TIME)))

    def pour(self, target_g: float, empty_g: float, fill: Fill) -> float:
        """Pour from the water tube (at its pour place, tipped MOVE_IN_TILT) until the balance tube reads
        ``target_g``. Returns the tilt it is left at, tipped back below any flow."""
        ball = 0.3265  # g: the reading goes up in steps of this (measured, docs/results.md)
        self.samples = []
        tilt, _ = self.tip(MOVE_IN_TILT, APPROACH_TILT, FAST_RATE)
        settled = empty_g
        restart = APPROACH_TILT
        for _ in range(MAX_ROUNDS):
            short = target_g - settled
            if short <= ball / 2:
                fill.over = short < -ball / 2
                break
            # Back up to just short of where water last flowed, fast; then creep until a ball lands.
            tilt, _ = self.tip(tilt, max(restart, APPROACH_TILT), FAST_RATE)
            start = tilt
            rate = FINE_RATE if short < FINE_G else CREEP_RATE
            tilt, arrived = self.tip(
                tilt, MAX_TILT, rate, until=lambda s=settled: self.samples[-1] > s + ARRIVED_G
            )
            if not arrived:
                raise Blocked(f"no more water at {tilt:.0f} degrees: the water tube is empty")
            flow = tilt
            let_run = short >= LET_RUN_G
            stopped = None
            if let_run:
                # Hold the tilt while the burst runs; stop it if it comes too close to the target.
                need = int(round(SETTLE_TIME / CONTROL))
                for _ in range(int(SETTLE_TIMEOUT / CONTROL)):
                    self.hold(tilt, 1)
                    if float(np.median(self.recent(0.1))) >= target_g - STOP_SHORT_G:
                        stopped = tilt
                        break
                    last = self.samples[-need:]
                    if len(last) == need and max(last) - min(last) <= SETTLE_BAND:
                        break  # the burst is over by itself
            if not let_run or stopped is not None:
                stopped = tilt
                tilt, _ = self.tip(tilt, tilt - TIP_BACK, STOP_RATE)
            after = self.settle(tilt)
            fill.rounds.append(
                Round(round(start, 1), round(flow, 1), let_run, stopped, round(settled, 2), round(after, 2))
            )
            settled = after
            restart = flow - RESTART
        fill.poured_g = round(settled, 2)
        tilt, _ = self.tip(tilt, MOVE_IN_TILT, BACK_RATE)
        return tilt

    def away_from_pour(self) -> None:
        """From the pour place (tipped below any flow): straight up until the water tube's bottom will clear
        the balance tube upright, then upright, then up to the travel height."""
        arm = self.arm
        pos, rot = self.pour_pinch(MOVE_IN_TILT, TUBE.height + 0.02)
        self.gaps.append(move_pose(arm, pos, rot))
        pos, rot = self.pour_pinch(0.0, TUBE.height + 0.02)
        self.gaps.append(move_pose(arm, pos, rot))
        self.gaps.append(move_pose(arm, np.array([*pos[:2], TRAVEL]), rot))

    # --- one balance tube --------------------------------------------------------------------------------

    def fill(self, blank_cell: int, water_cell: int, target_g: float) -> Fill:
        """Fill the empty tube in supply ``blank_cell`` from the water tube in ``water_cell`` to ``target_g``,
        and put both back in their cells. Errors are caught and reported."""
        arm = self.arm
        result = Fill(blank_cell, water_cell, round(target_g, 2))
        start = arm.time
        self.gaps = []
        cells = cell_centres(LAYOUT.supply_rack)
        supply = (self.supply,)
        try:
            bxy, wxy = np.array(cells[blank_cell]), np.array(cells[water_cell])
            blank, water = tube_at(arm, bxy), tube_at(arm, wxy)
            # The empty tube into the balance's holder, and its empty weight.
            turn = self.pick_top(bxy, floor_height(), blank, supply)
            self.place_top(self.holder_xy, pan_top(), self.on_balance, self.holder_turn())
            result.empty_g, _ = self.read_steady()
            # The water tube, from the side, to its pour place; pour; and back to its cell.
            approach = self.pick_side(wxy, water)
            self.move_in()
            self.check_tilts(MOVE_IN_TILT, MAX_TILT)
            self.pour(target_g, result.empty_g, result)
            self.away_from_pour()
            self.place_side(wxy, approach)
            result.final_g, result.steady = self.read_steady()
            # The filled tube back to its cell, held by its top as it was picked.
            self.pick_top(self.holder_xy, pan_top(), blank, self.on_balance)
            self.place_top(bxy, floor_height(), supply, turn)
        except Blocked as e:
            result.error = str(e)
        result.seconds = round(arm.time - start, 1)
        result.smallest_gap_mm = round(min(self.gaps, default=MARGIN) * 1000, 1)
        return result
