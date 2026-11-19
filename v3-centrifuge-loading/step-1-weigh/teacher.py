"""The teacher for the learned models: the programmed weighing, re-planned from wherever the arm is.

The programmed arm (``weigh.py``) plans a whole straight line, checks it, then plays it back. A teacher for a
learned model must instead answer "from exactly here, what is the right move?" at any moment, because:

- **nudged demonstrations** push the arm a few millimetres off course while recording, and the teacher steers
  back: the model sees how to recover from being slightly off (the "DART" method);
- **corrections after training** let the trained model drive and ask the teacher the right move in the places
  the model's own mistakes lead to (the "DAgger" method).

So every ``TICK`` (0.1 s) the teacher moves a reference point one step along the straight line to the next
waypoint (speeding up and slowing down gently) and sets the joint targets that put the pinch point there. The
arm's motors trail their targets by about 0.2 s; stepping from where the arm *is* would make it crawl, so the
reference runs ahead, and it restarts from the arm's real place whenever the two drift more than ``REANCHOR``
apart. The waypoints are the programmed arm's (the same
cells, holder, heights and finger turns, which scored 60 of 60). The step it chooses is the **action** that a
demonstration records; under a nudge the arm is sent somewhere slightly different, but the recorded action is
still the teacher's.

The teacher may read the simulator (it only labels demonstrations); the learned model only gets the pictures,
the joint angles, the gripper and a sentence.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from centrifuge_scene import scene
from centrifuge_scene.racks import floor_height
from centrifuge_scene.scale import pan_top

from motion import GRIP_SECONDS, GRIP_SQUEEZE, HOLD_CAP, OPEN, Arm, Blocked, Held, fingers_down, turn_between
from weigh import DROP, TRAVEL, Weigher, grip_height

TICK = 0.1  # s between the teacher's decisions (and the recorded frames)
FAST = 0.15  # m/s, in the open
SLOW = (
    0.04  # m/s, the last NEAR above a tube's place (the fingers round a cap, a tube in a cell or the holder)
)
NEAR = 0.07  # m: a tube's tip leaves a 6 cm rack cell within this lift
ACCEL = 0.5  # m/s^2, speeding up and slowing down
TURN_STEP = math.radians(6)  # the most the fingers turn in one tick
REACHED = 0.001  # m: close enough to a waypoint where the fingers then close or open (the arm also stopped)
PASSED = 0.004  # m: close enough to a waypoint in the open (no need to stop there)
TURN_OK = math.radians(1.0)
STILL = 0.05  # rad/s: the arm has stopped (before the fingers close or open)
WAYPOINT_TIMEOUT = 15.0  # s
# m: the reference restarts from the arm's real place if further than this (at full speed the arm trails it
# by about 3 cm; a nudge moves it 2 to 5 mm).
REANCHOR = 0.05


@dataclass
class Waypoint:
    """Where the pinch point goes next (``pos``, finger ``turn``), or a gripper action (``grip``)."""

    pos: np.ndarray | None = None
    turn: float | None = None
    slow: bool = False
    grip: str | None = None  # "open" or "close"
    tube: int = -1  # the tube the fingers close on


@dataclass
class Tick:
    """One decision: what the arm saw and what the teacher told it to do."""

    state: np.ndarray  # 6 joint angles, then the gripper's opening (0 open, 1 closed)
    action: np.ndarray  # 6 joint targets, then the gripper command (0 open, 1 closed)
    nudged: bool


@dataclass
class Episode:
    ticks: list[Tick] = field(default_factory=list)
    nudges: int = 0
    error: str = ""
    per_waypoint: list[int] = field(default_factory=list)  # ticks spent on each waypoint


class Nudger:
    """Pushes the arm a few millimetres (and degrees) off the teacher's step, now and then.

    Only in the open: never within ``KEEP_CLEAR`` above a place where the fingers grip or let go, where the
    open fingers pass caps with 5 mm to spare. A push lasts a few ticks; the teacher then steers back.
    """

    KEEP_CLEAR = 0.08  # m above a grip or release point
    NEAR = 0.05  # m sideways from it

    NOT_NEAR_GOAL = 0.03  # m: no push this close to the waypoint being approached
    MOST = 4  # pushes per episode

    def __init__(self, rng: np.random.Generator, chance: float = 0.12, size=(0.002, 0.005), turn_deg=3.0):
        self.rng, self.chance, self.size, self.turn = rng, chance, size, math.radians(turn_deg)
        self.left = 0  # ticks left in the current push
        self.cooldown = 0
        self.offset = np.zeros(3)
        self.dturn = 0.0
        self.count = 0

    def __call__(
        self, pinch: np.ndarray, low_points: list[np.ndarray], to_goal: float
    ) -> tuple[np.ndarray, float]:
        if self.left > 0:
            self.left -= 1
            if self.left == 0:
                self.cooldown = 5
            return self.offset, self.dturn
        if self.cooldown > 0:
            self.cooldown -= 1
            return np.zeros(3), 0.0
        near = any(
            np.linalg.norm(pinch[:2] - p[:2]) < self.NEAR and pinch[2] - p[2] < self.KEEP_CLEAR
            for p in low_points
        )
        near = near or to_goal < self.NOT_NEAR_GOAL or self.count >= self.MOST
        if near or self.rng.random() > self.chance:
            return np.zeros(3), 0.0
        angle = self.rng.uniform(0, 2 * math.pi)
        r = self.rng.uniform(*self.size)
        self.offset = np.array([r * math.cos(angle), r * math.sin(angle), self.rng.uniform(-0.002, 0.002)])
        self.dturn = self.rng.uniform(-self.turn, self.turn)
        self.left = int(self.rng.integers(3, 7))
        self.count += 1
        return self.offset, self.dturn


class Teacher:
    """Drives the arm through waypoints, deciding every TICK from where the arm is."""

    def __init__(self, arm: Arm):
        self.arm = arm
        m = arm.model
        self.drivers = [m.joint(f"arm/gripper/{s}_driver_joint").qposadr[0] for s in ("right", "left")]
        self.driver_closed = 0.8  # the driver joint's angle fully closed (rad, Robotiq 2F-85)
        self.speed = 0.0
        self.ref: np.ndarray | None = None  # the reference point the teacher steers the pinch point to
        self.ref_turn = 0.0

    # --- what the arm sees -------------------------------------------------------------------------------

    def state(self) -> np.ndarray:
        d = self.arm.data
        opening = float(np.mean(d.qpos[self.drivers])) / self.driver_closed
        return np.concatenate([d.qpos[self.arm.joints], [np.clip(opening, 0.0, 1.0)]])

    def grip_value(self) -> float:
        """The gripper command now, as 0 (open) to 1 (closed)."""
        return float(self.arm.data.ctrl[self.arm.gripper]) / 255.0

    # --- one tick ----------------------------------------------------------------------------------------

    def anchor(self) -> None:
        """Restart the reference from where the arm really is, if it has drifted away from it."""
        here = self.arm.pinch()
        if self.ref is None or np.linalg.norm(self.ref - here) > REANCHOR:
            self.ref = here
            self.ref_turn = self.arm.turn_now()

    def step_towards(self, wp: Waypoint) -> tuple[np.ndarray, float, float]:
        """Move the reference one tick towards ``wp``: (pinch target, finger turn, distance it has left)."""
        self.anchor()
        gap = wp.pos - self.ref
        dist = float(np.linalg.norm(gap))
        limit = SLOW if wp.slow else FAST
        # Speed up gently, and slow down so as to stop at the waypoint.
        self.speed = min(limit, self.speed + ACCEL * TICK, math.sqrt(2 * ACCEL * dist))
        step = min(dist, max(self.speed * TICK, 0.0005))
        if dist > 1e-9:
            self.ref = self.ref + gap / dist * step
        goal_turn = wp.turn if wp.turn is not None else self.ref_turn
        self.ref_turn += float(np.clip(turn_between(self.ref_turn, goal_turn), -TURN_STEP, TURN_STEP))
        return self.ref.copy(), self.ref_turn, float(np.linalg.norm(wp.pos - self.ref))

    def drive(self, q_goal: np.ndarray) -> None:
        """Slide the joint targets to ``q_goal`` over one tick."""
        a = self.arm
        start = a.data.ctrl[a.actuators].copy()
        n = int(round(TICK / a.model.opt.timestep))
        for i in range(1, n + 1):
            a.data.ctrl[a.actuators] = start + (q_goal - start) * i / n
            a.step()

    def run(
        self,
        waypoints: list[Waypoint],
        record: Callable[[Tick], None] | None = None,
        nudger: Nudger | None = None,
    ) -> Episode:
        """Go through ``waypoints``; each tick is passed to ``record``. Stops at the first problem."""
        a = self.arm
        ep = Episode()
        lows = [w.pos for w in waypoints if w.pos is not None and w.slow]
        try:
            for i, wp in enumerate(waypoints):
                exact = i + 1 == len(waypoints) or waypoints[i + 1].grip is not None
                ep.per_waypoint.append(len(ep.ticks))
                if wp.grip:
                    self._grip(wp, record, ep)
                    continue
                self.speed = 0.0
                ticks = 0
                while True:
                    off = float(np.linalg.norm(wp.pos - a.pinch()))
                    still = np.abs(a.data.qvel[a.dofs]).max() < STILL
                    turn_left = abs(turn_between(a.turn_now(), wp.turn)) if wp.turn is not None else 0.0
                    if exact and off < REACHED and turn_left < TURN_OK and still:
                        break
                    if not exact and off < PASSED and turn_left < 3 * TURN_OK:
                        break
                    target, turn, dist = self.step_towards(wp)
                    q_label = a.solve(target, fingers_down(turn), a.data.qpos[a.joints].copy())
                    offset, dturn = nudger(a.pinch(), lows, off) if nudger else (np.zeros(3), 0.0)
                    nudged = bool(np.any(offset) or dturn)
                    q_drive = (
                        a.solve(target + offset, fingers_down(turn + dturn), q_label) if nudged else q_label
                    )
                    tick = Tick(self.state(), np.concatenate([q_label, [self.grip_value()]]), nudged)
                    if record:
                        record(tick)
                    ep.ticks.append(tick)
                    self.drive(q_drive)
                    ticks += 1
                    if ticks * TICK > WAYPOINT_TIMEOUT:
                        raise Blocked(f"did not reach {np.round(wp.pos, 3)}")
                if wp.turn is not None:
                    a.turn = a.turn_now()
        except Blocked as e:
            ep.error = str(e)
        ep.nudges = nudger.count if nudger else 0
        return ep

    def _grip(self, wp: Waypoint, record, ep: Episode) -> None:
        a = self.arm
        if wp.grip == "close":
            a.data.ctrl[a.gripper] = scene.grip_command(HOLD_CAP, GRIP_SQUEEZE)
            a.held = Held(wp.tube)
        else:
            a.data.ctrl[a.gripper] = scene.grip_command(OPEN, squeeze=0.0)
            a.held = None
        hold = a.data.ctrl[a.actuators].copy()
        for _ in range(int(round(GRIP_SECONDS / TICK))):
            tick = Tick(self.state(), np.concatenate([hold, [self.grip_value()]]), False)
            if record:
                record(tick)
            ep.ticks.append(tick)
            self.drive(hold)


# --- the two tasks ------------------------------------------------------------------------------------------


def _down_up(xy_low: np.ndarray, turn: float, middle: list[Waypoint]) -> list[Waypoint]:
    """Waypoints down to ``xy_low`` (fast, then slow for the last NEAR), ``middle``, and back up."""
    above = xy_low.copy()
    above[2] = TRAVEL
    near = xy_low + [0, 0, NEAR]
    down = [Waypoint(near, turn), Waypoint(xy_low, turn, slow=True)]
    up = [Waypoint(near, turn, slow=True), Waypoint(above, turn)]
    return down + middle + up


def to_holder(w: Weigher, xy: np.ndarray, tube: int, t_rack: float, t_hold: float) -> list[Waypoint]:
    """From wherever the arm is: pick the tube in the cell at ``xy`` and stand it in the balance's holder."""
    rack_grip = np.array([*xy, grip_height(floor_height())])
    release = np.array([*w.holder_xy, grip_height(pan_top()) + DROP])
    return [
        Waypoint(np.array([*xy, TRAVEL]), t_rack),
        Waypoint(grip="open"),
        *_down_up(rack_grip, t_rack, [Waypoint(grip="close", tube=tube)]),
        Waypoint(np.array([*w.holder_xy, TRAVEL]), t_hold),
        *_down_up(release, t_hold, [Waypoint(grip="open")]),
    ]


def back_to_cell(w: Weigher, xy: np.ndarray, tube: int, t_rack: float, t_hold: float) -> list[Waypoint]:
    """From above the holder: pick the tube up again and put it back in its rack cell at ``xy``."""
    holder_grip = np.array([*w.holder_xy, grip_height(pan_top())])
    release = np.array([*xy, grip_height(floor_height()) + DROP])
    return [
        Waypoint(np.array([*w.holder_xy, TRAVEL]), t_hold),
        *_down_up(holder_grip, t_hold, [Waypoint(grip="close", tube=tube)]),
        Waypoint(np.array([*xy, TRAVEL]), t_rack),
        *_down_up(release, t_rack, [Waypoint(grip="open")]),
    ]
