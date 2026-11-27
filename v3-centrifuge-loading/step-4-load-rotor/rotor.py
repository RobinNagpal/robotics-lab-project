"""Step 4's moves: put one tube into one rotor hole (the programmed arm; nothing here is learned).

For one tube (docs/approach.md, docs/insert-test.md):

    pick it from its rack cell (samples: Step 1's grip on the cap; balance tubes: Step 3's grip on the top)
    ──► up to the travel height ──► turn the joints so the hand is tilted to the hole's 35 degrees, the tube
        20 cm (holes 4 and 5: 10 cm) back along the hole's axis
    ──► slide in along the axis, watching the wrist force sensor:
          pushed back more than 4 N? ── wait (near the bottom: the tip is down, stop)
    ──► hold the arm where it is ──► open ──► back out along the axis

What the arm uses: the bench layout (rack cells, the rotor's holes), which cells hold which tube (Step 0),
the plan (Step 2) and the wrist force sensor. Nothing here reads where a tube really is: the examiner's
numbers are worked out by the callers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np
from centrifuge_scene import scene
from centrifuge_scene.racks import cell_centres, floor_height
from centrifuge_scene.settings import HOME, LAYOUT, ROTOR, TUBE
from fill import Filler
from motion import (
    CAP_DEPTH,
    GRIP_SQUEEZE,
    HOLD_CAP,
    OPEN,
    SAFE,
    Arm,
    Blocked,
    fingers_down,
)
from moves import (
    HOLD_TOP,
    OPEN_SQUEEZE,
    TOP_DEPTH,
    HeldAt,
    hand_rot,
    move_joints,
    move_pose,
    plan_pose,
    rot_about,
    self_gap,
)
from weigh import REST, TRAVEL, Weigher, grip_height, tube_at

# How far back along the hole's axis the slide starts (20 cm for holes 0 to 3; holes 4 and 5 lean towards the
# arm, which only reaches 10 cm back there).
BACK = {0: 0.20, 1: 0.20, 2: 0.20, 3: 0.20, 4: 0.10, 5: 0.10}
# The slide: the tube goes in at FAST, then SLOW from SLOW_FROM above the hole's top (along the axis).
FAST, SLOW = 0.03, 0.01  # m/s
SLOW_FROM = 0.015
CONTROL = 0.01  # s between joint target updates
SLIDE_TIMEOUT = 40.0  # s
# The push back along the axis (N, from the reading before the slide): above PAUSE_FORCE the slide waits (a
# jam at the lead-in, or the bottom); within NEAR_BOTTOM of the bottom it means the tip is down: stop.
# Waiting longer than JAM_SECONDS anywhere else is a jam.
PAUSE_FORCE = 4.0
NEAR_BOTTOM = 0.008
JAM_SECONDS = 3.0
# "force" slides: the hand gives way sideways like a soft spring, GIVE metres per newton of sideways push (at
# most MAX_SHIFT), easing towards that over GIVE_TIME; when the push goes, the hand comes back.
GIVE = 0.0003
MAX_SHIFT = 0.003
GIVE_TIME = 0.1
SIDE_DEADBAND = 0.5  # N
# A slide that pauses more than RETRY_PAUSES times (or RETRY_SECONDS in all) inside the hole is wedged: back
# the tube RETRY_BACK out and slide again, at most RETRIES times; then stop without letting go.
RETRY_PAUSES = 25
RETRY_SECONDS = 1.5
RETRY_BACK = 0.015
RETRIES = 2
# On a retry the fingers hold the tube this lightly (m past its width), so it can line itself up.
LOOSE_SQUEEZE = 0.0005
# Let go this far short of the bottom (as Steps 1 and 3 let go 2 mm above a seat): the tube drops in.
DROP = 0.002
# Finger openings tried when letting go in a hole, widest first (m).
RELEASE = (OPEN, 0.041, 0.039)
BACK_OUT = (
    0.10,
    0.08,
    0.06,
)  # m along the axis after letting go, the first that is clear
# Finger turns tried at a hole, degrees from "closing along the rim". With both neighbours empty the clear
# ones are within 45 degrees of along the rim (grasp-plan); with a full neighbour only a narrow window.
HOLE_TURNS = list(range(-90, 91, 5))
SAME_GAP = 0.0005
# An open tube with water may lean at most this far on the way to a hole (it is 35 degrees in the hole;
# Step 3: balls start to leave a full tube at about 82 degrees).
MAX_TILT = 40.0  # m: clear turns this close count as equally clear


@dataclass(frozen=True)
class Grip:
    """How a tube is held: on the cap (capped samples) or on the top of the open tube (balance tubes)."""

    capped: bool

    @property
    def depth(self) -> float:
        """The pinch point's distance under the tube's top."""
        return CAP_DEPTH if self.capped else TOP_DEPTH

    @property
    def length(self) -> float:
        return TUBE.capped_height if self.capped else TUBE.height

    @property
    def width(self) -> float:
        return HOLD_CAP if self.capped else HOLD_TOP

    @property
    def squeeze(self) -> float:
        return GRIP_SQUEEZE if self.capped else OPEN_SQUEEZE


def hole_frame(data: mujoco.MjData, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Hole ``k``'s bottom (where a seated tube's tip rests) and rotation (z out of the hole). Bench layout."""
    s = data.site(scene.slot_site(k))
    rot = s.xmat.reshape(3, 3).copy()
    return s.xpos.copy() + rot[:, 2] * 0.0005, rot


def grip_at_hole(tip: np.ndarray, hole_rot: np.ndarray, turn_deg: float, grip: Grip):
    """The pinch point (position, rotation) holding a tube with its tip at ``tip`` on the hole's axis, the
    fingers closing ``turn_deg`` from along the rim."""
    z_t, e1 = hole_rot[:, 2], hole_rot[:, 0]
    e2 = np.cross(z_t, e1)
    t = math.radians(turn_deg)
    y = math.cos(t) * e1 + math.sin(t) * e2
    z = -z_t
    return tip + (grip.length - grip.depth) * z_t, np.column_stack(
        [np.cross(y, z), y, z]
    )


def wrist_force(data: mujoco.MjData) -> np.ndarray:
    """The wrist sensor's force, in the world frame."""
    return (
        data.site("arm/wrist_ft").xmat.reshape(3, 3)
        @ data.sensor("arm/wrist_force").data
    )


@dataclass
class Slide:
    """How a slide went: why it stopped, and what the force sensor felt."""

    stop: str = "reached the end of the slide"
    seconds: float = 0.0
    pauses: int = 0
    retries: int = 0
    peak_push_n: float = 0.0
    peak_side_n: float = 0.0
    final_shift_mm: float = 0.0
    log: list[dict] = field(default_factory=list, repr=False)

    @property
    def jammed(self) -> bool:
        return self.stop.startswith(("jammed", "inverse"))


class Loader:
    """Loads tubes into the rotor with the programmed arm."""

    def __init__(self, arm: Arm, mode: str = "plain", on_tick=None):
        self.arm, self.mode = arm, mode
        m = arm.model
        self.weigher = Weigher(arm)
        self.filler = Filler(arm)
        self.rotor = m.body("centrifuge/rotor").id
        self.hole_body = {
            k: m.body(f"centrifuge/hole{k}").id for k in range(ROTOR.slots)
        }
        self.racks = {r: m.body(r).id for r in scene.RACKS}
        self.gaps: list[float] = []
        # Called every CONTROL s of the slide with (loader, tip target along the axis): for the tests' logs.
        self.on_tick = on_tick

    # --- picking -----------------------------------------------------------------------------------------

    def pick(self, name: str, rack: str, cell: int, grip: Grip) -> None:
        """Pick the tube in ``rack``'s ``cell`` and lift it to the travel height."""
        arm = self.arm
        xy = np.array(cell_centres(getattr(LAYOUT, rack))[cell])
        body = tube_at(arm, xy)
        if body != arm.model.body(name).id:
            raise Blocked(f"{name} is not the tube standing in {rack} cell {cell}")
        if grip.capped:
            # Step 1's cap grip.
            turn, gap = self.weigher.rack_turn(xy, body)
            self.gaps.append(gap)
            low = np.array([*xy, grip_height(floor_height())])
            self.gaps.append(arm.move_line(np.array([*xy, TRAVEL]), turn))
            arm.open()
            self.gaps.append(arm.move_line(low))
            arm.close_on(body)
            rot = fingers_down(turn)
            tip = np.array([*xy, floor_height() + REST])
            up = np.array([0.0, 0.0, 1.0])
            arm.held = HeldAt.planned(
                body, low, rot, tip, np.column_stack([rot[:, 0], -rot[:, 1], up])
            )
            self.gaps.append(
                arm.move_line(np.array([*xy, TRAVEL]), allow=(self.racks[rack],))
            )
        else:
            # Step 3's grip on the open tube's top (1 mm squeeze).
            self.filler.pick_top(xy, floor_height(), body, (self.racks[rack],))

    # --- lining up ---------------------------------------------------------------------------------------

    def best_turn(
        self, hole: int, believed: np.ndarray, grip: Grip
    ) -> tuple[int, float]:
        """The finger turn with most room for the slide into ``hole``: (degrees, gap m)."""
        arm = self.arm
        _, hole_rot = hole_frame(arm.data, hole)
        axis = hole_rot[:, 2]
        back = BACK[hole]
        best: tuple[float, int] | None = None
        for t in HOLE_TURNS:
            start, rot = grip_at_hole(believed + back * axis, hole_rot, t, grip)
            try:
                q0 = arm.solve_any(start, rot)
                qs = plan_pose(
                    arm,
                    grip_at_hole(believed, hole_rot, t, grip)[0],
                    rot,
                    start=(start, rot, q0),
                )
            except Blocked:
                continue
            gap, _ = arm.clearance([q0, *qs], allow=(self.hole_body[hole],))
            # Equally clear turns: the one furthest from "along the rim" wins, where gravity turns the tube
            # least about the line between the pads.
            if (
                best is None
                or gap > best[0] + SAME_GAP
                or (gap > best[0] - SAME_GAP and abs(t) > abs(best[1]))
            ):
                best = (gap, t)
        if best is None or best[0] < SAFE:
            raise Blocked(f"no clear finger turn at hole {hole} (best {best})")
        return best[1], best[0]

    def poses_for(
        self, pos: np.ndarray, rot: np.ndarray, near: np.ndarray, tries: int = 60
    ) -> list[np.ndarray]:
        """Every distinct way the arm can put the pinch point at ``pos`` turned ``rot``: the solver started
        from ``near``, from home turned towards ``pos`` and from ``tries`` scattered angles; each answer taken
        the fewest whole turns from ``near``."""
        arm = self.arm
        rng = np.random.default_rng(0)
        starts = [near]
        for extra in (0.0, 0.6, -0.6):
            s = np.array(HOME)
            s[0] += math.atan2(pos[1], pos[0]) - math.atan2(0.492, -0.134)
            s[5] += extra
            starts.append(s)
        starts += [
            rng.uniform(arm.limits[:, 0] / 2, arm.limits[:, 1] / 2)
            for _ in range(tries)
        ]
        found: list[np.ndarray] = []
        for s in starts:
            try:
                q = arm.solve(pos, rot, s)
            except Blocked:
                continue
            q = q - 2 * np.pi * np.round((q - near) / (2 * np.pi))
            if np.any(q < arm.limits[:, 0]) or np.any(q > arm.limits[:, 1]):
                continue
            if not any(np.allclose(q, f, atol=0.05) for f in found):
                found.append(q)
        return found

    def tilt_deg(self, q: np.ndarray) -> float:
        """How far the held tube leans from upright with the arm's joints at ``q`` (fingers along the tube)."""
        m, work = self.arm.model, self._work
        work.qpos[self.arm.joints] = q
        mujoco.mj_kinematics(m, work)
        z = work.site_xmat[self.arm.site].reshape(3, 3)[:, 2]
        return math.degrees(math.acos(float(np.clip(-z[2], -1.0, 1.0))))

    def joint_sweep(
        self, q0: np.ndarray, q1: np.ndarray, max_tilt: float | None
    ) -> tuple[float, list[np.ndarray]]:
        """The joint angles every 0.02 rad from ``q0`` to ``q1``, and their smallest gap (to everything, and the
        hand to the arm); -1 if the tube would lean more than ``max_tilt`` degrees on the way."""
        n = max(2, math.ceil(float(np.abs(q1 - q0).max()) / 0.02))
        qs = [q0 + (q1 - q0) * t for t in np.linspace(0, 1, n + 1)[1:]]
        if max_tilt is not None and max(self.tilt_deg(q) for q in qs) > max_tilt:
            return -1.0, qs
        return min(self.arm.clearance(qs)[0], self_gap(self.arm, qs)), qs

    def go_to_start(
        self, start: np.ndarray, rot: np.ndarray, max_tilt: float | None = None
    ) -> float:
        """Turn the hand to the hole's slant and bring the tube to the start of the slide. Returns the room, m.

        The arm can hold that pose in several ways (elbow up or down, the shoulder a turn round); the one
        whose joint move keeps the most room is used. (For holes 4 and 5 the first answer reached over the
        arm's head and swept the tube through the centrifuge.)

        ``max_tilt`` (open tubes with water): the tube may never lean more than this on the way. A straight
        joint move does not turn the hand evenly: to hole 4 the only clear one swung the tube to 92 degrees
        and emptied it. Then the joints go to an upright pose above the start in that same arm shape, and the
        hand tilts evenly from there along a straight line to the start.
        """
        arm = self.arm
        self._work = mujoco.MjData(arm.model)
        self._work.qpos[:] = arm.data.qpos
        now = arm.data.qpos[arm.joints].copy()
        found = self.poses_for(start, rot, now)
        best: tuple[float, np.ndarray | None] = (-1.0, None)
        for q in found:
            gap, _ = self.joint_sweep(now, q, max_tilt)
            if gap > best[0]:
                best = (gap, q)
        if best[1] is not None and best[0] >= SAFE:
            move_joints(arm, best[1])
            return best[0]
        # No clear straight joint move (capped tubes too: hole 5, loaded last beside a full hole 0, had none):
        # upright first, then tilt on a straight line. The upright pose: the start's hand turned so the
        # tube stands straight, raised, solved from each arm shape found for the start.
        tube_up = -rot[:, 2]
        k = np.cross(tube_up, [0.0, 0.0, 1.0])
        upright = rot_about(k, math.asin(min(1.0, float(np.linalg.norm(k))))) @ rot
        tries = []
        for q_shape in sorted(found, key=lambda q: -self.joint_sweep(now, q, None)[0]):
            for lift in (0.05, 0.10, 0.15):
                above = start + np.array([0.0, 0.0, lift])
                try:
                    q_up = arm.solve(above, upright, q_shape)
                    tilt = plan_pose(arm, start, rot, start=(above, upright, q_up))
                except Blocked:
                    continue
                g1, _ = self.joint_sweep(now, q_up, max_tilt)
                g2 = min(arm.clearance(tilt)[0], self_gap(arm, tilt))
                tries.append((min(g1, g2), q_up, tilt))
                if min(g1, g2) >= SAFE:
                    break
            if tries and tries[-1][0] >= SAFE:
                break
        if not tries or tries[-1][0] < SAFE:
            raise Blocked(
                f"no clear way to the start of the slide ({len(found)} poses tried, and upright first)"
            )
        gap, q_up, tilt = tries[-1]
        move_joints(arm, q_up)
        arm.follow(tilt)
        return gap

    # --- the slide ---------------------------------------------------------------------------------------

    def slide(self, hole: int, believed: np.ndarray, turn: int, grip: Grip) -> Slide:
        """Slide the held tube from the start along the hole's axis until its tip is down (or it jams)."""
        arm, data = self.arm, self.arm.data
        _, hole_rot = hole_frame(data, hole)
        axis = hole_rot[:, 2]
        _, rot = grip_at_hole(believed, hole_rot, turn, grip)
        # The sensor's reading while still (gripper and tube): forces are measured from it.
        still = []
        for _ in range(20):
            arm.step(5)
            still.append(wrist_force(data).copy())
        base = np.mean(still, axis=0)
        out = Slide()
        q = data.qpos[arm.joints].copy()
        s = BACK[hole]  # how far back along the axis the tip's target is
        shift = np.zeros(3)
        paused = 0.0
        n_ctrl = round(CONTROL / arm.model.opt.timestep)
        i = 0
        tries_pauses, tries_paused = 0, 0.0
        for i in range(int(SLIDE_TIMEOUT / CONTROL)):
            f = wrist_force(data) - base
            # The hole pushing the tube back out reads as a force along -axis at the sensor.
            push = -float(f @ axis)
            side = f + push * axis
            out.peak_push_n = max(out.peak_push_n, push)
            out.peak_side_n = max(out.peak_side_n, float(np.linalg.norm(side)))
            if self.on_tick and i % 10 == 0:
                self.on_tick(self, s, push, float(np.linalg.norm(side)), shift)
            if s <= DROP:
                break
            hold = push > PAUSE_FORCE
            if hold and s < NEAR_BOTTOM:
                out.stop = f"bottom: pushed back {push:.1f} N"
                break
            if hold:
                out.pauses += paused == 0.0
                tries_pauses += paused == 0.0
                paused += CONTROL
                tries_paused += CONTROL
                if paused > JAM_SECONDS:
                    out.stop = f"jammed {s * 1000:.0f} mm from the bottom"
                    break
            else:
                paused = 0.0
            # Creeping in by pausing again and again: the tube is wedged at a slant (seed 0 of the whole run:
            # 76 pauses, stuck 12 mm short, and letting go of it blew the simulation up). Back out and try again.
            if s < ROTOR.depth and (
                tries_pauses > RETRY_PAUSES or tries_paused > RETRY_SECONDS
            ):
                if out.retries >= RETRIES:
                    out.stop = f"jammed {s * 1000:.0f} mm from the bottom after {RETRIES} retries"
                    break
                out.retries += 1
                s = min(BACK[hole], s + RETRY_BACK)
                # Held firmly (the pads resist twisting), a tube 3 degrees crooked in the fingers cannot line
                # itself up with the hole (it forgives about 1.4 degrees) and wedges again. Loosened, it can.
                data.ctrl[arm.gripper] = scene.grip_command(grip.width, LOOSE_SQUEEZE)
                tries_pauses, tries_paused, paused = 0, 0.0, 0.0
            if self.mode == "force":
                want = (
                    GIVE * side if np.linalg.norm(side) > SIDE_DEADBAND else np.zeros(3)
                )
                if np.linalg.norm(want) > MAX_SHIFT:
                    want = want / np.linalg.norm(want) * MAX_SHIFT
                shift = shift + (want - shift) * min(1.0, CONTROL / GIVE_TIME)
            if not hold:
                s = max(
                    DROP, s - (SLOW if s < ROTOR.depth + SLOW_FROM else FAST) * CONTROL
                )
            pinch, _ = grip_at_hole(believed + s * axis + shift, hole_rot, turn, grip)
            try:
                q = arm.solve(pinch, rot, q)
            except Blocked as e:
                out.stop = f"inverse kinematics: {e}"
                break
            data.ctrl[arm.actuators] = q
            arm.step(n_ctrl)
        out.seconds = round(i * CONTROL, 2)
        out.final_shift_mm = round(float(np.linalg.norm(shift)) * 1000, 2)
        # Hold the arm where it is: it trails its targets by about 0.2 s, and left chasing the last one it
        # would keep pressing the tube into the hole's bottom (that blew the simulation up once).
        data.ctrl[arm.actuators] = data.qpos[arm.joints]
        arm.wait(0.3)
        return out

    def let_go(self, hole: int) -> None:
        """Open the fingers and back out along the hole's axis, as far as is clear: 10 cm, else 8 or 6 cm (the
        open fingers come within 1.2 mm of the lid 10 cm back from holes 4 and 5; 6 cm clears the tube)."""
        arm = self.arm
        body = arm.held.body
        q = arm.data.qpos[arm.joints].copy()
        allow = (body, self.hole_body[hole], self.rotor)
        # Open only as wide as is clear here: opening swings the finger links out, towards a full neighbouring
        # hole's cap (seed 0 of the whole run: the simulation blew up as the fingers opened at hole 4 beside a
        # full hole 3). The cap is 35 mm wide.
        self.release_gaps = {}
        for gap in RELEASE:
            g, what = arm.clearance(
                [q], allow=allow, fingers=arm.fingers(gap), holding=False
            )
            self.release_gaps[gap] = (round(g * 1000, 1), what)
            if g >= SAFE:
                break
        else:
            raise Blocked(
                f"hole {hole}: no clear way to open the fingers ({self.release_gaps})"
            )
        arm.open(gap)
        _, hole_rot = hole_frame(arm.data, hole)
        for dist in BACK_OUT:
            out = arm.pinch() + dist * hole_rot[:, 2]
            try:
                self.gaps.append(move_pose(arm, out, hand_rot(arm), allow=allow))
                return
            except Blocked as e:
                last = e
        raise last

    # --- one tube ----------------------------------------------------------------------------------------

    def load(
        self,
        name: str,
        rack: str,
        cell: int,
        hole: int,
        capped: bool,
        miss: np.ndarray | None = None,
    ):
        """Pick ``name`` from its cell and load it into ``hole``. ``miss``: the arm's error in where it thinks
        the hole is (m, a vector), for tests. Returns (Slide, finger turn in degrees). Raises Blocked."""
        grip = Grip(capped)
        self.pick(name, rack, cell, grip)
        seat, hole_rot = hole_frame(self.arm.data, hole)
        believed = seat if miss is None else seat + miss
        turn, gap = self.best_turn(hole, believed, grip)
        self.gaps.append(gap)
        start, rot = grip_at_hole(
            believed + BACK[hole] * hole_rot[:, 2], hole_rot, turn, grip
        )
        self.gaps.append(self.go_to_start(start, rot, None if capped else MAX_TILT))
        self.arm.wait(0.2)
        result = self.slide(hole, believed, turn, grip)
        if result.jammed:
            raise Blocked(f"hole {hole}: {result.stop}")
        self.let_go(hole)
        return result, turn

    def return_home(self) -> None:
        """From a tilted hand at the rotor back to the start pose, fingers down.

        Up to the travel height with the hand as it is, then the joints slide home (the sweep checked first,
        as in Step 3). Straight up from the near holes the wrist passes 1 mm from the open lid; then it first
        backs a further 5 or 10 cm out along the hand's axis, or slides the joints home from where it is.
        """
        arm = self.arm
        rot = hand_rot(arm)
        here = arm.pinch()
        for back in (0.0, 0.05, 0.10):
            way = (
                here - back * rot[:, 2]
            )  # the fingers point down the tube: back out is against them
            try:
                if back:
                    plan = plan_pose(arm, way, rot)
                    if min(arm.clearance(plan)[0], self_gap(arm, plan)) < SAFE:
                        continue
                if way[2] < TRAVEL - 0.005:
                    q0 = arm.solve(way, rot, arm.data.qpos[arm.joints].copy())
                    up = plan_pose(
                        arm, np.array([*way[:2], TRAVEL]), rot, start=(way, rot, q0)
                    )
                    if min(arm.clearance(up)[0], self_gap(arm, up)) < SAFE:
                        continue
                if back:
                    self.gaps.append(move_pose(arm, way, rot))
                if way[2] < TRAVEL - 0.005:
                    self.gaps.append(move_pose(arm, np.array([*way[:2], TRAVEL]), rot))
                self.filler.go_home()
                return
            except Blocked:
                continue
        self.filler.go_home()  # the checked joint sweep from here, or Blocked

    def rise(self) -> None:
        """Straight up 25 cm from wherever the hand is (out of the rotor and the rotor camera's view)."""
        self.gaps.append(
            move_pose(
                self.arm,
                self.arm.pinch() + np.array([0.0, 0.0, 0.25]),
                hand_rot(self.arm),
            )
        )
