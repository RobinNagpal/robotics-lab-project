"""Uncapping, programmed: the baseline. One arm holds a capped tube, the other unscrews its cap.

    Arm A (arm2, beside the capped rack)       Arm B (arm1, beside the source rack)
    ------------------------------------       ------------------------------------
    pick a capped tube (Step 1b's side grasp)  wait at home
    carry it to the middle, hold it up
    "ready"  --------------------------------> look down at the cap: where is it exactly?
    keep holding                               fingers down over the cap, grip it
                                               quarter turns until the lift test says "off"
                                               lift the cap away, set it on the cap holder
    "cap put down" <-------------------------- home
    put the open tube in the destination rack
    (Step 3's put-back), home

Everything the arms decide comes from what a real arm could sense: Step 0's camera (where the tubes and
racks are, which tubes are capped), the wrist camera's depth picture (where the cap is), the wrist joint's
torque and the force on the gripper (whether the cap is off). The simulator's own knowledge (is the cap
on its thread, where is everything) is used only to grade the run afterwards.

How it works: docs/uncapping-programmed.md (the measurements: docs/uncapping-programmed-details.md).

    python -m uncap.episode --seed 150 --size 0.15
"""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from collections.abc import Iterator
from pathlib import Path

import mujoco
import numpy as np
from carry import episode as carry_episode
from carry import look, motion
from carry.episode import (
    CARRY,
    EMPTY_CLEARANCE,
    FAST,
    POUR,
    RECEIVER_ARM,
    SLOW,
    SOURCE_ARM,
    Choice,
    Coordinator,
    SideGrasp,
    load_net,
)
from grasp import side
from lab_scene import kinematics
from lab_scene import scene as lab
from lab_scene.racks import slot_positions
from lab_scene.settings import HOME, RECEIVERS
from putback.place import PutBack, cells
from putback.upright import lean

from capping import caps, scene
from capping.cap_trial import (
    FREE_FORCE,
    GRIP_SQUEEZE,
    LIFT_HOLD,
    LIFT_TEST,
    LIFT_TIME,
    OPEN_WIDTH,
    QUARTER,
    TURN_TIME,
)

from . import grip, holder, record
from .perceive import CapPerception, SeenTube

HOLD_ARM = RECEIVER_ARM  # Arm A holds the tube
CAP_ARM = SOURCE_ARM  # Arm B opens the cap
# Arm A holds the tube's top (the cap's top, as Step 0 measures it) here: Step 1b's pouring point, above
# the middle of the table, 35 cm up. It is where Step 1b already carries a receiver.
HOLD = POUR
# How far below the top Arm A grips (m), in the order tried: low, so the cap stands clear of its hand.
# (A 100 mm tube is gripped 3 cm down whatever is asked: in its rack, Step 1b keeps the fingers 2 cm
# above the rack's top.)
HOLD_HEIGHT_ORDER = (0.07, 0.05, 0.03)
ABOVE_MIDDLE = 0.025  # m: the grip stays at least this far above the tube's middle
CAP_ABOVE_RIM = 0.0046  # m: a capped tube's top (the cap's) above its rim, as Step 0 measures it


def hold_heights(size: float) -> tuple[float, ...]:
    """The grip heights (m below the top) Arm A tries on a tube of ``size``, lowest first, keeping the
    grip ABOVE_MIDDLE above the tube's middle. The others follow, so Step 1b's list stays complete;
    they come last."""
    keep = [b for b in HOLD_HEIGHT_ORDER if size + CAP_ABOVE_RIM - b >= size / 2 + ABOVE_MIDDLE]
    return tuple(keep + [b for b in HOLD_HEIGHT_ORDER[::-1] if b not in keep])


# Arm B's pads grip the cap this far above the middle of its side. With the cap trial's 4 mm
# (GRIP_ABOVE_MIDDLE) the pads' ends reach about 8 mm below the cap, and as the cap came off its thread
# they brushed the tube under it (4 of 8 runs; at 8 mm 2 of those 4, at 12 mm 1). At 15 mm a pad's tip,
# a little below the cap's bottom and leaning in, still grazed the top of the tube's collision walls (they
# reach 1 mm above the drawn rim) as the cap left the thread. At 18 mm it stays clear, and the grip still
# breaks the seal at 0.61 to 0.62 N m (scenes 150 to 153, 200 mm).
CAP_GRIP_ABOVE_MIDDLE = 0.018  # m
# The lift test's second look: once the pull is under FREE_FORCE, Arm B should be carrying the cap's
# weight only (4 g: 0.04 N). More means something comes up with the cap: the tube, slipping out of Arm A's
# hand (a 200 mm tube and its cap weigh 0.41 N). AutoBio's copy (docs/autobio-uncapping.md) showed a
# failing hold reads as "off" otherwise.
CAP_ONLY = 0.2  # N
READ_LAST = 0.1  # s: the lift test is read over the end of its hold
TUBE_HELD = 5.0  # degrees: graded "held" if the tube turned less than this in Arm A's hand
# The first quarter turn followed by a lift test. Each quarter turns the wrist 90 degrees, and the cap at
# most as far (its fingers can slip, not gain), so a cap with TURNS of thread (1.5: 540 degrees) cannot be
# off before quarter 6: tests before then can never find it off. 1 tests after every quarter, as the
# first teacher did. (In 193 demonstrations the cap came off in quarter 7 or 8, never earlier.)
TEST_FROM_SAFE = math.ceil(caps.TURNS * 4)
MAX_QUARTERS = 10  # 1.5 turns take about 7 (each 90 degrees of the wrist turns the cap about 69)
# Looking for the cap: the wrist camera's depth picture from this high above where the cap should be,
# searched this far round that point. The cap is the highest thing there; its top is every pixel within
# DISC of the highest.
LOOK_ABOVE = 0.25  # m
SEARCH = 0.03  # m
DISC = 0.003  # m
CAP_RADIUS = caps.OUTER_R + caps.RIB  # m, 14.4: what the look should find
ABOVE_CAP = 0.08  # m: Arm B comes down over the cap from this far above its top
CAP_AWAY = 0.06  # m: lifted straight up this far once off, before going anywhere
ABOVE_POST = 0.002  # m: the cap's bottom is let go this far above the holder's post
HAND_UP = 0.10  # m: the open hand goes up this far after letting go
# The ways Arm B's fingers may open across the cap (radians from the table's x), tried in this order.
YAWS = [math.radians(a) for a in (45, 30, 60, 15, 75, 0, 90, 105, 120, 135, 150, 165)]
# Arm A's firm hold while the cap is turned (hold_firm): ties to this body of its hand, at these points
# of the tube (in its own frame, m across, heights as shares of the tube's height), as stiff as
# the cap trial's ties to the world.
HAND = f"{HOLD_ARM}/gripper/base"
TIE_POINTS = ((0, 0, 0.2), (0, 0, 0.85), (0.10, 0, 0.5))
TIE_SOLREF = (0.004, 1.0)
TIME_LIMIT = 240.0  # s of simulated time
SETTLE = 1.0  # s at the end, before grading


class Watch:
    """Steps a capped scene (caps.step) and notes what each arm touches that it should not."""

    EVERY = 5

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, allowed: dict[str, set[str]]):
        self.model, self.data = model, data
        names = [model.body(model.geom_bodyid[g]).name for g in range(model.ngeom)]
        self.names = names
        self.arm_of = {a: np.array([n.startswith(f"{a}/") for n in names]) for a in lab.ARM_NAMES}
        self.allowed = {a: np.array([n in allowed.get(a, set()) for n in names]) for a in lab.ARM_NAMES}
        self.touched = {a: set() for a in lab.ARM_NAMES}
        self.parts: set[tuple] = set()  # (arm, the arm's part, what it touched, when)
        self.steps = 0

    def step(self) -> None:
        caps.step(self.model, self.data)
        self.steps += 1
        if self.steps % self.EVERY:
            return
        for c in self.data.contact[: self.data.ncon]:
            for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                for arm, mine in self.arm_of.items():
                    if mine[a] and not mine[b] and not self.allowed[arm][b]:
                        self.touched[arm].add(self.names[b])
                        self.parts.add((arm, self.names[a], self.names[b], round(float(self.data.time), 2)))


class CapPlanner(motion.Planner):
    """Step 1b's planner, with a tube's cap counted as part of the tube.

    Step 1b's checks let Arm A's fingers come close to the tube it is picking (its last move slides in
    past the tube's top), but the cap is a body of its own, so they would stop at it: 3.6 mm from the cap
    is inside the 5 mm margin. The cap moves with its tube, so wherever a check lets the tube be touched,
    its cap may be too."""

    def check(self, data, arm, path, held, ignore=frozenset(), margin=0.0):
        super().check(data, arm, path, held, _with_caps(ignore), margin)

    def check_closing(self, data, arm, q, direction, target):
        saved = self.names
        cap = {caps.follower_name(target), caps.cap_name(target)}
        self.names = [target if n in cap else n for n in saved]
        try:
            super().check_closing(data, arm, q, direction, target)
        finally:
            self.names = saved


def _with_caps(names) -> set[str]:
    out = set(names)
    for n in names:
        out |= {caps.follower_name(n), caps.cap_name(n)}
    return out


class Uncapper(Coordinator):
    """Step 1b's coordinator (choosing, picking, carrying, checking moves), with Arm B's cap script."""

    lifts_cap = False  # does ``unscrew`` end with the cap lifted away (a learned policy's does)

    def __init__(self, model, data, perception, net, ties: bool = False):
        super().__init__(model, data, perception, net)
        self.ties = ties  # the old simplification (hold_firm) instead of Arm A's real grip
        self.test_from = 1  # the first quarter with a lift test (TEST_FROM_SAFE: the teacher that tests less)
        self.pause = 0.0  # s held still after an untested quarter turn, before letting go
        self.rec: record.Recorder | None = None  # Arm B's unscrewing, recorded (record.py), if asked
        self.recording = False
        old = self.planner
        self.planner = CapPlanner(model)
        self.planner.way_in_closing, self.planner.shove = old.way_in_closing, old.shove
        self.report: dict = {"quarters": []}

    # ---------------------------------------------------------------- choosing

    def choose_capped(self, sizes: list[float]) -> tuple[SeenTube, SideGrasp] | None:
        """The first capped tube, by size in ``sizes``, that Arm A can pick and carry to HOLD.

        Step 1b grips as high as it can (3 cm below the top first). Held so, Arm A's fingers and wrist
        are just under the cap, and Arm B's fingers cannot get round it (scene 150: all 12 ways blocked).
        So here the lowest grip is tried first (7 cm below the top, then 5, then 3), but only grips
        ABOVE_MIDDLE or more above the tube's middle: held lower, the tube no longer hangs from the
        fingers but balances on them, and swings over (scene 150, 150 mm tube gripped 7 cm down: 47
        degrees)."""
        camera = look.Camera(self.model, self.data)
        saved = carry_episode.HEIGHT_ORDER
        try:
            for size in sizes:
                # read by Step 1b's grasp_of
                carry_episode.HEIGHT_ORDER = hold_heights(size)
                for t in self.table.in_rack("capped_rack"):
                    if t.size == size and t.capped:
                        g = self.grasp_of(camera, t, HOLD_ARM, "receiver")
                        if g is not None:
                            return t, g
        finally:
            carry_episode.HEIGHT_ORDER = saved
            camera.close()
        return None

    # ---------------------------------------------------------------- Arm B

    def sense(self) -> dict:
        """What Arm B feels: its wrist joint's torque, and the up force its fingers put on the cap (what
        its wrist force sensor reads, less the gripper's own weight). Read from the contacts between
        its fingers and the cap, as in the cap trial."""
        m, d = self.model, self.data
        force = np.zeros(3)
        for k in range(d.ncon):
            c = d.contact[k]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if ({b1, b2} & self.cap_ids) and ({b1, b2} & self.arm_b):
                f6 = np.zeros(6)
                mujoco.mj_contactForce(m, d, k, f6)
                world = c.frame.reshape(3, 3).T @ f6[:3]
                force += world if b2 in self.cap_ids else -world
        return {"wrist_torque": float(d.actuator_force[self.wrist3_act]), "pull": float(force[2])}

    def find_cap(self) -> tuple[np.ndarray, float, float]:
        """Where the cap is: the middle of its top (x, y), its top's height, and its radius, from the
        wrist camera's depth picture looking straight down (as Step 1b's close-ups are taken)."""
        camera = look.Camera(self.model, self.data)
        try:
            heights, calib = camera.heights(HOLD[:2], HOLD[2] + LOOK_ABOVE)
        finally:
            camera.close()
        v, u = np.indices(heights.shape)
        depth = calib.pos[2] - heights
        points = calib.deproject(u.ravel().astype(float), v.ravel().astype(float), depth.ravel())
        near = np.linalg.norm(points[:, :2] - HOLD[:2], axis=1) < SEARCH
        h = heights.ravel()
        top = float(h[near].max())
        disc = near & (h > top - DISC)
        centre = points[disc, :2].mean(axis=0)
        pixel = calib.pos[2] - top  # depth of the top, for the size of one pixel there
        area = disc.sum() * (pixel / calib.focal) ** 2
        return centre, top, math.sqrt(area / math.pi)

    def ramp(self, goal: np.ndarray, seconds: float, log: list | None = None) -> Iterator:
        """Slide Arm B's joint targets straight to ``goal`` over ``seconds`` (as the cap trial does)."""
        start = self.data.ctrl[self.arm_b_act].copy()
        steps = max(1, round(seconds / self.model.opt.timestep))
        for i in range(1, steps + 1):
            self.data.ctrl[self.arm_b_act] = start + (goal - start) * i / steps
            yield
            if log is not None:
                log.append(self.sense())

    def solve_down(self, data: mujoco.MjData, target: np.ndarray, wrist3: float) -> np.ndarray:
        """Arm B's joints with the pinch point at ``target``, fingers down, the last joint at ``wrist3``.

        The last joint is set first and the rest solved from there: the turn about the fingers' axis is
        free in the solve, so it stays, and the pinch point stays where it is asked to be (set after the
        solve it swung 3 mm round, as the point is not exactly on that joint's axis; cap_trial.py)."""
        work = mujoco.MjData(self.model)
        work.qpos[:] = data.qpos
        work.qpos[self.arm_b_q[5]] = wrist3
        mujoco.mj_kinematics(self.model, work)
        q = kinematics.solve(self.model, work, target, CAP_ARM)
        work.qpos[self.arm_b_q] = q
        mujoco.mj_kinematics(self.model, work)
        if np.linalg.norm(work.site(lab.pinch_site(CAP_ARM)).xpos - target) > motion.REACHED:
            raise motion.Blocked(f"{CAP_ARM} cannot reach {np.round(target, 3)}")
        return q

    def turn_path(self, data: mujoco.MjData, target: np.ndarray, wrist3: float, n: int = 12) -> list:
        q0 = data.qpos[self.arm_b_q].copy()
        q1 = self.solve_down(data, target, wrist3)
        return [q0 + (q1 - q0) * i / n for i in range(1, n + 1)]

    def plan_cap(self, centre: np.ndarray, grip_z: float) -> tuple[float, list]:
        """The way the fingers open across the cap (YAWS), and the checked way down to it.

        For each, the whole thing is checked before Arm B moves: down over the cap, and at the cap
        with the wrist turned a quarter and back, fingers open (as when gripping again), touching
        nothing but the cap."""
        here = motion.pinch(self.data, CAP_ARM)
        yaw0 = motion.open_angle(self.data, CAP_ARM)
        above = np.array([centre[0], centre[1], grip_z + ABOVE_CAP])
        grip = np.array([centre[0], centre[1], grip_z])
        high = np.array([here[0], here[1], max(here[2], self.transfer)])
        reasons = []
        for yaw in YAWS:
            try:
                work = mujoco.MjData(self.model)
                work.qpos[:] = self.data.qpos
                plan = []
                for a, b, y0, y1, speed in (
                    (here, high, yaw0, yaw0, FAST),
                    (high, above, yaw0, yaw, FAST),
                    (above, grip, yaw, yaw, SLOW),
                ):
                    path = self.planner.solve_line(work, CAP_ARM, a, b, y0, y1)
                    self.planner.check(work, CAP_ARM, path, None)
                    plan.append((path, speed))
                    work.qpos[self.arm_b_q] = path[-1]
                    mujoco.mj_kinematics(self.model, work)
                w0 = float(work.qpos[self.arm_b_q[5]])
                turned = self.turn_path(work, grip + [0, 0, caps.PITCH], w0 - QUARTER)
                self.planner.check(work, CAP_ARM, turned, None, self.cap_names)
                return yaw, plan
            except motion.Blocked as e:
                reasons.append(f"{math.degrees(yaw):.0f} deg: {e}")
        raise motion.Blocked(f"{CAP_ARM} has no clear way to the cap: {reasons}")

    def grip(self, width: float, squeeze: float, seconds: float = 0.4) -> Iterator:
        start, goal = self.data.ctrl[self.grip_b], lab.grip_command(width, squeeze=squeeze)
        steps = round(seconds / self.model.opt.timestep)
        for i in range(1, steps + 1):
            self.data.ctrl[self.grip_b] = start + (goal - start) * i / steps
            yield
        yield from motion.wait(self.model, 0.2)

    def cap_script(self, tube: str) -> Iterator:
        arm, m, d = CAP_ARM, self.model, self.data
        self.cap_names = {caps.follower_name(tube), caps.cap_name(tube)}
        self.cap_ids = {m.body(n).id for n in self.cap_names}
        self.say(arm, "waiting")  # Step 1b's receiver script waits for this before it picks
        while "ready" not in self.flags:
            yield
        # Arm A grips tighter and its fingers lock (grip.py): Step 1b's grip alone let the tube turn in
        # its fingers at 0.05 N m; the cap needs 0.6.
        if self.ties:
            hold_firm(m, d, tube, True)
            yield from motion.wait(m, 0.3)
        else:
            grip.squeeze(m, d, HOLD_ARM)
            yield from motion.wait(m, grip.SETTLE)
            grip.hold_tight(m, d, HOLD_ARM, True)
        self.say(HOLD_ARM, "holding tight")
        # 1. Where exactly is the cap? The tube leans a little in Arm A's fingers.
        centre, top, radius = self.find_cap()
        # For grading only: where the cap really is (its body sits at the inside of its top, caps.py).
        true = d.xpos[caps.cap_body(m, d, tube)]
        self.report["look"] = {
            "radius_mm": round(radius * 1000, 1),
            "centre_error_mm": round(float(np.linalg.norm(centre - true[:2])) * 1000, 1),
            "top_error_mm": round((top - (true[2] + caps.TOP)) * 1000, 1),
        }
        if abs(radius - CAP_RADIUS) > 0.004:
            raise motion.Blocked(f"the look found no cap ({radius * 1000:.1f} mm across, not 14.4)")
        self.say(arm, "found the cap")
        if self.recording:
            self.rec = record.Recorder(m, d, tube)
            self.rec.on = True
            self.report["seen_cap"] = {"centre": centre.round(5).tolist(), "top": round(top, 5)}
        # The pads 4 mm above the middle of the cap's side (cap_trial.py): its top less the closed top's
        # thickness is the inside of the top, the reference the trial grips from.
        grip_z = top - caps.TOP - ((caps.SKIRT - caps.TOP) / 2 - CAP_GRIP_ABOVE_MIDDLE)
        yield from self.grip(OPEN_WIDTH, 0.0)
        yaw, plan = self.plan_cap(centre, grip_z)
        self.report["yaw_deg"] = round(math.degrees(yaw))
        for path, speed in plan[:-1]:
            yield from self.checked(arm, path)
            yield from motion.follow(m, d, arm, path, speed)
        # Just above the cap: turn the wrist a quarter and back, open, touching nothing. The torque that
        # takes is the wrist's own (its motor moving its own inertia), taken off the readings later, as a
        # real arm's are zeroed ("tared"), as in the cap trial.
        w0 = float(d.qpos[self.arm_b_q[5]])
        tare_at = np.array([centre[0], centre[1], grip_z + 0.04])
        yield from self.ramp(self.solve_down(d, tare_at, w0), 1.0)
        log: list = []
        yield from self.ramp(self.solve_down(d, tare_at, w0 - QUARTER), TURN_TIME, log)
        tare = np.array([s["wrist_torque"] for s in log])
        yield from self.ramp(self.solve_down(d, tare_at, w0), TURN_TIME)
        # Down round the cap, fingers open, the last joint as it was before the tare.
        down = self.turn_path(d, np.array([centre[0], centre[1], grip_z]), w0)
        yield from self.checked(arm, down)
        yield from motion.follow(m, d, arm, down, SLOW)
        squeeze = (2 * caps.OUTER_R, GRIP_SQUEEZE)
        self._phase("grip")
        yield from self.grip(*squeeze)
        self.say(arm, "gripped the cap")
        self.yaw_gripped = _yaw(d, self.tube)

        yield from self.unscrew(centre, grip_z, w0, tare, squeeze)
        yield from self.put_cap_away(grip_z, top, lifted=self.lifts_cap)

    def unscrew(self, centre: np.ndarray, grip_z: float, w0: float, tare: np.ndarray, squeeze) -> Iterator:
        """Arm B, the cap gripped: quarter turns until the lift test says "off" (the part a learned
        policy replaces: learn/rollout.py)."""
        arm, m, d = CAP_ARM, self.model, self.data
        # 2. Quarter turns, each followed by the lift test (cap_trial.py). Anticlockwise seen from above
        # is a negative turn of the last joint (the fingers point down). Each turn aims a quarter of the
        # thread's pitch higher.
        z = grip_z
        free = False
        for q in range(1, MAX_QUARTERS + 1):
            z += caps.PITCH / 4
            at = np.array([centre[0], centre[1], z])
            log = []
            self._phase("turn", q)
            yield from self.ramp(self.solve_down(d, at, w0 - QUARTER), TURN_TIME, log)
            net = np.array([s["wrist_torque"] for s in log])[: len(tare)] - tare[: len(log)]
            if q < self.test_from:
                # Too early for the cap to be off (see TEST_FROM_SAFE): no lift test, only (if asked)
                # a pause holding still, for the experiment in docs (the cap turns on while held).
                if self.pause:
                    self._phase("test")
                    yield from motion.wait(m, self.pause)
                self.report["quarters"].append(
                    {
                        "quarter": q,
                        "peak_torque_Nm": round(float(np.max(np.abs(net))), 3),
                        "lift_force_N": None,
                        "arm_says_off": False,
                        "turned": round(caps.turned(m, d, self.tube), 3),
                        "on_thread": caps.on_thread(m, d, self.tube),
                    }
                )
            else:
                hold: list = []
                self._phase("test")
                yield from self.ramp(self.solve_down(d, at + [0, 0, LIFT_TEST], w0 - QUARTER), LIFT_TIME)
                for _ in range(round(LIFT_HOLD / m.opt.timestep)):
                    yield
                    hold.append(self.sense())
                # The pull at the end of the hold: the cap often leaves the thread during the hold, and the
                # pull then drops from about 12 N to the cap's weight part way through.
                lift_force = float(np.mean([s["pull"] for s in hold[-round(READ_LAST / m.opt.timestep) :]]))
                if CAP_ONLY < lift_force < FREE_FORCE:
                    # Neither on nor off: hold a moment longer and read again.
                    hold = []
                    for _ in range(round(LIFT_HOLD / m.opt.timestep)):
                        yield
                        hold.append(self.sense())
                    lift_force = float(
                        np.mean([s["pull"] for s in hold[-round(READ_LAST / m.opt.timestep) :]])
                    )
                free = lift_force < FREE_FORCE  # decided from what the arm feels only
                if free and lift_force > CAP_ONLY:
                    self.report["quarters"].append(
                        {"quarter": q, "lift_force_N": round(lift_force, 2), "arm_says_off": False}
                    )
                    raise motion.Blocked(
                        f"the lift test pulls {lift_force:.2f} N: under {FREE_FORCE} N but more than "
                        "the cap's weight, so the tube is coming up with it (Arm A's hold has failed)"
                    )
                self.report["quarters"].append(
                    {
                        "quarter": q,
                        "peak_torque_Nm": round(float(np.max(np.abs(net))), 3),
                        "lift_force_N": round(lift_force, 2),
                        "arm_says_off": free,
                        # for grading only:
                        "turned": round(caps.turned(m, d, self.tube), 3),
                        "on_thread": caps.on_thread(m, d, self.tube),
                    }
                )
                if free:
                    break
                # Down again, let go, turn back, grip again.
                self._phase("down")
                yield from self.ramp(self.solve_down(d, at, w0 - QUARTER), LIFT_TIME)
            self._phase("open")
            yield from self.grip(OPEN_WIDTH, 0.0)
            self._phase("turn back")
            yield from self.ramp(self.solve_down(d, at, w0), 0.6)
            self._phase("regrip")
            yield from self.grip(*squeeze)
        self.report["tube_turned_deg"] = round(
            math.degrees(motion._round(self.yaw_gripped, _yaw(d, self.tube))), 1
        )
        if not free:
            raise motion.Blocked(f"the cap did not come off in {MAX_QUARTERS} quarter turns")
        self.say(arm, "cap off")

    def put_cap_away(self, grip_z: float, top: float, lifted: bool = False) -> Iterator:
        """Arm B, the cap off: lift it away (unless ``lifted``), set it on the holder's post, go home."""
        arm, m, d = CAP_ARM, self.model, self.data
        # 3. Away with it, onto the holder's post. Where the cap sits in the fingers: its bottom is its
        # height below its top, which was gripped at a known height below the top.
        cap = caps.cap_name(self.tube)
        held = motion.grab(d, arm, cap)
        yaw = motion.open_angle(d, arm)
        pinch = motion.pinch(d, arm)
        bottom_below_pinch = (grip_z - top) + caps.HEIGHT  # pinch point above the cap's bottom
        # Carried no lower than it is once lifted away: the tube is still held up in the middle (and the
        # checks cannot see the cap meet its own tube: the two never touch in the simulation, caps.py).
        carry_z = max(
            pinch[2] + (0.0 if lifted else CAP_AWAY),
            max(self.table.tallest_rim, holder.TOP) + EMPTY_CLEARANCE,
        )
        post = holder.top()
        down_at = post + [0, 0, ABOVE_POST + bottom_below_pinch]
        self._phase("lift away")
        if not lifted:  # (a learned policy has lifted it already)
            yield from self.line(arm, pinch + [0, 0, CAP_AWAY], yaw, SLOW, held, ignore={self.tube})
        if self.rec is not None:
            self.rec.on = False  # the demonstration ends with the cap lifted away
        yield from self.line(arm, np.array([pinch[0], pinch[1], carry_z]), yaw, CARRY, held)
        yield from self.line(arm, np.array([down_at[0], down_at[1], carry_z]), yaw, CARRY, held)
        yield from self.line(arm, down_at, yaw, SLOW, held, ignore={holder.NAME})
        yield from self.grip(OPEN_WIDTH, 0.0)
        yield from motion.wait(m, 0.3)
        self.say(arm, "cap put down")
        # Back to Step 1b's grip, as Step 3's put-back expects: unlocked, so the fingers can open.
        if self.ties:
            hold_firm(m, d, self.tube, False)
        else:
            grip.hold_tight(m, d, HOLD_ARM, False)
            grip.squeeze(m, d, HOLD_ARM, lab.GRIP_SQUEEZE)
        yield from self.line(arm, down_at + [0, 0, HAND_UP], yaw, SLOW, ignore={cap})
        q0 = d.qpos[self.arm_b_q].copy()
        home = [q0 + (np.array(HOME) - q0) * i / 20 for i in range(1, 21)]
        try:
            yield from self.checked(arm, home)
            yield from motion.follow(m, d, arm, home, FAST)
            self.say(arm, "home")
        except motion.Blocked:
            self.say(arm, "stays: no clear way home")
        self.cap_done = True
        while True:
            yield

    def _phase(self, name: str, quarter: int | None = None) -> None:
        if self.rec is not None:
            self.rec.phase = name
            if quarter is not None:
                self.rec.quarter = quarter

    # ---------------------------------------------------------------- the run

    def run_uncap(self, sizes: list[float], watch_allowed) -> dict:
        m, d = self.model, self.data
        self.table = self.perception.table(m, d)
        self.transfer = max(side.TRANSFER, self.table.tallest_rim + EMPTY_CLEARANCE)
        self.arm_b = {b for b in range(m.nbody) if m.body(b).name.startswith(f"{CAP_ARM}/")}
        self.arm_b_act = lab.arm_actuator_ids(m, CAP_ARM)
        self.arm_b_q = lab.arm_joint_ids(m, CAP_ARM)
        self.wrist3_act = self.arm_b_act[5]
        self.grip_b = lab.gripper_actuator_id(m, CAP_ARM)
        self.cap_done = False
        chosen = self.choose_capped(sizes)
        if chosen is None:
            return {"chosen": False, "failure": "no capped tube can be picked", "log": self.log}
        tube_seen, grasp = chosen
        self.tube = tube = self.truth_name(tube_seen.xy)
        watch = Watch(m, d, watch_allowed(tube))
        choice = Choice(tube_seen, None, grasp, None)
        scripts = [self.receiver_script(choice, tube), self.cap_script(tube)]
        failure, put_plan = None, None
        try:
            while d.time < TIME_LIMIT and not self.cap_done:
                for s in scripts:
                    next(s)
                watch.step()
                if self.rec is not None:
                    self.rec.step()
                if not np.isfinite(d.qpos).all():
                    raise motion.Blocked("simulation blew up")
            if not self.cap_done:
                raise motion.Blocked("out of time")
            # 4. Arm A puts the open tube in the destination rack (Step 3's put-back), Arm B at home.
            put = PutBack(m, d, self.table.tallest_rim, watch.step)
            cells_free = destination_options(self.table, d.body(f"{HOLD_ARM}/base").xpos)
            put_plan = put.plan(HOLD_ARM, tube, cells_free)
            put.run(HOLD_ARM, tube, put_plan)
            self.log += [(t, a, f"put back: {msg}") for t, a, msg in put.log]
        except motion.Blocked as e:
            failure = str(e)
            # Which move it was (the innermost frame of this package or Step 1b's), for the report.
            tb = traceback.extract_tb(e.__traceback__)
            self.report["failed_in"] = [
                f"{f.name}:{f.lineno}" for f in tb if "uncap" in f.filename or "carry" in f.filename
            ][-3:]
        for _ in range(int(SETTLE / m.opt.timestep)):
            watch.step()
        return self.grade(tube, tube_seen, failure, watch, put_plan)

    def grade(self, tube, seen, failure, watch, put_plan) -> dict:
        """The check, from the simulator (the arms never see these)."""
        m, d = self.model, self.data
        cap = caps.cap_name(tube)
        cap_pos = d.body(cap).xpos
        on_post = bool(
            not caps.on_thread(m, d, tube)
            and np.linalg.norm(cap_pos[:2] - holder.top()[:2]) < 0.008
            and abs(cap_pos[2] - holder.top()[2] - caps.HEIGHT) < 0.004  # the cap body sits at its top
            and d.body(cap).xmat[8] > 0.98  # right way up, as it was on the tube
        )
        in_rack = where(d, tube, "destination_rack")
        quarters = self.report["quarters"]
        said_off = [q for q in quarters if q.get("arm_says_off")]
        result = {
            "tube": tube,
            "size": seen.size,
            "failure": failure,
            "cap_off": not caps.on_thread(m, d, tube),
            "cap_on_holder": on_post,
            "tube_in_rack": in_rack,
            "touched": sorted({(arm, what) for arm, _, what, _t in watch.parts}),
            "quarters": len(quarters),
            "seal_torque_Nm": quarters[0]["peak_torque_Nm"] if quarters else None,
            "off_said_too_early": bool(said_off and said_off[0].get("on_thread")),
            "lift_forces_N": [q["lift_force_N"] for q in quarters],
            # how far the tube turned in Arm A's fingers while the cap was being turned
            "tube_turned_deg": self.report.get("tube_turned_deg"),
            "touches": _touch_spans(watch.parts),
            "cap_at": np.round(cap_pos, 4).tolist(),
            "cap_upright": round(float(d.body(cap).xmat[8]), 3),
            "look": self.report.get("look"),
            "failed_in": self.report.get("failed_in"),
            "fingers_yaw_deg": self.report.get("yaw_deg"),
            "gripped_below_top_cm": self.grips.get("receiver"),
            "put_back": None if put_plan is None else {"cell": put_plan.cell, "turn": put_plan.turn},
            "log": self.log,
            "sim_seconds": round(d.time, 1),
        }
        turned = result["tube_turned_deg"]
        result["tube_held"] = turned is not None and abs(turned) < TUBE_HELD
        result["done"] = bool(
            failure is None
            and result["cap_on_holder"]
            and in_rack["in_cell"]
            and not result["touched"]
            and result["tube_held"]
        )
        return result


def _touch_spans(parts: set[tuple]) -> list[tuple]:
    """Each (arm, its part, what it touched) with when it first and last touched (s)."""
    spans: dict[tuple, list[float]] = {}
    for arm, part, what, t in parts:
        spans.setdefault((arm, part, what), []).append(t)
    return sorted((*k, min(v), max(v)) for k, v in spans.items())


def destination_options(table, base: np.ndarray) -> list[tuple[int, np.ndarray, str]]:
    """Free cells of the destination rack, nearest Arm A's base first (as Step 3 orders them)."""
    taken = [t.xy for t in table.tubes]
    dst = cells(table.racks["destination_rack"])
    free = [i for i, c in enumerate(dst) if all(np.linalg.norm(c[:2] - xy) > 0.015 for xy in taken)]
    free.sort(key=lambda i: np.linalg.norm(dst[i][:2] - base[:2]))
    return [(i, dst[i], "destination_rack") for i in free]


def where(data: mujoco.MjData, tube: str, rack: str) -> dict:
    """Is the tube standing in a cell of ``rack`` (as Step 3's check)."""
    floor = slot_positions((0.0, 0.0))[0][2]
    slots = np.array(slot_positions(tuple(data.body(rack).xpos[:2])))
    bottom = data.body(tube).xpos
    dist = np.linalg.norm(slots[:, :2] - bottom[:2], axis=1)
    i = int(np.argmin(dist))
    tilt = lean(data, tube)
    return {
        "cell": i,
        "off_mm": round(float(dist[i]) * 1000, 1),
        "lean_deg": round(tilt, 1),
        "in_cell": bool(dist[i] < 0.005 and tilt < 8.0 and abs(bottom[2] - floor) < 0.004),
    }


def _yaw(data: mujoco.MjData, tube: str) -> float:
    """How the tube is turned about the vertical (from its body's x axis)."""
    x = data.body(tube).xmat.reshape(3, 3)[:, 0]
    return math.atan2(x[1], x[0])


def prepare(model: mujoco.MjModel, data: mujoco.MjData, arm_start: dict | None = None) -> None:
    """Arms home, grippers opened 35 mm (Step 1b's side grasps), caps on, and a moment to settle.
    ``arm_start`` maps an arm to joint offsets (rad) from home, for the stress test."""
    scene.reset(model, data)
    for arm in lab.ARM_NAMES:
        data.ctrl[lab.gripper_actuator_id(model, arm)] = lab.grip_command(side.PRESHAPE_GAP, squeeze=0.0)
        if arm_start and arm in arm_start:
            q = np.array(HOME) + np.array(arm_start[arm])
            data.qpos[lab.arm_joint_ids(model, arm)] = q
            data.ctrl[lab.arm_actuator_ids(model, arm)] = q
    mujoco.mj_forward(model, data)
    caps.step(model, data, round(0.5 / model.opt.timestep))


# The stress test (``vary``): what a real lab varies, drawn at random for each run.
PLACED_OFF = 0.002  # m: a tube stands up to this far from the middle of its cell (30 mm cell, 25 mm tube)
TIGHTNESS = (0.45, 1.25)  # N m to start undoing a 24 mm cap (caps.py: 40 to 60 % of 1.1 to 2.0 N m)
# rad: with ``vary_arms``, each arm joint starts up to this far from home. Off by default: a real arm
# goes home to within 0.03 mm, and with this on, 6 of 30 runs failed after the cap was off (Step 3's
# put-back found no way to put the tube down from the pose Arm A then held it in, 4; the tube landed
# 16 mm off its cell, 1; the cap was dropped on the way to the holder, 1). All 6 pass from home.
ARM_START = 0.03
DEFAULT_TIGHTNESS = (caps.BREAKAWAY_TORQUE, caps.RUNNING_TORQUE)


def variation(vary: int | None, vary_arms: bool = False) -> dict | None:
    """The random differences for stress-test run ``vary`` (None: none): where each capped tube stands
    in its cell, how tight the cap is, and (``vary_arms``) the arms' starting pose."""
    if vary is None:
        return None
    rng = np.random.default_rng(10_000 + vary)
    placed = {}
    for tube in scene.CAPPED:
        r, a = PLACED_OFF * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
        placed[tube] = (float(r * np.cos(a)), float(r * np.sin(a)), float(rng.uniform(0, 2 * np.pi)))
    return {
        "placed": placed,
        "breakaway_Nm": float(rng.uniform(*TIGHTNESS)),
        "arm_start": {
            a: (rng.uniform(-ARM_START, ARM_START, 6) * (1.0 if vary_arms else 0.0)).round(4).tolist()
            for a in lab.ARM_NAMES
        },
    }


def build(seed: int, spread: bool = True, placed=None) -> tuple[mujoco.MjModel, object]:
    """The capped scene with the cap holder, the capped tubes starting asleep, and (``spread``) the
    capped tubes further apart than v1 stands them (scene.SPREAD_CELLS: two caps next to each other
    blocked Arm A's picks).

    capping.scene keeps capped tubes, their caps and their rack always awake: in the cap trial the tube
    is held by ties to the world, and asleep its thread froze while being turned. But awake, a tube
    standing in its cell slowly leans over against the cell's wall with nothing touching it (scene 150:
    5.6 degrees after 15 s), and Arm A's planned way to it stops fitting. Here they start asleep, as v1's
    tubes do, and wake when Arm A touches them; held in Arm A's fingers (always awake) a tube stays awake
    while its cap is turned."""

    def edit(spec: mujoco.MjSpec) -> None:
        holder.add(spec)
        for tube in scene.CAPPED:
            for name in (tube, caps.cap_name(tube)):
                spec.body(name).sleep = mujoco.mjtSleepPolicy.mjSLEEP_INIT
        spec.body("empty_rack").sleep = mujoco.mjtSleepPolicy.mjSLEEP_INIT
        add_ties(spec)
        record.add_wrist_sensors(spec)

    return scene.build(seed, edit=edit, spread=spread, placed=placed)


def add_ties(spec: mujoco.MjSpec) -> None:
    """Three ties between each capped tube and Arm A's hand, off until hold_firm switches them on."""
    for tube in scene.CAPPED:
        for i in range(len(TIE_POINTS)):
            spec.add_equality(
                name=f"{tube}_held{i}",
                type=mujoco.mjtEq.mjEQ_CONNECT,
                objtype=mujoco.mjtObj.mjOBJ_BODY,
                name1=tube,
                name2=HAND,
                active=False,
                solref=TIE_SOLREF,
            )


def hold_firm(model: mujoco.MjModel, data: mujoco.MjData, tube: str, on: bool) -> None:
    """Make Arm A's hold on ``tube`` firm (``on``), or leave it to the fingers again.

    **A simplification.** Holding the tube against the cap's 0.6 N m breakaway needs a firm grip, about
    25 N or more with rubber pads on glass; a real Robotiq 2F-85 grips with 20 to 235 N, set for the
    task. The simulated gripper does not hold a tube that firmly from the side. Measured on scene 150
    (twisting the held tube steadily for 2 s): with Step 1b's grip the tube spun at 0.2 N m. Closing
    harder made it tip over in the fingers (25 degrees in 5 s); a smooth outside for the pads made no
    difference; gripping with the tube between the pads' middles and a higher motor limit held 0.6 N m
    on one tube and spun at 0.2 N m on another. So while the cap is turned, the tube is tied to Arm A's
    hand where the fingers hold it, by three points (two on its axis, one 10 cm out to the side), a
    little springy as a gripper is: the cap trial's way of holding the tube, with the hand in place of
    the world. The fingers stay closed on it as well."""
    hand = data.body(HAND)
    body = data.body(tube)
    rt, rh = body.xmat.reshape(3, 3), hand.xmat.reshape(3, 3)
    height = next(c.height for c in RECEIVERS if c.name == tube)
    for i, point in enumerate(TIE_POINTS):
        eq = model.equality(f"{tube}_held{i}").id
        if on:
            local = np.array(point) * [1, 1, height]
            world = body.xpos + rt @ local
            model.eq_data[eq, 0:3] = local
            model.eq_data[eq, 3:6] = rh.T @ (world - hand.xpos)
        data.eq_active[eq] = 1 if on else 0


# The tube sizes uncapped, in the order tried. 100 mm tubes are left out, as v1 leaves them out of its
# task: in its rack, Step 1b grips a 100 mm tube 3 cm under its top whatever is asked, so Arm A's hand is
# right under the cap and Arm B's fingers cannot get round it (3 of 12 runs failed so).
SIZES = (0.15, 0.2)


def seen_size(result: dict) -> float:
    return float(result.get("size") or 0.0)


def run_episode(
    seed: int,
    sizes: list[float],
    perception=None,
    net=None,
    spread: bool = True,
    ties: bool = False,
    vary: int | None = None,
    vary_arms: bool = False,
    record_to: Path | None = None,
    test_from: int = 1,
    pause: float = 0.0,
    uncapper: type | None = None,
) -> dict:
    """One uncapping run. ``uncapper``: the Uncapper class to use (a learned one: learn/rollout.py)."""
    wall = time.time()
    var = variation(vary, vary_arms)
    # The cap's tightness is read by caps.step as it runs; the seal lets go to a sixth of it, as caps.py.
    if var:
        caps.BREAKAWAY_TORQUE, caps.RUNNING_TORQUE = var["breakaway_Nm"], var["breakaway_Nm"] / 6
    else:
        caps.BREAKAWAY_TORQUE, caps.RUNNING_TORQUE = DEFAULT_TIGHTNESS
    # Arm A picks with its pinch point on the tube's axis, so the tube is held between the pads' middles
    # (grip.py); with the ties it is Step 1b's grasp, as before.
    grip.use_hold_depth(grip.STEP1B_DEPTH if ties else grip.HOLD_DEPTH)
    model, truth = build(seed, spread, var["placed"] if var else None)
    data = mujoco.MjData(model)
    prepare(model, data, var["arm_start"] if var else None)
    u = (uncapper or Uncapper)(
        model, data, perception or CapPerception(fixtures=(holder.XY,)), net or load_net(), ties
    )
    u.containers = truth.containers
    u.recording = record_to is not None
    u.test_from = test_from
    u.pause = pause

    def allowed(tube: str) -> dict[str, set[str]]:
        cap_bodies = {caps.follower_name(tube), caps.cap_name(tube)}
        return {HOLD_ARM: {tube}, CAP_ARM: cap_bodies}

    result = u.run_uncap(sizes, allowed)
    result.update(
        {"seed": seed, "variation": var, "test_from": test_from, "wall_seconds": round(time.time() - wall, 1)}
    )
    # How long Arm B took, from gripping the cap to "cap off" (from its own messages).
    said = {msg: t for t, a, msg in reversed(result["log"]) if a == CAP_ARM}
    if "gripped the cap" in said and "cap off" in said:
        result["unscrew_seconds"] = round(said["cap off"] - said["gripped the cap"], 2)
    if record_to is not None and u.rec is not None:
        meta = {
            k: result.get(k) for k in ("seed", "size", "tube", "done", "failure", "quarters", "variation")
        }
        meta["seen_cap"] = u.report.get("seen_cap")
        record_to.mkdir(parents=True, exist_ok=True)
        path = record_to / f"seed{seed}_size{round(seen_size(result) * 1000)}.npz"
        u.rec.save(path, meta)
        result["recorded"] = str(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=150)
    parser.add_argument("--size", type=float, default=0.15)
    parser.add_argument("--ties", action="store_true", help="the old simplification: tube tied to the hand")
    parser.add_argument(
        "--vary", type=int, default=None, help="stress test: random differences, from this seed"
    )
    args = parser.parse_args()
    sizes = [args.size] + [s for s in SIZES if s != args.size]
    print(json.dumps(run_episode(args.seed, sizes, ties=args.ties, vary=args.vary), indent=1, default=str))


if __name__ == "__main__":
    main()
