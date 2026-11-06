"""A trial of the screw cap with the real arm and gripper: what does the arm feel?

Arm A's gripper comes down over a capped receiver standing in its rack, grips the cap, and then:

1. **pulls straight up** without turning: the cap must stay on;
2. **turns it in quarter turns**, as a gripper with a limited wrist does: grip, turn 90 degrees
   anticlockwise (rising with the thread), open, turn back, grip again. After every quarter turn it
   **tries to lift** the cap 6 mm, holds there, and notes the force that takes.

The tube is held still the whole time, as the other arm would hold it: here by tying it to where it
stands, a little springy as a gripper is.

It records only what a real UR5e can sense: the torque in its wrist joint while turning (its joint
currents) and the force on the gripper while lifting (its wrist force sensor). Whether the cap is still
on the thread is read from the simulation only to check those readings, never to decide anything.

    python -m capping.cap_trial --seed 150 --tube receiver_150
"""

from __future__ import annotations

import argparse
import json
import math

import mujoco
import numpy as np
from lab_scene import kinematics
from lab_scene import scene as lab
from lab_scene.settings import RECEIVERS

from . import caps, scene

ARM = "arm2"  # Arm A, next to the empty rack
QUARTER = math.pi / 2
TURN_TIME = 1.0  # s per quarter turn
# The lift that tells a free cap from one still on its thread: up 6 mm and hold, then the pull while
# holding. (3 mm was too little: the arm, holding the cap down a little after turning, sags 1.5 to 2.4 mm
# below where it is told to be, so a short lift only let go of that and never pulled.)
LIFT_TEST = 0.006  # m
LIFT_TIME, LIFT_HOLD = 0.4, 0.3  # s
OPEN_WIDTH = 0.040  # m: the fingers open to this to let go of the 28.8 mm cap
# How to hold the cap, from a sweep of heights and squeezes: the pads 4 mm above the middle of the cap's
# side (lower, they also press on the tube below it), told to close 10 mm past it. That grips with about
# 70 N (a real Robotiq 2F-85 grips with 20 to 235 N), enough for about 1.2 N m of turning, twice what a
# tight cap takes to start.
GRIP_ABOVE_MIDDLE = 0.004  # m
GRIP_SQUEEZE = 0.010  # m
# Lifting a cap off its thread takes its own weight (0.04 N); one still on it holds back with tens of
# newtons. The UR5e's wrist force sensor is good to about 4 N, so 5 N tells them apart.
FREE_FORCE = 5.0  # N


class Trial:
    def __init__(self, seed: int, tube: str):
        def hold(spec: mujoco.MjSpec) -> None:
            """The other arm holds the tube where it stands: three of its points tied to where they are,
            two on its axis and one 10 cm out to the side, so that it cannot turn (1 N m turns it 0.4
            degrees). (A weld was unstable with the cap's stiff thread on the same tube.)"""
            height = next(c.height for c in RECEIVERS if c.name == tube)
            for i, point in enumerate(([0, 0, 0.2 * height], [0, 0, 0.85 * height], [0.10, 0, 0.5 * height])):
                spec.add_equality(
                    name=f"other_arm_holds_tube{i}",
                    type=mujoco.mjtEq.mjEQ_CONNECT,
                    objtype=mujoco.mjtObj.mjOBJ_BODY,
                    name1=tube,
                    data=point + [0] * (mujoco.mjNEQDATA - 3),
                    solref=[0.004, 1],
                )

        self.model, self.truth = scene.build(seed, edit=hold)
        m = self.model
        self.data = d = mujoco.MjData(m)
        scene.reset(m, d)
        self.tube = tube
        # The cap is two bodies: on the thread, and free once off (lab_scene/caps.py).
        self.caps = {m.body(caps.follower_name(tube)).id, m.body(caps.cap_name(tube)).id}
        self.arm_act = lab.arm_actuator_ids(m, ARM)
        self.arm_q = lab.arm_joint_ids(m, ARM)
        self.wrist3_act = self.arm_act[5]
        self.grip_act = lab.gripper_actuator_id(m, ARM)
        self.arm_bodies = {b for b in range(m.nbody) if m.body(b).name.startswith(f"{ARM}/")}
        self.step(300)

    # --- the simulation ------------------------------------------------------------------------
    def step(self, n: int = 1, log: list | None = None) -> None:
        m, d = self.model, self.data
        for _ in range(n):
            caps.step(m, d)
            if log is not None:
                log.append(self.sense())

    def sense(self) -> dict:
        """What the arm can feel: its wrist joint's torque, and the force its fingers put on the cap
        (what the wrist force sensor reads, less the gripper's own weight)."""
        m, d = self.model, self.data
        force, squeeze = np.zeros(3), 0.0
        for k in range(d.ncon):
            c = d.contact[k]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if ({b1, b2} & self.caps) and ({b1, b2} & self.arm_bodies):
                f6 = np.zeros(6)
                mujoco.mj_contactForce(m, d, k, f6)
                world = c.frame.reshape(3, 3).T @ f6[:3]  # on geom2, in the world
                squeeze += f6[0]
                force += world if b2 in self.caps else -world
        return {
            "t": d.time,
            "wrist_torque": float(d.actuator_force[self.wrist3_act]),
            "pull": float(force[2]),  # up on the cap, N
            "squeeze": squeeze,
            # for checking only, never used to decide:
            "turned": caps.turned(m, d, self.tube),
            "on_thread": caps.on_thread(m, d, self.tube),
            "cap_up_mm": 1000 * self.cap_height(),
        }

    def cap_height(self) -> float:
        d = self.data
        cap = caps.cap_body(self.model, d, self.tube)
        return float(d.xpos[cap][2] - (d.xpos[self.model.body(self.tube).id][2] + self.seat_offset()))

    def seat_offset(self) -> float:
        return float(self.model.body_pos[self.model.body(caps.follower_name(self.tube)).id][2])

    # --- the arm -------------------------------------------------------------------------------
    def move(self, target: np.ndarray, seconds: float, wrist3: float | None = None, log=None) -> None:
        """Slide the joint targets to put the pinch point at ``target``, fingers down; ``wrist3``, if
        given, sets the last joint (the turn about the fingers' axis), which does not move the point."""
        m, d = self.model, self.data
        start = d.ctrl[self.arm_act].copy()
        work = mujoco.MjData(m)
        work.qpos[:] = d.qpos
        if wrist3 is not None:
            # Set the turn first and solve the rest from there: the turn about the fingers' axis is
            # free in the solve, so it stays, and the point between the fingers stays over the cap.
            # (Setting it after the solve swung that point 3 mm round, as it is not exactly on the
            # last joint's axis.)
            work.qpos[self.arm_q[5]] = wrist3
        mujoco.mj_kinematics(m, work)
        goal = kinematics.solve(m, work, target, ARM)
        steps = max(1, round(seconds / m.opt.timestep))
        for i in range(1, steps + 1):
            d.ctrl[self.arm_act] = start + (goal - start) * i / steps
            self.step(1, log)

    def grip(self, width: float, seconds: float = 0.4, squeeze: float = GRIP_SQUEEZE) -> None:
        d = self.data
        start, goal = d.ctrl[self.grip_act], lab.grip_command(width, squeeze=squeeze)
        steps = round(seconds / self.model.opt.timestep)
        for i in range(1, steps + 1):
            d.ctrl[self.grip_act] = start + (goal - start) * i / steps
            self.step()
        self.step(200)

    def pinch(self) -> np.ndarray:
        return self.data.site(lab.pinch_site(ARM)).xpos.copy()


def steady(log: list, key: str, start: float = 0.2, end: float = 0.8) -> float:
    """The average of ``key`` over part of a move (by default its middle, away from the jolt of starting
    and stopping)."""
    part = log[round(start * len(log)) : max(round(end * len(log)), round(start * len(log)) + 1)]
    return float(np.mean([s[key] for s in part]))


def run(seed: int, tube: str, quarters: int = 8) -> dict:
    t = Trial(seed, tube)
    m, d = t.model, t.data
    cap_top = d.xpos[caps.cap_body(t.model, d, tube)].copy()  # the middle of the inside of the cap's top
    grip_at = cap_top - [0, 0, (caps.SKIRT - caps.TOP) / 2 - GRIP_ABOVE_MIDDLE]
    report: dict = {"seed": seed, "tube": tube}

    # Down over the cap, open. There, turn the wrist a quarter and back without touching anything: the
    # torque that takes is the wrist's own (its motor moving its own inertia), to take off the readings
    # later, as a real arm's are zeroed ("tared").
    # (Done 4 cm above the cap with the fingers open only as wide as for a re-grip: wide open beside a
    # short tube, the turning fingers hit the taller one next to it in the rack.)
    d.ctrl[t.grip_act] = 0
    t.move(grip_at + [0, 0, 0.08], 2.0)
    t.grip(OPEN_WIDTH, squeeze=0.0)
    t.move(grip_at + [0, 0, 0.04], 1.0)
    w0 = float(d.qpos[t.arm_q[5]])
    log: list = []
    t.move(grip_at + [0, 0, 0.04], TURN_TIME, wrist3=w0 - QUARTER, log=log)
    tare = np.array([s["wrist_torque"] for s in log])
    t.move(grip_at + [0, 0, 0.04], TURN_TIME, wrist3=w0)
    t.move(grip_at, 1.0, wrist3=w0)
    fifth = len(tare) // 5
    report["wrist_torque_turning_nothing_Nm"] = round(float(np.mean(tare[fifth:-fifth])), 3)
    t.grip(2 * caps.OUTER_R)
    report["grip_squeeze_N"] = round(t.sense()["squeeze"], 1)

    # 1. Pull straight up 6 mm, then 10 mm, without turning.
    pulls = []
    for lift in (LIFT_TEST, 0.010):
        log: list = []
        t.move(grip_at + [0, 0, lift], 0.5, log=log)
        t.step(200, log)
        pulls.append(
            {
                "lift_mm": lift * 1000,
                "peak_pull_N": round(max(s["pull"] for s in log), 1),
                "cap_up_mm": round(log[-1]["cap_up_mm"], 2),
                "fingers_slid_mm": round(1000 * (t.pinch()[2] - grip_at[2]) - log[-1]["cap_up_mm"], 1),
                "on_thread": log[-1]["on_thread"],
            }
        )
        t.move(grip_at, 0.5)
    report["pull_without_turning"] = pulls
    # The fingers may have slid on the cap: grip it again where it is.
    t.grip(OPEN_WIDTH, squeeze=0.0)
    t.grip(2 * caps.OUTER_R)

    # 2. Quarter turns. Anticlockwise seen from above is a negative turn of the last joint (the
    # fingers point down, so the joint's axis points down). Every move aims at a fixed plan: the
    # cap's axis, a quarter of the thread's pitch higher after every quarter turn.
    rows = []
    axis, z = grip_at[:2], float(grip_at[2])

    def at(height: float) -> np.ndarray:
        return np.array([axis[0], axis[1], height])

    for q in range(1, quarters + 1):
        start = t.sense()
        log: list = []
        z += caps.PITCH / 4
        t.move(at(z), TURN_TIME, wrist3=w0 - QUARTER, log=log)
        turning = log
        net = np.array([s["wrist_torque"] for s in turning])[: len(tare)] - tare[: len(turning)]
        # Try to lift it 6 mm, still holding it: a cap off its thread comes up for its own weight.
        log = []
        t.move(at(z + LIFT_TEST), LIFT_TIME, wrist3=w0 - QUARTER, log=log)
        hold: list = []
        t.step(round(LIFT_HOLD / m.opt.timestep), hold)
        probe = log + hold
        lift_force = float(np.mean([s["pull"] for s in hold]))
        free = lift_force < FREE_FORCE  # decided from what the arm feels only
        rows.append(
            {
                "quarter": q,
                # The torque the cap takes: the wrist's reading less what turning nothing took at the
                # same moment of the same move. Its largest, and its average over the middle.
                "peak_torque_Nm": round(float(np.max(np.abs(net))), 3),
                "steady_torque_Nm": round(float(np.mean(np.abs(net[len(net) // 5 : -len(net) // 5]))), 3),
                "lift_force_N": round(lift_force, 2),
                "arm_says_free": free,
                # for checking only:
                "cap_turned_deg": round(360 * (turning[-1]["turned"] - start["turned"]), 1),
                "total_turned": round(probe[-1]["turned"], 3),
                "on_thread": probe[-1]["on_thread"],
            }
        )
        if free:
            break
        # Down again, let go, turn back, grip again.
        t.move(at(z), LIFT_TIME, wrist3=w0 - QUARTER)
        t.grip(OPEN_WIDTH, squeeze=0.0)
        t.move(at(z), 0.6, wrist3=w0)
        t.grip(2 * caps.OUTER_R)
    report["quarter_turns"] = rows
    report["unstable"] = bool(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=150)
    parser.add_argument("--tube", default="receiver_150")
    parser.add_argument("--quarters", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(run(args.seed, args.tube, args.quarters), indent=1))


if __name__ == "__main__":
    main()
