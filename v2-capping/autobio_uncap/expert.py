"""AutoBio's scripted uncapping expert (``screw_loosen.py``, ``ScrewLooseExpert``), ported to run here.

The moves are theirs, in order, with their numbers:

1. Both grippers open to 0.030. Arm 1 goes above the tube (14 cm over its bottom), hand pointing down,
   and comes down 1.5 cm; at the same time arm 2 goes to a point 10 cm to the side of where the tube will
   be held. Arm 1 closes on the cap (gripper 0.02).
2. Arm 1 lifts 10 cm, pulling the tube out of the rack by its cap, and carries it to a random point in a
   5 cm ball round (0, 0, 1.05). Arm 2 moves in sideways and grips the tube's body 5.5 cm lower, tight
   (gripper 0.005).
3. Unscrewing, four rounds: arm 1 turns its wrist in -18 degree steps (4 steps the first round, then 8),
   so 72 + 3 x 144 = 504 degrees in all, 1.4 turns. After each of the first three rounds it opens (0.037),
   goes 1 cm up, turns its wrist back to +75 degrees, comes down and grips again. After the fourth it
   does not open but lifts 2 cm (3 cm above where it gripped). No sensing: a fixed number of turns.
4. Graded by the simulator (``check``): the cap 15 cm above the tube's bottom (it sits at 12.0 cm on),
   arm 1's left finger touching the cap, and the threads no longer touching.

The arm's paths are AutoBio's too: each move is a straight line of poses, solved with the arm's exact
IK, splined and timed as fast as 3 rad/s and 2.5 rad/s^2 per joint allow, sent at 20 Hz to the joints'
position servos. The random draws are made in AutoBio's order, so seed ``n`` here is their seed ``n``
(the arm's start pose and the tube's hole from ``reset``, the hand-over point in ``execute``).

Changes, each a switch (``Options``), to move it towards our case one at a time:
- ``tight``: our cap's 0.6 N m to start turning (thread.py);
- ``lift_test``: our lift test after every round (6 mm up, hold 0.3 s; under 5 N pulling = off) instead
  of the fixed four rounds;
- ``turns``: where the thread ends.

    python -m autobio_uncap.expert --seed 0
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field

import mujoco
import numpy as np

from . import scene, thread
from .kin import AlohaAnalyticalIK, Pose, slerp
from .topp import Topp

# Our lift test (v2 uncap/episode.py).
LIFT_TEST = 0.006  # m up
LIFT_HOLD = 0.3  # s
FREE_FORCE = 5.0  # N: pulling less than this while lifted means the cap is off
MAX_ROUNDS = 8


@dataclass
class Options:
    tight: bool = False
    lift_test: bool = False
    turns: float = thread.TURNS
    hold_grip: float = 0.005  # arm 2's gripper command on the tube (AutoBio's)
    cap_grip: float = 0.02  # arm 1's on the cap (AutoBio's)
    free_force: float = FREE_FORCE  # N: the lift test's threshold
    friction: float = 1.0  # where the fingers touch the tube and cap (AutoBio's 1)


class AlohaArm:
    """AutoBio's AlohaArm: the 6 arm joints, their position servos, the gripper and its IK."""

    def __init__(self, model: mujoco.MjModel, prefix: str):
        self.model = model
        self.prefix = prefix
        self.jnt_adr = model.joint(f"{prefix}left/waist").qposadr.item()
        self.act_id = model.actuator(f"{prefix}left/waist").id
        self.site_id = model.site(f"{prefix}left/gripper").id
        self.gripper_id = model.actuator(f"{prefix}left/gripper").id
        self.dof = 6
        self.jnt_span = range(self.jnt_adr, self.jnt_adr + self.dof)
        self.act_span = range(self.act_id, self.act_id + self.dof)
        bodies = [model.body(f"{prefix}left/{f}_finger_link").id for f in ("left", "right")]
        self.finger_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] in bodies}
        self.ik: AlohaAnalyticalIK | None = None

    def register_ik(self, data: mujoco.MjData) -> None:
        parent = self.model.body(f"{self.prefix}left/base_link").parentid.item()
        self.ik = AlohaAnalyticalIK(Pose(data.xpos[parent].copy(), data.xquat[parent].copy()))

    def get_site_pose(self, data: mujoco.MjData) -> Pose:
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, data.site_xmat[self.site_id])
        return Pose(data.site_xpos[self.site_id].copy(), quat)

    @staticmethod
    def qpos_perturb() -> np.ndarray:
        lows = (-0.1, 0.0, 0.0, -0.025, 0.0, -0.05)
        highs = (0.1, 0.15, 0.05, 0.025, 0.15, 0.05)
        return np.random.uniform(lows, highs)


def randomsphere(x, y, z, d):
    """AutoBio's: a point uniformly inside a ball of diameter d."""
    radius = d / 2.0
    while True:
        direction = np.random.normal(0, 1, size=3)
        norm = np.linalg.norm(direction)
        if norm > 0:
            break
    u = np.random.uniform(0, 1)
    return np.array([x, y, z]) + direction / norm * radius * u ** (1 / 3)


@dataclass
class Record:
    seed: int
    options: dict
    ok_their_check: bool = False
    tube_held: bool = False
    cap_held: bool = False
    nothing_dropped: bool = False
    cap_off: bool = False
    error: str | None = None
    turned: float = 0.0  # turns the cap was undone
    commanded: float = 0.0  # turns arm 1's wrist was told to make while gripping
    rounds: list = field(default_factory=list)
    tube_slip_deg: float = 0.0  # how far the tube turned in arm 2's hand while the cap was turned
    max_thread_torque: float = 0.0
    hold_squeeze_N: float = 0.0  # arm 2's fingers' total normal force on the tube, once gripped
    cap_squeeze_N: float = 0.0
    cap_height: float = 0.0
    time: float = 0.0


class Expert:
    def __init__(self, options: Options | None = None, freq: int = 20):
        self.opt = options or Options()
        self.spec = scene.load_spec(self.opt.turns)
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.thread = thread.Thread(self.model, tight=self.opt.tight)
        self.dt = self.model.opt.timestep
        self.freq = freq
        self.period = int(round(1.0 / self.dt / self.freq))
        self.arm1 = AlohaArm(self.model, "1/aloha:")
        self.arm2 = AlohaArm(self.model, "2/aloha:")
        m = self.model
        self.tube = m.body(thread.TUBE).id
        self.tube_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == self.tube}
        self.cap_geoms = {
            b: {g for g in range(m.ngeom) if m.geom_bodyid[g] == b}
            for b in (self.thread.cap, self.thread.free)
        }
        self.cap_cyl = {self.thread.cap: m.geom("6/cap_cyl").id, self.thread.free: m.geom("6f/cap_cyl").id}
        self.leftfinger = m.geom("1/aloha:leftfinger").id
        self.on_step = None  # optional callback(expert), every step
        self.phase = ""  # what the arms are doing now, in plain words (for pictures and clips)
        if self.opt.friction != 1.0:
            m.pair_friction[:, :2] = self.opt.friction
            for g in (
                self.arm1.finger_geoms
                | self.arm2.finger_geoms
                | self.tube_geoms
                | set().union(*self.cap_geoms.values())
            ):
                m.geom_friction[g, 0] = self.opt.friction

    # --- AutoBio's Task / ScrewLoose ---
    def reset(self, seed: int) -> None:
        np.random.seed(seed)
        scene.reset(self.model, self.data, self.thread)
        d = self.data
        for arm in (self.arm1, self.arm2):
            p = arm.qpos_perturb()
            d.qpos[list(arm.jnt_span)] += p
            d.ctrl[list(arm.act_span)] += p
        # CentrifugeTube.randomposition: one tube type (50 ml), a hole of the 5 x 2 grid.
        np.random.choice(["50ml"])
        x = -0.072 + np.random.randint(0, 5) * 0.036
        y = -0.018 + np.random.randint(0, 2) * 0.036
        for j in ("centrifuge_50ml_screw_body", "centrifuge_50ml_screw_cap"):
            d.joint(j).qpos[0], d.joint(j).qpos[1] = x, y
        self.seed = seed
        self.rec = Record(seed=seed, options=asdict(self.opt))
        self.arm1.register_ik(d)
        self.arm2.register_ik(d)
        self.planner = Topp(dof=6, qc_vel=3.0, qc_acc=2.5, ik=None)

    def step_and_log(self) -> None:
        thread.step(self.model, self.data, self.thread)
        if self.on_step is not None:
            self.on_step(self)

    # --- ScrewLooseExpert ---
    @staticmethod
    def interpolate(start: Pose, end: Pose, num_steps: int) -> list[Pose]:
        return [
            Pose(
                (1 - i / num_steps) * start.pos + i / num_steps * end.pos,
                slerp(start.quat, end.quat, i / num_steps),
            )
            for i in range(num_steps + 1)
        ]

    def path_follow(self, path: list[Pose], arm: AlohaArm) -> None:
        self.planner.ik = arm.ik.solve
        trajectory = self.planner.jnt_traj(path)
        num_steps = int((trajectory.duration + 0.2) / self.dt)
        for step in range(num_steps):
            if step % self.period == 0:
                self.data.ctrl[list(arm.act_span)] = self.planner.query(trajectory, step * self.dt)
            self.step_and_log()

    def path_follow_dual(self, path_1, path_2, arm_1: AlohaArm, arm_2: AlohaArm) -> None:
        self.planner.ik = arm_1.ik.solve
        trajectory_1 = self.planner.jnt_traj(path_1)
        self.planner.ik = arm_2.ik.solve
        trajectory_2 = self.planner.jnt_traj(path_2)
        num_steps = int((max(trajectory_1.duration, trajectory_2.duration) + 0.2) / self.dt)
        for step in range(num_steps):
            if step % self.period == 0:
                t = step * self.dt
                self.data.ctrl[list(arm_1.act_span)] = Topp.query(trajectory_1, t)
                self.data.ctrl[list(arm_2.act_span)] = Topp.query(trajectory_2, t)
            self.step_and_log()

    def gripper_control(self, value: float, arm: AlohaArm) -> None:
        self.data.ctrl[arm.gripper_id] = value
        for _ in range(200):
            self.step_and_log()

    def position_track(self, arm: AlohaArm, target: Pose, num_steps: int) -> None:
        self.path_follow(self.interpolate(arm.get_site_pose(self.data), target, num_steps), arm)

    def _turn(self, now_pos: Pose, inner_loops: int) -> Pose:
        """Arm 1's wrist turned in -18 degree steps about the hand's axis (AutoBio's inner loop, with its
        3 micrometre rise per step)."""
        now_path = []
        for _ in range(inner_loops):
            ang = -np.pi / 10
            new_quat = np.zeros(4)
            mujoco.mju_mulQuat(new_quat, now_pos.quat, np.array([np.cos(ang / 2), np.sin(ang / 2), 0.0, 0.0]))
            next_pos = Pose(pos=now_pos.pos + (0.0, 0.0, 0.003 / 1000), quat=new_quat)
            now_path.extend(self.interpolate(now_pos, next_pos, 20)[1:])
            now_pos = next_pos
        before = self.thread.turned(self.data)
        self.phase = f"Round {len(self.rec.rounds) + 1}: arm 1 turns the cap"
        self.path_follow(now_path, self.arm1)
        self.rec.commanded += inner_loops / 20
        self.rec.rounds.append(
            {"wrist_turns": inner_loops / 20, "cap_turns": round(self.thread.turned(self.data) - before, 3)}
        )
        return now_pos

    def _regrip(self, mid1_pos: Pose, next_pos: Pose) -> None:
        """Open, back up 1 cm, turn the wrist back (to +75 degrees), come down, as AutoBio does."""
        mid1_pos.quat = np.array([0.560977785, 0.430454196, 0.560977785, -0.430454196])
        mid2_pos = Pose(pos=next_pos.pos, quat=mid1_pos.quat)
        self.phase = "Arm 1 lets go, turns its wrist back, grips again"
        self.gripper_control(0.037, self.arm1)
        self.position_track(self.arm1, mid1_pos, 5)
        self.position_track(self.arm1, mid2_pos, 5)

    def cap_loose(self, gripper_value: float, loops: int, mid1_pos: Pose) -> None:
        for j in range(loops):
            self.gripper_control(gripper_value, self.arm1)
            next_pos = self._turn(self.arm1.get_site_pose(self.data), 4 if j == 0 else 8)
            if j == 3:
                self.phase = "Arm 1 lifts the cap away"
                self.position_track(
                    self.arm1, Pose(pos=mid1_pos.pos + (0.0, 0.0, 0.02), quat=next_pos.quat), 5
                )
            else:
                self._regrip(mid1_pos, next_pos)

    def cap_loose_lift_test(self, gripper_value: float, mid1_pos: Pose) -> None:
        """Rounds as AutoBio's, but after each one our lift test decides whether the cap is off."""
        for j in range(MAX_ROUNDS):
            self.gripper_control(gripper_value, self.arm1)
            next_pos = self._turn(self.arm1.get_site_pose(self.data), 4 if j == 0 else 8)
            up = Pose(pos=next_pos.pos + (0.0, 0.0, LIFT_TEST), quat=next_pos.quat)
            z0 = self._heights()
            self.phase = "Lift test: does the cap come up freely?"
            self.position_track(self.arm1, up, 5)
            pulls = []
            for _ in range(round(LIFT_HOLD / self.dt)):
                self.step_and_log()
                pulls.append(self.cap_pull())
            pull = float(np.mean(pulls))
            rise = (self._heights() - z0) * 1000
            self.rec.rounds[-1].update(
                lift_pull_N=round(pull, 2),
                on_thread=self.thread.on(self.data),
                rise_mm={
                    k: round(float(v), 1)
                    for k, v in zip(("hand", "cap", "tube", "holding_hand"), rise, strict=True)
                },
            )
            if pull < self.opt.free_force:
                self.phase = "Cap is free: arm 1 lifts it away"
                self.position_track(
                    self.arm1, Pose(pos=mid1_pos.pos + (0.0, 0.0, 0.02), quat=next_pos.quat), 5
                )
                return
            self.position_track(self.arm1, next_pos, 5)
            self._regrip(mid1_pos, next_pos)

    def execute(self) -> None:
        self.hand_over()
        self.unscrew()

    def hand_over(self) -> None:
        """Steps 1 and 2: arm 1 takes the tube out of the rack by its cap, arm 2 grips its body."""
        d = self.data
        self.arm1.ik.initial_qpos = d.qpos[list(self.arm1.jnt_span)].copy()
        self.arm2.ik.initial_qpos = d.qpos[list(self.arm2.jnt_span)].copy()
        gripper_val = self.opt.cap_grip
        self.gripper_control(0.030, self.arm1)
        self.gripper_control(0.030, self.arm2)
        target_quat1 = np.array([0.7071, 0.0, 0.7071, 0.0])
        target_quat1 /= np.linalg.norm(target_quat1)
        target_quat2 = np.array([0.7071, 0.0, 0.0, 0.7071])
        target_quat2 /= np.linalg.norm(target_quat2)
        tube_pos = Pose(d.xpos[self.tube].copy(), d.xquat[self.tube].copy())
        arm1_pose = self.arm1.get_site_pose(d)
        arm2_pose = self.arm2.get_site_pose(d)
        tube_pos_random = Pose(pos=randomsphere(0.0, 0.0, 1.05, 0.05), quat=target_quat1)

        tube_pos_mid1 = Pose(pos=tube_pos_random.pos + (0.0, -0.1, -0.055), quat=target_quat2)
        tube_pos_mid2 = Pose(pos=tube_pos.pos + (0.0, 0.0, 0.14), quat=target_quat1)
        tube_pos_mid3 = Pose(pos=tube_pos_mid2.pos + (0.0, 0.0, -0.015), quat=target_quat1)
        path_1_1 = self.interpolate(arm1_pose, tube_pos_mid2, 5)
        path_2_1 = self.interpolate(arm2_pose, tube_pos_mid1, 5)
        path_1_1.extend(self.interpolate(tube_pos_mid2, tube_pos_mid3, 5)[1:])
        self.phase = "Arm 1 reaches down to the cap"
        self.path_follow_dual(path_1_1, path_2_1, self.arm1, self.arm2)
        self.phase = "Arm 1 grips the cap"
        self.gripper_control(gripper_val, self.arm1)
        self.rec.cap_squeeze_N = round(self.squeeze(self.arm1, self.cap_geoms[self.thread.cap_body(d)]), 2)

        tube_pos_mid4 = Pose(pos=self.arm1.get_site_pose(d).pos + (0.0, 0.0, 0.1), quat=target_quat1)
        tube_pos_mid5 = Pose(pos=tube_pos_random.pos + (0.0, 0.0, -0.055), quat=target_quat2)
        path_1_3 = self.interpolate(self.arm1.get_site_pose(d), tube_pos_mid4, 5)
        path_1_3.extend(self.interpolate(tube_pos_mid4, tube_pos_random, 5)[1:])
        self.phase = "Arm 1 lifts the tube out of the rack, by its cap"
        self.path_follow(path_1_3, self.arm1)
        self.phase = "Arm 2 reaches for the tube"
        self.position_track(self.arm2, tube_pos_mid5, 5)
        self.phase = "Arm 2 grips the tube tightly"
        self.gripper_control(self.opt.hold_grip, self.arm2)
        self.rec.hold_squeeze_N = round(self.squeeze(self.arm2, self.tube_geoms), 2)
        self.tube_pos_random, self.target_quat1 = tube_pos_random, target_quat1

    def unscrew(self) -> None:
        """Step 3: the rounds of turning."""
        gripper_val = self.opt.cap_grip
        start = self.tube_in_hand()
        tube_pos_mid6 = Pose(pos=self.tube_pos_random.pos + (0.0, 0.0, 0.01), quat=self.target_quat1)
        if self.opt.lift_test:
            self.cap_loose_lift_test(gripper_val, tube_pos_mid6)
        else:
            self.cap_loose(gripper_val, 4, tube_pos_mid6)
        self.rec.tube_slip_deg = round(self.slip_deg(start, self.tube_in_hand()), 1)

    # --- measurements (ours) ---
    def contact_force(self, geoms_a: set, geoms_b: set) -> np.ndarray:
        """Total force (world frame) the contacts between the two sets put on ``geoms_b``."""
        d = self.data
        total = np.zeros(3)
        f6 = np.zeros(6)
        for k in range(d.ncon):
            c = d.contact[k]
            g1, g2 = c.geom1, c.geom2
            if (g1 in geoms_a and g2 in geoms_b) or (g2 in geoms_a and g1 in geoms_b):
                mujoco.mj_contactForce(self.model, d, k, f6)
                world = c.frame.reshape(3, 3).T @ f6[:3]  # force on geom2, from geom1
                total += world if g2 in geoms_b else -world
        return total

    def squeeze(self, arm: AlohaArm, geoms: set) -> float:
        """Sum of the normal forces between the arm's fingers and the object."""
        d = self.data
        f6 = np.zeros(6)
        total = 0.0
        for k in range(d.ncon):
            c = d.contact[k]
            if (c.geom1 in arm.finger_geoms and c.geom2 in geoms) or (
                c.geom2 in arm.finger_geoms and c.geom1 in geoms
            ):
                mujoco.mj_contactForce(self.model, d, k, f6)
                total += f6[0]
        return float(total)

    def cap_pull(self) -> float:
        """The up force arm 1's fingers put on the cap (what its wrist would feel pulling)."""
        return float(
            self.contact_force(self.arm1.finger_geoms, self.cap_geoms[self.thread.cap_body(self.data)])[2]
        )

    def _heights(self) -> np.ndarray:
        """Heights of arm 1's hand, the cap, the tube and arm 2's hand."""
        d = self.data
        return np.array(
            [
                d.site_xpos[self.arm1.site_id][2],
                d.xpos[self.thread.cap_body(d)][2],
                d.xpos[self.tube][2],
                d.site_xpos[self.arm2.site_id][2],
            ]
        )

    def tube_in_hand(self) -> np.ndarray:
        """The tube's orientation in arm 2's gripper frame."""
        d = self.data
        return d.site_xmat[self.arm2.site_id].reshape(3, 3).T @ d.xmat[self.tube].reshape(3, 3)

    @staticmethod
    def slip_deg(a: np.ndarray, b: np.ndarray) -> float:
        r = b @ a.T
        return math.degrees(math.acos(np.clip((np.trace(r) - 1) / 2, -1, 1)))

    def touching(self, geoms_a: set, geoms_b: set) -> bool:
        d = self.data
        return any(
            (c.geom1 in geoms_a and c.geom2 in geoms_b) or (c.geom2 in geoms_a and c.geom1 in geoms_b)
            for c in d.contact[: d.ncon]
        )

    def check(self) -> bool:
        """AutoBio's ScrewLoose.check: cap 15 cm above the tube's bottom, arm 1's left finger on the cap,
        threads apart (here: the cap is off the thread)."""
        d = self.data
        cap = self.thread.cap_body(d)
        height = d.xpos[cap][2] - d.xpos[self.tube][2]
        self.rec.cap_height = round(float(height), 4)
        return bool(
            height > 0.15 and self.touching({self.leftfinger}, {self.cap_cyl[cap]}) and not self.thread.on(d)
        )

    def finish(self) -> Record:
        d = self.data
        r = self.rec
        r.ok_their_check = self.check()
        r.turned = round(self.thread.turned(d), 3)
        r.cap_off = not self.thread.on(d)
        cap = self.thread.cap_body(d)
        r.tube_held = self.touching(self.arm2.finger_geoms, self.tube_geoms)
        r.cap_held = self.touching(self.arm1.finger_geoms, self.cap_geoms[cap])
        # Dropped: the tube or the cap not in a hand, or below the hand-over height (a fall).
        low = min(d.xpos[self.tube][2], d.xpos[cap][2]) < 0.9
        r.nothing_dropped = bool(r.tube_held and r.cap_held and not low)
        r.max_thread_torque = round(self.thread.max_torque, 3)
        r.time = round(float(d.time), 2)
        return r

    def run(self, seed: int) -> Record:
        self.reset(seed)
        try:
            self.execute()
        except ValueError as e:  # IK could not reach a pose
            self.rec.error = str(e)[:200]
        return self.finish()


def main() -> None:
    p = argparse.ArgumentParser(description="One run of AutoBio's uncapping expert, ported.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tight", action="store_true")
    p.add_argument("--lift-test", action="store_true")
    p.add_argument("--turns", type=float, default=thread.TURNS)
    a = p.parse_args()
    e = Expert(Options(tight=a.tight, lift_test=a.lift_test, turns=a.turns))
    print(json.dumps(asdict(e.run(a.seed)), indent=1))


if __name__ == "__main__":
    main()
