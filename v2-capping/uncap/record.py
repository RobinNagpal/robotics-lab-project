"""Recording Arm B's unscrewing, as demonstrations for a learned policy (ACT).

While Arm B unscrews the cap and lifts it away, this samples, RATE times a second:

- **what the arm senses** (the policy's possible inputs): its 6 joint angles and speeds, the gripper's
  opening, the force and torque at its wrist (a sensor between the wrist and the gripper, where a UR5e's
  own sensor sits), and its 6 joint motors' torques. The wrist force includes the gripper's weight
  (10.3 N along the tool in free air); a real sensor is zeroed for that, so each run stores ``tare``, the
  reading in free air at its start, to subtract;
- **what the teacher commands** (the actions to learn): the 6 joint targets and the gripper command;
- **labels, for analysing and grading only** (never inputs): which part of the programmed loop it is in,
  which quarter turn, and from the simulator how far the cap has turned and whether it is still on.

Values are saved clean: sensor noise (a UR5e's wrist sensor is good to about 4 N and 0.3 N m) is added
when the data are used, so it can be varied without recording again.

Each run is one .npz file: arrays of shape (samples, ...) plus the run's facts in ``meta`` (JSON).
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from lab_scene import scene as lab

from capping import caps

RATE = 50  # samples per second
TARE_TIME = 0.5  # s at the start, the arm moving in free air, for zeroing the wrist sensor
ARM = "arm1"  # Arm B
FT_SITE = "wrist_ft"  # site on the gripper's mount, for the wrist force/torque sensor
# The parts of the programmed loop, numbered for the labels.
PHASES = ("approach", "grip", "turn", "test", "down", "open", "turn back", "regrip", "lift away")


def add_wrist_sensors(spec: mujoco.MjSpec, arm: str = ARM) -> None:
    """A force and a torque sensor between ``arm``'s wrist and its gripper (the UR5e's own wrist sensor)."""
    mount = spec.body(f"{arm}/gripper/base_mount")
    mount.add_site(name=f"{arm}/{FT_SITE}", pos=[0, 0, 0], size=[0.005, 0, 0])
    spec.add_sensor(
        name=f"{arm}/wrist_force",
        type=mujoco.mjtSensor.mjSENS_FORCE,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname=f"{arm}/{FT_SITE}",
    )
    spec.add_sensor(
        name=f"{arm}/wrist_torque",
        type=mujoco.mjtSensor.mjSENS_TORQUE,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname=f"{arm}/{FT_SITE}",
    )


class Recorder:
    """Samples Arm B every 1/RATE s while ``on``; ``phase`` and ``quarter`` are set by the teacher."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, tube: str):
        self.model, self.data, self.tube = model, data, tube
        self.q = lab.arm_joint_ids(model, ARM)
        self.dof = np.array([model.joint(f"{ARM}/{j}").dofadr[0] for j in lab.ARM_JOINTS])
        self.act = lab.arm_actuator_ids(model, ARM)
        self.grip = lab.gripper_actuator_id(model, ARM)
        self.driver = model.joint(f"{ARM}/gripper/right_driver_joint").qposadr[0]
        self.force = model.sensor(f"{ARM}/wrist_force")
        self.torque = model.sensor(f"{ARM}/wrist_torque")
        self.every = max(1, round(1.0 / RATE / model.opt.timestep))
        self.steps = 0
        self.on = False
        self.phase = "approach"
        self.quarter = 0
        self.rows: dict[str, list] = {
            k: []
            for k in (
                "time", "q", "dq", "grip_opening", "wrist_force", "wrist_torque", "joint_torque",
                "action_q", "action_grip", "phase", "quarter", "cap_turned", "on_thread", "cap_z",
            )
        }  # fmt: skip

    def step(self) -> None:
        """Call once per simulation step."""
        self.steps += 1
        if not self.on or self.steps % self.every:
            return
        m, d, r = self.model, self.data, self.rows
        r["time"].append(d.time)
        r["q"].append(d.qpos[self.q].copy())
        r["dq"].append(d.qvel[self.dof].copy())
        r["grip_opening"].append(float(d.qpos[self.driver]))  # the finger linkage's angle (0 open)
        r["wrist_force"].append(d.sensordata[self.force.adr[0] : self.force.adr[0] + 3].copy())
        r["wrist_torque"].append(d.sensordata[self.torque.adr[0] : self.torque.adr[0] + 3].copy())
        r["joint_torque"].append(d.actuator_force[self.act].copy())
        r["action_q"].append(d.ctrl[self.act].copy())
        r["action_grip"].append(float(d.ctrl[self.grip]))
        r["phase"].append(PHASES.index(self.phase))
        r["quarter"].append(self.quarter)
        r["cap_turned"].append(caps.turned(m, d, self.tube))
        r["on_thread"].append(caps.on_thread(m, d, self.tube))
        r["cap_z"].append(float(d.xpos[caps.cap_body(m, d, self.tube)][2]))

    def save(self, path: Path, meta: dict) -> None:
        arrays = {k: np.asarray(v) for k, v in self.rows.items()}
        # The zeroing ("tare"): the wrist's reading in free air, over the first TARE_TIME s of the approach.
        start = arrays["time"] < arrays["time"][0] + TARE_TIME
        tare = {
            "force": arrays["wrist_force"][start].mean(axis=0).round(4).tolist(),
            "torque": arrays["wrist_torque"][start].mean(axis=0).round(5).tolist(),
        }
        meta = {**meta, "rate": RATE, "phases": PHASES, "samples": len(self.rows["time"]), "tare": tare}
        np.savez_compressed(path, meta=json.dumps(meta, default=str), **arrays)
