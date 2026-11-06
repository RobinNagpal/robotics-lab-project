"""Arm A's grip on the capped tube, firm enough to hold it while the cap is turned (0.6 N m).

Measured with grip_test.py (docs/uncapping-programmed-details.md, section 3). Step 1b's grip held the
tube against only 0.05 N m of twist. Three things were wrong, and each part here fixes one:

1. **Too deep in the hand** (HOLD_DEPTH). Step 1b's fingers go 11.5 mm past the tube's axis, and the
   tube ends up 12 mm behind the middle of the pads, at their inner end. There the Robotiq's fingers
   wrap round it (the gripper's "encompassing" grip): the squeeze is only 8 N and the fingertips give
   way. With the pinch point on the tube's axis, the tube sits between the pads' middles (within 5 mm
   on 8 tubes) and the squeeze is 45 N.
2. **The fingertips can swing** (finger_lock). In the model each fingertip is carried by a link held
   only by a weak spring (0.05 N m per radian); twisting the tube pushes one pad along its finger and
   that fingertip swings away (scene 150, at 0.5 N m). The real 2F-85 is self-locking: once closed, its
   fingers stay where they are with the object pressing on them (Robotiq). So once Arm A has closed
   tighter, every joint of both fingers is held where it is by a stiff spring.
3. **The contacts creep** (NOSLIP). MuJoCo's contacts are a little soft, and a held object slowly
   slides under a steady sideways force well below its friction limit. Its "noslip" pass removes that;
   AutoBio's uncapping uses it too (noslip_iterations 2). It is on only while the cap is turned.

With all three and the fingers told to close 10 mm past the tube (HOLD_SQUEEZE, about 87 N, inside the
real gripper's 20 to 235 N), the tube held 1.37 to over 1.5 N m on all 8 tubes tried (scenes 150 to 153,
150 and 200 mm), and turned at most 0.21 degrees in 2 s under a steady 0.7 N m.
"""

from __future__ import annotations

import mujoco
from carry import episode as carry_episode
from grasp import side
from lab_scene import scene as lab

# m past the tube's axis that Arm A's pinch point goes when it picks; Step 1b's own, read before any
# change, is 11.5 mm.
HOLD_DEPTH = 0.0
STEP1B_DEPTH = side.DEPTH
# Held from the side, the tube settles this much further back into the hand than the pinch point went
# past its axis (Step 1b: 11.5 mm past, held 15 mm behind the pinch point).
SETTLES_BACK = 0.0035
TUBE_WIDTH = 0.025  # m
HOLD_SQUEEZE = 0.010  # m: told to close this far narrower than the tube (Step 1b: 4 mm), about 87 N
FINGER_JOINTS = ("driver", "coupler", "spring_link", "follower")
JOINT_LOCK = 50.0  # N m per radian: each finger joint held where it is
JOINT_DAMPING = 0.5  # N m s per radian
NOSLIP = 2  # the noslip pass's iterations while the cap is turned (AutoBio's)
SETTLE = 0.6  # s for the fingers to close tighter before they are locked


def use_hold_depth(depth: float = HOLD_DEPTH) -> None:
    """Make Step 1b's side grasp go ``depth`` past the tube's axis (and know where the held tube then
    sits), in this process only: v1's files are not changed. Step 1b reads both at the time of the pick."""
    side.DEPTH = depth
    carry_episode.HELD_BEHIND = depth + SETTLES_BACK


def squeeze(model: mujoco.MjModel, data: mujoco.MjData, arm: str, mm: float = HOLD_SQUEEZE) -> None:
    """Tell ``arm``'s fingers to close ``mm`` narrower than the tube."""
    data.ctrl[lab.gripper_actuator_id(model, arm)] = lab.grip_command(TUBE_WIDTH, squeeze=mm)


_saved: dict = {}


def finger_lock(model: mujoco.MjModel, data: mujoco.MjData, arm: str, on: bool = True) -> None:
    """Hold every joint of ``arm``'s fingers where it is now (the self-locking 2F-85), or (``on=False``)
    give them back their own springs, so the gripper can open again."""
    for s in ("right", "left"):
        for j in FINGER_JOINTS:
            jnt = model.joint(f"{arm}/gripper/{s}_{j}_joint")
            jid, q, v = jnt.id, jnt.qposadr[0], jnt.dofadr[0]
            key = (id(model), jid)
            if on:
                _saved.setdefault(key, (model.jnt_stiffness[jid], model.qpos_spring[q], model.dof_damping[v]))
                model.jnt_stiffness[jid] = JOINT_LOCK
                model.qpos_spring[q] = data.qpos[q]
                model.dof_damping[v] = JOINT_DAMPING
            elif key in _saved:
                model.jnt_stiffness[jid], model.qpos_spring[q], model.dof_damping[v] = _saved.pop(key)


def hold_tight(model: mujoco.MjModel, data: mujoco.MjData, arm: str, on: bool) -> None:
    """Everything above but the squeeze's wait: the finger lock and the noslip pass on (``on``) or off.
    Call after ``squeeze`` and SETTLE seconds of stepping."""
    finger_lock(model, data, arm, on)
    model.opt.noslip_iterations = NOSLIP if on else 0
