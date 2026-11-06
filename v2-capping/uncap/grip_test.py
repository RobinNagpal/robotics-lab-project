"""How much twist Arm A's grip on a capped tube holds, and why it gives way.

Arm A picks a capped tube and holds it up, exactly as in an uncapping run (episode.Uncapper, Step 1b's
pick). Then, instead of Arm B turning the cap, the tube is twisted about its own axis by a torque that
rises steadily (the way the cap's thread would drag it, anticlockwise from above) until the tube turns
in the hand. The cap needs 0.6 N m to start turning, so the grip must hold more than that.

Each variant (a grip setting) starts from the same held state, copied, so only the setting differs.

    python -m uncap.grip_test --seed 150 --size 0.15
"""

from __future__ import annotations

import argparse
import copy
import json
import math

import mujoco
import numpy as np
from carry.episode import EMPTY_CLEARANCE, Choice, load_net
from grasp import side
from lab_scene import scene as lab

from capping import caps

from . import episode, grip, holder
from .perceive import CapPerception

ARM = episode.HOLD_ARM
RATE = 0.3  # N m per s: how fast the twist rises
MAX_TORQUE = 1.5  # N m
SLIPPED = 2.0  # degrees the tube turns in the hand: "slipped"
SETTLE = 0.6  # s after a setting is changed, before twisting
NEED = 0.6  # N m: the cap's breakaway torque (caps.py)


def grasp_depth(depth: float | None) -> None:
    """Step 1b's side grasp ``depth`` past the tube's axis (grip.use_hold_depth); None leaves it."""
    if depth is not None:
        grip.use_hold_depth(depth)


def hold(seed: int, size: float, spread: bool = True):
    """Arm A holding a capped tube of ``size`` at the uncapping point, as in a run. None if it cannot."""
    model, truth = episode.build(seed, spread)
    data = mujoco.MjData(model)
    episode.prepare(model, data)
    u = episode.Uncapper(model, data, CapPerception(fixtures=(holder.XY,)), load_net())
    u.containers = truth.containers
    u.table = u.perception.table(model, data)
    u.transfer = max(side.TRANSFER, u.table.tallest_rim + EMPTY_CLEARANCE)
    chosen = u.choose_capped([size])
    if chosen is None:
        return None
    seen, grasp = chosen
    tube = u.truth_name(seen.xy)
    u.say(episode.CAP_ARM, "waiting")
    script = u.receiver_script(Choice(seen, None, grasp, None), tube)
    while "ready" not in u.flags:
        next(script)
        caps.step(model, data)
        if data.time > 120:
            raise RuntimeError("Arm A did not get the tube to the uncapping point")
    return model, data, tube, u.grips.get("receiver")


def _pads(model: mujoco.MjModel) -> list[int]:
    return [model.geom(f"{ARM}/gripper/{p}").id for p in lab.PAD_GEOMS]


def grip_state(model, data, tube: str) -> dict:
    """What the fingers do: the squeeze on the tube (sum of the pads' contact forces), the gripper motor's
    force, the finger joints, and where the tube's axis is between the pads along the fingers."""
    pads = set(_pads(model))
    tube_id = model.body(tube).id
    squeeze = 0.0
    for k in range(data.ncon):
        c = data.contact[k]
        if {c.geom1, c.geom2} & pads and tube_id in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]):
            f = np.zeros(6)
            mujoco.mj_contactForce(model, data, k, f)
            squeeze += f[0]
    joints = {}
    for side_ in ("right", "left"):
        for j in ("driver", "spring_link", "follower"):
            joints[f"{side_}_{j}"] = round(float(data.joint(f"{ARM}/gripper/{side_}_{j}_joint").qpos[0]), 3)
    # Along the fingers (the hand's approach axis): the tube's axis relative to the pads' middles (+ is
    # towards the fingertips). The pads are two boxes each, pad1 at the tip and pad2 nearer the palm.
    hand = data.body(episode.HAND)
    approach = data.site(lab.pinch_site(ARM)).xpos - hand.xpos
    approach /= np.linalg.norm(approach)
    tube_axis_point = data.body(tube).xpos
    tube_z = data.body(tube).xmat.reshape(3, 3)[:, 2]
    pad_mid = np.mean([data.geom_xpos[g] for g in pads], axis=0)
    # The point on the tube's axis level with the pads' middle.
    on_axis = tube_axis_point + tube_z * float(np.dot(pad_mid - tube_axis_point, tube_z))
    act = model.actuator(f"{ARM}/{lab.GRIPPER_ACTUATOR}").id
    return {
        "squeeze_N": round(squeeze, 1),
        "motor_force": round(float(data.actuator_force[act]), 2),
        "joints": joints,
        "axis_vs_pad_middle_mm": round(float(np.dot(on_axis - pad_mid, approach)) * 1000, 1),
    }


def _turned_in_hand(data, tube: str, start: np.ndarray) -> float:
    """Degrees the tube has turned about its own axis relative to the hand since ``start`` (the tube's
    x axis in the hand's frame then)."""
    rh = data.body(episode.HAND).xmat.reshape(3, 3)
    rt = data.body(tube).xmat.reshape(3, 3)
    x_now = rh.T @ rt[:, 0]
    axis = rh.T @ rt[:, 2]
    a = start - axis * np.dot(start, axis)
    b = x_now - axis * np.dot(x_now, axis)
    return math.degrees(math.atan2(np.dot(np.cross(a, b), axis), np.dot(a, b)))


def twist(model, data, tube: str) -> dict:
    """Twist the held tube with a rising torque until it turns SLIPPED degrees in the hand."""
    tid = model.body(tube).id
    before = grip_state(model, data, tube)
    rh = data.body(episode.HAND).xmat.reshape(3, 3)
    start = rh.T @ data.body(tube).xmat.reshape(3, 3)[:, 0]
    steps = round(MAX_TORQUE / RATE / model.opt.timestep)
    slip_at, worst = None, 0.0
    at_need = None
    curve = []  # (torque N m, degrees turned in the hand), every 0.1 N m
    for i in range(1, steps + 1):
        torque = RATE * i * model.opt.timestep
        axis = data.body(tube).xmat.reshape(3, 3)[:, 2]
        data.xfrc_applied[tid, 3:6] = torque * axis
        caps.step(model, data)
        turned = _turned_in_hand(data, tube, start)
        worst = max(worst, abs(turned))
        if i % round(0.1 / RATE / model.opt.timestep) == 0:
            curve.append((round(torque, 2), round(turned, 2)))
        if at_need is None and torque >= NEED:
            at_need = {"turned_deg": round(turned, 2), **grip_state(model, data, tube)}
        if abs(turned) > SLIPPED:
            slip_at = torque
            break
    after = grip_state(model, data, tube)
    data.xfrc_applied[tid] = 0
    return {
        "curve": curve,
        "holds_Nm": round(slip_at, 3) if slip_at else f">{MAX_TORQUE}",
        "holds_0.6": slip_at is None or slip_at > NEED,
        "before": before,
        "at_0.6": at_need,
        "at_slip": after,
    }


STEADY = 0.7  # N m: a little over the cap's breakaway
STEADY_TIME = 2.0  # s


def steady(model, data, tube: str, torque: float = STEADY) -> dict:
    """Twist the held tube with a steady ``torque`` (raised over 0.3 s) for STEADY_TIME: does it stop
    turning (the fingers give a little and hold) or keep turning (it slips)?"""
    tid = model.body(tube).id
    rh = data.body(episode.HAND).xmat.reshape(3, 3)
    start = rh.T @ data.body(tube).xmat.reshape(3, 3)[:, 0]
    ramp = round(0.3 / model.opt.timestep)
    steps = ramp + round(STEADY_TIME / model.opt.timestep)
    turned = []
    for i in range(1, steps + 1):
        axis = data.body(tube).xmat.reshape(3, 3)[:, 2]
        data.xfrc_applied[tid, 3:6] = torque * min(1.0, i / ramp) * axis
        caps.step(model, data)
        turned.append(_turned_in_hand(data, tube, start))
    data.xfrc_applied[tid] = 0
    last = round(0.5 / model.opt.timestep)
    rate = (turned[-1] - turned[-last]) / 0.5
    return {
        "torque_Nm": torque,
        "turned_deg_at": {
            f"{t}s": round(turned[min(len(turned), ramp + round(t / model.opt.timestep)) - 1], 2)
            for t in (0.0, 0.5, 1.0, 2.0)
        },
        "still_turning_deg_per_s": round(rate, 2),
        "holds": abs(rate) < 1.0 and abs(turned[-1]) < 10,
        "grip": grip_state(model, data, tube),
    }


# Grip settings tried, each on a copy of the same held state. Each is (what it changes, function).
def _noslip(model, data, tube):
    model.opt.noslip_iterations = 2  # AutoBio's: a second pass that removes slipping at contacts


def _squeeze(mm):
    def f(model, data, tube):
        data.ctrl[lab.gripper_actuator_id(model, ARM)] = lab.grip_command(2 * 0.0125, squeeze=mm / 1000)

    return f


def _stiff_pads(model, data, tube):
    for g in _pads(model):  # AutoBio's contact softness between fingers and tube
        model.geom_solref[g] = (0.002, 1.0)
        model.geom_solimp[g, :3] = (0.9, 0.95, 0.001)


LOCK_STIFFNESS = 2000.0  # of the tendon's length: stiff, so the fingers do not move
LOCK_FORCE = 100.0


def self_lock(model, data, arm: str = ARM) -> None:
    """The Robotiq 2F-85 is self-locking: closed on an object, its fingers stay where they are even with
    the object pushing against them, and with the power off. The model's finger motor is a weak position
    servo (force at most 5) that a twisting tube pushes open. Here, once the fingers have closed, the
    motor holds the fingers' present position stiffly instead: force = LOCK_STIFFNESS x (where they are
    now - where they are), up to LOCK_FORCE."""
    a = model.actuator(f"{arm}/{lab.GRIPPER_ACTUATOR}").id
    length = float(data.actuator_length[a])
    model.actuator_gainprm[a, 0] = 0.0
    model.actuator_biasprm[a, :3] = (LOCK_STIFFNESS * length, -LOCK_STIFFNESS, -10.0)
    model.actuator_forcerange[a] = (-LOCK_FORCE, LOCK_FORCE)


def finger_lock(model, data, arm: str = ARM) -> None:
    grip.finger_lock(model, data, arm, True)


def _finger_locked(*pre):
    def f(model, data, tube):
        for g in pre:
            g(model, data, tube)
        if pre:
            caps.step(model, data, round(SETTLE / model.opt.timestep))
        finger_lock(model, data)

    return f


def _locked(*pre):
    def f(model, data, tube):
        for g in pre:
            g(model, data, tube)
        if pre:
            caps.step(model, data, round(SETTLE / model.opt.timestep))
        self_lock(model, data)

    return f


def _all(*fs):
    def f(model, data, tube):
        for g in fs:
            g(model, data, tube)

    return f


VARIANTS = {
    "as in the run": lambda m, d, t: None,
    "noslip": _noslip,
    "squeeze 7 mm": _squeeze(7),
    "squeeze 10 mm": _squeeze(10),
    "squeeze 10 mm + noslip": _all(_squeeze(10), _noslip),
    "AutoBio contacts (noslip + stiff pads)": _all(_noslip, _stiff_pads),
    "squeeze 10 mm + AutoBio contacts": _all(_squeeze(10), _noslip, _stiff_pads),
    "squeeze 10 mm + noslip 10": _all(_squeeze(10), lambda m, d, t: setattr(m.opt, "noslip_iterations", 10)),
    "finger lock": _finger_locked(),
    "finger lock + noslip": _finger_locked(_noslip),
    "squeeze 10 mm + finger lock": _finger_locked(_squeeze(10)),
    "squeeze 10 mm + finger lock + noslip": _finger_locked(_squeeze(10), _noslip),
    "self-lock": _locked(),
    "squeeze 7 mm + self-lock": _locked(_squeeze(7)),
    "squeeze 10 mm + self-lock": _locked(_squeeze(10)),
}


def run(seed: int, size: float, names=None, depth: float | None = None) -> dict:
    grasp_depth(depth)
    held = hold(seed, size)
    if held is None:
        return {"seed": seed, "size": size, "failure": "no capped tube can be picked"}
    model, data, tube, below = held
    out = {
        "seed": seed,
        "size": size,
        "depth_mm": None if depth is None else depth * 1000,
        "tube": tube,
        "gripped_below_top_cm": below,
        "variants": {},
    }
    for name in names or VARIANTS:
        m = copy.copy(model)
        d = copy.copy(data)
        VARIANTS[name](m, d, tube)
        caps.step(m, d, round(SETTLE / m.opt.timestep))
        d2 = copy.copy(d)
        out["variants"][name] = twist(m, d, tube)
        out["variants"][name]["steady"] = steady(m, d2, tube)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=150)
    parser.add_argument("--size", type=float, default=0.15)
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--depth", type=float, default=None, help="m past the axis (Step 1b: 0.0115)")
    args = parser.parse_args()
    print(json.dumps(run(args.seed, args.size, args.variants, args.depth), indent=1, default=str))


if __name__ == "__main__":
    main()
