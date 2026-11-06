"""How much twist AutoBio's grips hold before slipping, and which of their settings it comes from.

Each seed runs AutoBio's expert to the moment both arms hold the tube (arm 1 the cap, arm 2 the body).
From that state, copied, each variant changes one setting, waits 0.6 s for it to settle, and then:

- **tube test** (arm 2's grip, the one our UR5e + 2F-85 cannot make hold): arm 1 lets go of the cap and
  moves 5 cm up out of the way, and the tube is twisted about its own axis by a torque rising steadily
  at 0.3 N m/s (anticlockwise from above, the way turning the cap off drags it);
- **cap test** (arm 1's grip on the cap): the thread is made frictionless and the same rising torque
  turns the cap; arm 2 holds the tube, which the cap no longer drags.

Slipped: turned 2 degrees in the hand. The torque then is what the grip holds. Our cap needs 0.6 N m to
start turning; AutoBio's loose cap needed at most 0.15 N m in its own runs.

    python -m autobio_uncap.grip_test --seeds 0-4
"""

from __future__ import annotations

import argparse
import copy
import json
from multiprocessing import get_context
from pathlib import Path

import mujoco
import numpy as np

from . import thread
from .evaluate import seeds as parse_seeds
from .expert import Expert, Options, Pose

OUT = Path(__file__).resolve().parents[1] / "runs" / "autobio"
RATE = 0.3  # N m per s
MAX_TORQUE = 1.5  # N m
SLIPPED = 2.0  # degrees
SETTLE = 0.6  # s

# One change each from AutoBio's settings. A setting that is a gripper command is (arm, command).
VARIANTS = {
    "autobio": {},
    "noslip_0": {"noslip": 0},  # no no-slip pass after the main solver
    "impratio_1": {"impratio": 1.0},  # MuJoCo's default: friction as soft as the normal force
    "pyramidal": {"cone": 0},  # the pyramid approximation of the friction cone, MuJoCo's default
    "no_pairs": {"pairs": False},  # fingers touch only the tube's and cap's collision boxes
    "pairs_only": {"boxes": False},  # fingers touch only the pair cylinders
    "finger_condim_3": {"finger_condim": 3},  # no twisting friction at the fingers' other contacts
    "pair_condim_6": {"pair_condim": 6},  # twisting friction in the pairs too
    "friction_0.5": {"friction": 0.5},  # half the friction everywhere the fingers touch
    "friction_2": {"friction": 2.0},  # twice
    "grip_0.010": {"grip": 0.010},  # gripper told to close less far (softer squeeze)
    "grip_0.002": {"grip": 0.002},  # told to close as far as allowed (harder squeeze)
}


def apply(e: Expert, model: mujoco.MjModel, v: dict, test: str) -> None:
    m = model
    fingers = e.arm1.finger_geoms | e.arm2.finger_geoms
    if "noslip" in v:
        m.opt.noslip_iterations = v["noslip"]
    if "impratio" in v:
        m.opt.impratio = v["impratio"]
    if "cone" in v:
        m.opt.cone = v["cone"]
    pair_geoms = {m.geom("7/body_cyl").id, m.geom("6/cap_cyl").id, m.geom("6f/cap_cyl").id}
    if v.get("pairs") is False:
        for g in pair_geoms:
            m.geom_size[g] = thread.TINY
            m.geom_rbound[g] = thread.TINY
    if v.get("boxes") is False:
        for g in e.tube_geoms | e.cap_geoms[e.thread.cap]:
            if g not in pair_geoms and m.geom_contype[g]:
                m.geom_contype[g] = m.geom_conaffinity[g] = 0
    if "finger_condim" in v:
        for g in fingers:
            m.geom_condim[g] = v["finger_condim"]
    if "pair_condim" in v:
        m.pair_dim[:] = v["pair_condim"]
    if "friction" in v:
        f = v["friction"]
        m.pair_friction[:, :2] = f
        for g in fingers | e.tube_geoms | e.cap_geoms[e.thread.cap]:
            m.geom_friction[g, 0] = f


def _twist(e: Expert, model, data, body: int, site: int, grip: tuple[int, float] | None) -> dict:
    """Ramp a torque on ``body`` about the tube's axis until it turns SLIPPED degrees in the hand."""
    m, d = model, data
    th = e.thread
    if grip is not None:
        d.ctrl[grip[0]] = grip[1]
    for _ in range(round(SETTLE / m.opt.timestep)):
        thread.step(m, d, th)

    def in_hand():
        return d.site_xmat[site].reshape(3, 3).T @ d.xmat[body].reshape(3, 3)

    start = in_hand()
    arm = e.arm2 if body == e.tube else e.arm1
    obj = e.tube_geoms if body == e.tube else e.cap_geoms[th.cap]
    saved = e.model, e.data
    e.model, e.data = m, d
    squeeze = e.squeeze(arm, obj)
    e.model, e.data = saved
    gripper_force = float(d.actuator_force[arm.gripper_id])
    torque, held = 0.0, None
    while torque < MAX_TORQUE:
        torque += RATE * m.opt.timestep
        axis = d.xmat[e.tube].reshape(3, 3)[:, 2]
        d.xfrc_applied[body, 3:] = torque * axis
        thread.step(m, d, th)
        if Expert.slip_deg(start, in_hand()) > SLIPPED:
            held = torque
            break
    d.xfrc_applied[body] = 0
    return {
        "holds_Nm": round(held, 3) if held is not None else f">{MAX_TORQUE}",
        "squeeze_N": round(squeeze, 1),
        "gripper_actuator_N": round(gripper_force, 1),
    }


def run_seed(seed: int) -> dict:
    e = Expert(Options())
    e.reset(seed)
    e.hand_over()
    held = mujoco.MjData(e.model)
    mujoco.mj_copyData(held, e.model, e.data)
    out = {}
    for name, v in VARIANTS.items():
        out[name] = {}
        for test in ("tube", "cap"):
            model = copy.deepcopy(e.model)
            data = mujoco.MjData(model)
            mujoco.mj_copyData(data, e.model, held)
            th = thread.Thread(model)
            e2 = Expert.__new__(Expert)
            e2.__dict__.update(e.__dict__)
            e2.model, e2.data, e2.thread = model, data, th
            e2.planner.ik = None
            apply(e2, model, v, test)
            if test == "tube":
                # Arm 1 lets go of the cap and moves up out of the way.
                e2.gripper_control(0.037, e2.arm1)
                p = e2.arm1.get_site_pose(data)
                e2.position_track(e2.arm1, Pose(p.pos + (0, 0, 0.05), p.quat), 5)
                grip = (e2.arm2.gripper_id, v["grip"]) if "grip" in v else None
                r = _twist(e2, model, data, e2.tube, e2.arm2.site_id, grip)
            else:
                th.friction = 0.0  # a frictionless thread: only arm 1's fingers hold the cap
                grip = (e2.arm1.gripper_id, v["grip"]) if "grip" in v else None
                r = _twist(e2, model, data, th.cap, e2.arm1.site_id, grip)
            out[name][test] = r
    return {"seed": seed, "variants": out}


def main() -> None:
    p = argparse.ArgumentParser(description="Twist test of AutoBio's grips.")
    p.add_argument("--seeds", default="0-4")
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    with get_context("spawn").Pool(a.workers) as pool:
        rows = pool.map(run_seed, parse_seeds(a.seeds), chunksize=1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "grip_test.json").write_text(json.dumps(rows, indent=1))
    head = f"{'tube: holds N m':>24s} {'squeeze N':>10s} {'cap: holds N m':>24s} {'squeeze N':>10s}"
    print(f"{'variant':16s} {head}")
    for name in VARIANTS:

        def col(test, key, name=name):
            return [r["variants"][name][test][key] for r in rows]

        def fmt(xs):
            return " ".join(str(x) for x in xs)

        print(
            f"{name:16s} {fmt(col('tube', 'holds_Nm')):>24s} {np.mean(col('tube', 'squeeze_N')):10.1f} "
            f"{fmt(col('cap', 'holds_Nm')):>24s} {np.mean(col('cap', 'squeeze_N')):10.1f}"
        )


if __name__ == "__main__":
    main()
