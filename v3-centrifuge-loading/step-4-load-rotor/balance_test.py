"""Step 4: load one filled balance tube (open, water inside) into its planned hole.

Starts from the scene as Step 3 left it (``filled_states.py``): the balance tubes filled and back in their
supply-rack cells. The arm picks the plan's balance tube by its top (Step 3's grip: 6 mm under the rim, 1 mm
squeeze), tilts it to the hole's 35 degrees and slides it in (rotor.py).

Measured from the simulator (the examiner's side): the tube's angle in the fingers, whether it is seated, and
the balls: how many left the tube (spilled) on the way.

    cd step-4-load-rotor
    PYTHONPATH=$PWD:$PWD/../step-1-weigh:$PWD/../step-3-fill-balance-tube:$PWD/../step-2-plan:$PWD/../simulation-environment \\
        pixi run --manifest-path ../simulation-environment/pixi.toml python balance_test.py --seed 1
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from centrifuge_scene import grading, liquid
from centrifuge_scene.settings import TUBE
from motion import Arm, Blocked
from moves import hand_rot

from filled_states import load_state
from rotor import BACK, MAX_TILT, Grip, Loader, grip_at_hole, hole_frame

OUT = Path(__file__).parent / "results"


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(
        math.acos(float(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))
    )


def run(seed: int, which: int, mode: str, hole: int | None, miss_mm: float) -> dict:
    t0 = time.time()
    model, data, truth, meta = load_state(seed)
    balance = [t for t in meta["plan"] if t["balance"]]
    entry = balance[which]
    name = entry["tube"]
    hole = entry["hole"] if hole is None else hole
    rack, cell = truth.cells[name]
    body = model.body(name).id
    out: dict = {
        "seed": seed,
        "tube": name,
        "hole": hole,
        "mode": mode,
        "miss_mm": miss_mm,
    }
    balls_before = len(liquid.balls_in(model, data, TUBE, name))
    spilled_before = liquid.spilled(model, data, truth.tubes)
    out["balls_in_tube"] = balls_before

    def in_fingers() -> float:
        return round(
            angle_deg(data.xmat[body].reshape(3, 3)[:, 2], -hand_rot(arm)[:, 2]), 2
        )

    arm = Arm(model, data)
    loader = Loader(arm, mode)
    grip = Grip(capped=False)
    seat, hole_rot = hole_frame(data, hole)
    axis = hole_rot[:, 2]
    believed = seat + miss_mm / 1000 * hole_rot[:, 0]
    worst = [0.0]

    def tick(_loader, *_):
        worst[0] = max(worst[0], in_fingers())

    loader.on_tick = tick
    try:
        loader.pick(name, rack, cell, grip)
        out["in_fingers_after_pick_deg"] = in_fingers()
        out["balls_after_pick"] = len(liquid.balls_in(model, data, TUBE, name))
        turn, gap = loader.best_turn(hole, believed, grip)
        out["turn_deg"], out["planned_gap_mm"] = turn, round(gap * 1000, 1)
        start, rot = grip_at_hole(believed + BACK[hole] * axis, hole_rot, turn, grip)
        out["way_in_gap_mm"] = round(loader.go_to_start(start, rot, MAX_TILT) * 1000, 1)
        out["in_fingers_on_arrival_deg"] = in_fingers()
        arm.wait(0.2)
        out["balls_after_tilt"] = len(liquid.balls_in(model, data, TUBE, name))
        sl = loader.slide(hole, believed, turn, grip)
        out.update(
            slide_stop=sl.stop,
            slide_seconds=sl.seconds,
            pauses=sl.pauses,
            peak_push_n=round(sl.peak_push_n, 1),
            peak_side_n=round(sl.peak_side_n, 1),
        )
        out["worst_in_fingers_deg"] = round(worst[0], 2)
        if sl.jammed:
            raise Blocked(sl.stop)
        loader.let_go(hole)
        arm.wait(1.0)
    except Blocked as e:
        out["error"] = str(e)
    tip = data.xpos[body].copy()
    out["tip_to_bottom_mm"] = round(float(np.linalg.norm(tip - seat)) * 1000, 2)
    out["axis_vs_hole_deg"] = round(
        angle_deg(data.xmat[body].reshape(3, 3)[:, 2], axis), 2
    )
    out["seated"] = grading.tubes_in_rotor(model, data, truth).get(hole) == name
    out["balls_after"] = len(liquid.balls_in(model, data, TUBE, name))
    out["balls_spilled"] = int(
        liquid.spilled(model, data, truth.tubes) - spilled_before
    )
    out["sim_blew_up"] = bool(np.abs(data.xpos[body]).max() > 2.0)
    out["wall_seconds"] = round(time.time() - t0, 1)
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--which",
        type=int,
        default=0,
        help="which of the plan's balance tubes (0 or 1)",
    )
    p.add_argument("--mode", choices=("plain", "force"), default="plain")
    p.add_argument("--hole", type=int, default=None, help="default: the plan's hole")
    p.add_argument("--miss-mm", type=float, default=0.0)
    a = p.parse_args()
    r = run(a.seed, a.which, a.mode, a.hole, a.miss_mm)
    print(json.dumps(r, indent=1))
    OUT.mkdir(exist_ok=True)
    tag = (
        f"balance_seed{a.seed}_tube{a.which}_hole{r['hole']}_{a.mode}_miss{a.miss_mm:g}"
    )
    (OUT / f"{tag}.json").write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
