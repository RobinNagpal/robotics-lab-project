"""Step 4, programmed: load every tube of the plan into its hole, in the plan's order, and grade the rotor.

Starts from the scene as Step 3 left it (``filled_states.py``: the balance tubes filled and back in their
supply-rack cells; for jobs without balance tubes, the scene as built). For each tube of Step 2's plan, in
its loading order (balance tubes first): pick it, slide it into its hole (rotor.py), back to the start pose.

Graded with the simulator's own answers (the examiner's side, `grading`):
- every tube seated in its planned hole (tip within 6 mm of the bottom, axis within 3 degrees);
- the rotor's imbalance from the tubes' true weights in the holes they are really in (limit 1.0 g);
- balls spilled; the simulation blowing up; time.

    cd step-4-load-rotor
    PYTHONPATH=$PWD:$PWD/../step-1-weigh:$PWD/../step-3-fill-balance-tube:$PWD/../step-2-plan:$PWD/../simulation-environment \\
        pixi run --manifest-path ../simulation-environment/pixi.toml python load.py --seeds 0 1 2 3 --workers 4
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from centrifuge_scene import grading, liquid, scene
from centrifuge_scene.settings import TUBE
from motion import Arm, Blocked

from filled_states import load_state
from rotor import Loader, hole_frame

OUT = Path(__file__).parent / "results"


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(
        math.acos(float(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))
    )


def run_seed(seed: int, mode: str = "plain") -> dict:
    t0 = time.time()
    model, data, truth, meta = load_state(seed)
    arm = Arm(model, data)
    loader = Loader(arm, mode)
    spilled_before = liquid.spilled(model, data, truth.tubes)
    rows, error = [], ""
    for i, t in enumerate(meta["plan"]):
        name, hole = t["tube"], t["hole"]
        rack, cell = truth.cells[name]
        row = {
            "order": i,
            "tube": name,
            "plan_name": t["name"],
            "hole": hole,
            "balance": t["balance"],
            "balls": len(liquid.balls_in(model, data, TUBE, name)),
        }
        try:
            if i:
                loader.return_home()
            sl, turn = loader.load(name, rack, cell, hole, truth.capped[name])
            row.update(
                turn_deg=turn,
                slide_stop=sl.stop,
                pauses=sl.pauses,
                peak_side_n=round(sl.peak_side_n, 1),
                peak_push_n=round(sl.peak_push_n, 1),
            )
        except Blocked as e:
            row["error"] = error = str(e)
        body = model.body(name).id
        seat, hole_rot = hole_frame(data, hole)
        row["tip_to_bottom_mm"] = round(
            float(np.linalg.norm(data.xpos[body] - seat)) * 1000, 2
        )
        row["axis_vs_hole_deg"] = round(
            angle_deg(data.xmat[body].reshape(3, 3)[:, 2], hole_rot[:, 2]), 2
        )
        row["balls_after"] = len(liquid.balls_in(model, data, TUBE, name))
        rows.append(row)
        if error:
            break
    if not error:
        try:
            loader.return_home()
        except Blocked as e:
            error = f"going home: {e}"
    scene.settle(model, data, 0.5)
    g = grading.grade(model, data, truth)
    planned = {t["hole"]: t["tube"] for t in meta["plan"]}
    blew_up = any(np.abs(data.xpos[model.body(n).id]).max() > 2.0 for n in truth.tubes)
    return {
        "seed": seed,
        "scenario": truth.scenario,
        "mode": mode,
        "tubes_planned": len(planned),
        "tubes_seated_as_planned": sum(
            g.seated.get(k) == n for k, n in planned.items()
        ),
        "all_seated": all(g.seated.get(k) == n for k, n in planned.items()),
        "imbalance_g": round(g.imbalance * 1000, 3),
        "balanced": g.balanced,
        "balls_spilled": int(liquid.spilled(model, data, truth.tubes) - spilled_before),
        "sim_blew_up": blew_up,
        "error": error,
        "smallest_gap_mm": round(min(loader.gaps, default=0.01) * 1000, 1),
        "sim_seconds": round(arm.time, 1),
        "wall_seconds": round(time.time() - t0, 1),
        "tubes": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    p.add_argument("--mode", choices=("plain", "force"), default="plain")
    p.add_argument("--workers", type=int, default=1, help="at most 4 on this Mac")
    p.add_argument("--name", default="evaluation")
    a = p.parse_args()
    runs = []
    with ProcessPoolExecutor(a.workers) as pool:
        for r in pool.map(run_seed, a.seeds, [a.mode] * len(a.seeds)):
            runs.append(r)
            mark = (
                "ok "
                if r["all_seated"] and r["balanced"] and not r["balls_spilled"]
                else "BAD"
            )
            print(
                f"{mark} seed {r['seed']:2d} {r['scenario']:8s} seated {r['tubes_seated_as_planned']}/"
                f"{r['tubes_planned']} imbalance {r['imbalance_g']:.2f} g spilled {r['balls_spilled']} "
                f"blew_up {r['sim_blew_up']} {r['wall_seconds']:.0f} s {r['error']}",
                flush=True,
            )
    OUT.mkdir(exist_ok=True)
    (OUT / f"{a.name}_{a.mode}.json").write_text(json.dumps(runs, indent=1))


if __name__ == "__main__":
    main()
