"""Save the scene just after Step 3 has filled its balance tubes, so Step 4's tests can start from there.

Step 4 begins where Step 3 ends: the balance tubes filled with water and standing back in their supply-rack
cells. Filling takes a few minutes per tube, so it is done once per seed here and the simulation's state is
saved (``results/filled/seed<N>.npz``, with the plan and the true weights in ``seed<N>.json``). ``load_state``
rebuilds the same scene and puts that state back.

    cd step-4-load-rotor
    PYTHONPATH=$PWD:$PWD/../step-1-weigh:$PWD/../step-3-fill-balance-tube:$PWD/../step-2-plan:$PWD/../simulation-environment \\
        pixi run --manifest-path ../simulation-environment/pixi.toml python filled_states.py --seeds 1 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(
    0, str(HERE.parent / "step-3-fill-balance-tube")
)  # Step 3's evaluate (it finds the solver)

import evaluate as step3
import test_scenes
from centrifuge_scene import scene
from centrifuge_scene.settings import WATER_TUBES
from fill import Filler
from motion import Arm

OUT = HERE / "results" / "filled"
STATE = mujoco.mjtState.mjSTATE_INTEGRATION


def make(seed: int) -> dict:
    """Plan and fill the seed's balance tubes as Step 3's evaluation does; save the state after."""
    t0 = time.time()
    model, data, truth = test_scenes.build(seed, changed=False)
    weights = {n: round(s.mass * 1000, 2) for n, s in truth.samples.items()}
    plan = step3.solver.plan_load(weights)
    arm = Arm(model, data)
    filler = Filler(arm)
    arm.wait(0.3)
    filler.balance.tare()
    blanks = sorted(truth.blanks)
    fills = []
    for i, tube in enumerate(plan.balance_tubes):
        if i == 0:
            target = tube.grams
        else:
            target = step3.second_target(
                plan, weights, plan.balance_tubes[0].name, fills[0]["read_g"]
            )
        f = filler.fill(
            truth.cells[blanks[i]][1], truth.cells[WATER_TUBES[i]][1], target
        )
        fills.append(
            {
                "balance_tube": tube.name,
                "blank": blanks[i],
                "target_g": round(target, 2),
                "read_g": f.final_g,
                "error": f.error,
                "true_g": round(step3.true_grams(model, data, truth, blanks[i]), 3),
            }
        )
        if f.error:
            break
    arm.home()
    scene.settle(model, data, 0.5)
    OUT.mkdir(parents=True, exist_ok=True)
    state = np.zeros(mujoco.mj_stateSize(model, STATE))
    mujoco.mj_getState(model, data, state, STATE)
    np.savez_compressed(OUT / f"seed{seed}.npz", state=state, ctrl=data.ctrl.copy())
    blank_of = {f["balance_tube"]: f["blank"] for f in fills}
    meta = {
        "seed": seed,
        "scenario": truth.scenario,
        "plan": [
            {
                "hole": t.hole,
                "name": t.name,
                "tube": blank_of.get(t.name, t.name),
                "balance": t.balance,
                "grams": round(t.grams, 2),
            }
            for t in plan.tubes
        ],
        "fills": fills,
        "wall_seconds": round(time.time() - t0, 1),
    }
    (OUT / f"seed{seed}.json").write_text(json.dumps(meta, indent=1))
    return meta


def load_state(seed: int) -> tuple[mujoco.MjModel, mujoco.MjData, object, dict]:
    """The seed's scene as Step 3 left it: (model, data, truth, plan and fills)."""
    model, data, truth = test_scenes.build(seed, changed=False)
    saved = np.load(OUT / f"seed{seed}.npz")
    mujoco.mj_setState(model, data, saved["state"], STATE)
    data.ctrl[:] = saved["ctrl"]
    mujoco.mj_forward(model, data)
    meta = json.loads((OUT / f"seed{seed}.json").read_text())
    return model, data, truth, meta


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 3])
    p.add_argument("--workers", type=int, default=1, help="at most 4 on this Mac")
    a = p.parse_args()
    keys = ("seed", "scenario", "fills", "wall_seconds")
    if a.workers > 1:
        with ProcessPoolExecutor(a.workers) as pool:
            for meta in pool.map(make, a.seeds):
                print(json.dumps({k: meta[k] for k in keys}), flush=True)
    else:
        for seed in a.seeds:
            print(json.dumps({k: make(seed)[k] for k in keys}), flush=True)


if __name__ == "__main__":
    main()
