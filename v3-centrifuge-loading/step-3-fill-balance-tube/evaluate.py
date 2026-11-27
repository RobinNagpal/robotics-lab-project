"""Measure the programmed Step 3 on seeds that need balance tubes, against the simulator's answers.

For each seed (odd seeds: "odd one" needs one balance tube, "mismatch" two):

1. **Plan** with Step 2's solver from the samples' weights as Step 1 reads them (the true weights to the
   balance's 0.01 g; Step 1 reads within 0.002 g, median).
2. **Fill** each balance tube with the arm: water0 for the first, water1 for the second. After the first is
   filled, the second's target is worked out again from what the first really weighs (the balance's reading):
   two balance tubes can cancel each other's errors, so the second makes up for the first.
3. **Grade** with the simulator's own answers:
   - each balance tube's true weight against its target (aim: 0.5 g with one balance tube, 0.3 g with two);
   - the rotor's imbalance with every tube's true weight in its planned hole (must be within 1.0 g);
   - balls spilled (in no tube at all), and how many of them;
   - every tube back in its cell (within 3 mm, standing within 3 degrees), nothing else moved over 2 mm;
   - simulated and wall-clock time.

    cd step-3-fill-balance-tube
    P="pixi run --manifest-path ../simulation-environment/pixi.toml"
    PYTHONPATH=$PWD:$PWD/../step-1-weigh:$PWD/../step-2-plan:$PWD/../simulation-environment \\
        $P python evaluate.py --seeds 20 --workers 4
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
from centrifuge_scene import liquid, scene
from centrifuge_scene.racks import cell_centres
from centrifuge_scene.settings import LAYOUT, LIQUID, TUBE, WATER_TUBES

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "step-2-plan"))
import solver  # noqa: E402
import test_scenes  # noqa: E402
from motion import Arm  # noqa: E402

from fill import Filler  # noqa: E402

OUT = HERE / "results"
IN_CELL = 0.003  # m
UPRIGHT_DEG = 3.0
DISTURBED = 0.002  # m
SEEDS = [s for s in range(1, 40, 2)]  # 20 seeds: odd seeds are the jobs with balance tubes


def tilt_deg(data: mujoco.MjData, body: int) -> float:
    return math.degrees(math.acos(min(1.0, abs(float(data.xmat[body].reshape(3, 3)[2, 2])))))


def second_target(plan: solver.Plan, weights: dict[str, float], first: str, first_g: float) -> float:
    """The weight for the plan's other balance tube, now the first one weighs ``first_g``: the one that
    leaves the least imbalance (kept within what an open tube can hold)."""
    masses = {t.hole: weights[t.name] for t in plan.tubes if not t.balance}
    holes = {t.name: t.hole for t in plan.tubes}
    masses[holes[first]] = first_g
    other = next(t for t in plan.balance_tubes if t.name != first)
    rx = sum(m * solver.DIRECTIONS[k][0] for k, m in masses.items())
    ry = sum(m * solver.DIRECTIONS[k][1] for k, m in masses.items())
    ex, ey = solver.DIRECTIONS[other.hole]
    lo = solver.EMPTY_TUBE_G + solver.TUBE_SPREAD_G + solver.CAP_G
    hi = solver.EMPTY_TUBE_G - solver.TUBE_SPREAD_G + solver.MAX_WATER_G + solver.CAP_G
    return min(hi, max(lo, -(rx * ex + ry * ey)))


def true_grams(model, data, truth, name: str) -> float:
    """A tube's true weight: its body (with cap, if any) and the balls inside it."""
    body = truth.samples[name].body_mass if name in truth.samples else truth.blanks.get(name, TUBE.body_mass)
    return 1000 * (body + len(liquid.balls_in(model, data, TUBE, name)) * LIQUID.ball_mass)


def run_seed(seed: int) -> dict:
    t0 = time.time()
    model, data, truth = test_scenes.build(seed, changed=False)
    bodies = {n: model.body(n).id for n in truth.tubes} | {r: model.body(r).id for r in scene.RACKS}
    mujoco.mj_forward(model, data)
    start = {n: data.xpos[b].copy() for n, b in bodies.items()}
    weights = {n: round(s.mass * 1000, 2) for n, s in truth.samples.items()}
    plan = solver.plan_load(weights)
    balance = plan.balance_tubes
    blanks = sorted(truth.blanks)
    arm = Arm(model, data)
    filler = Filler(arm)
    arm.wait(0.3)
    filler.balance.tare()

    fills, used = [], {}
    for i, tube in enumerate(balance):
        target = tube.grams if i == 0 else second_target(plan, weights, balance[0].name, fills[0]["read_g"])
        blank, water = blanks[i], WATER_TUBES[i]
        f = filler.fill(truth.cells[blank][1], truth.cells[water][1], target)
        used[tube.name] = blank
        row = asdict(f)
        row.update(
            balance_tube=tube.name,
            blank=blank,
            water=water,
            planned_g=round(tube.grams, 2),
            read_g=f.final_g,
            true_g=round(true_grams(model, data, truth, blank), 3),
        )
        row["off_true_g"] = round(row["true_g"] - row["target_g"], 3)
        fills.append(row)
        if f.error:
            break
    arm.home()
    scene.settle(model, data, 0.5)

    # The rotor with every tube's true weight in its planned hole.
    done = len(fills) == len(balance) and not any(r["error"] for r in fills)
    imbalance = None
    if done:
        masses = {}
        for t in plan.tubes:
            masses[t.hole] = true_grams(model, data, truth, used[t.name] if t.balance else t.name)
        imbalance = round(solver.imbalance(masses), 3)

    back = {}
    for name in [*used.values(), *WATER_TUBES[: len(fills)]]:
        rack, cell = truth.cells[name]
        b = bodies[name]
        off = float(np.linalg.norm(data.xpos[b][:2] - np.array(cell_centres(getattr(LAYOUT, rack))[cell])))
        back[name] = {"off_mm": round(off * 1000, 2), "tilt_deg": round(tilt_deg(data, b), 2)}
    handled = set(back)
    moved = {
        n: round(float(np.linalg.norm(data.xpos[b] - start[n])) * 1000, 2)
        for n, b in bodies.items()
        if n not in handled
    }
    return {
        "seed": seed,
        "job": truth.scenario,
        "weights": weights,
        "plan": plan.describe(),
        "fills": fills,
        "imbalance_g": imbalance,
        "spilled": liquid.spilled(model, data, truth.tubes),
        "back": back,
        "back_ok": all(v["off_mm"] < IN_CELL * 1000 and v["tilt_deg"] < UPRIGHT_DEG for v in back.values()),
        "disturbed": [n for n, mm in moved.items() if mm > DISTURBED * 1000],
        "wall_seconds": round(time.time() - t0, 1),
    }


def balanced(run: dict) -> bool:
    return run["imbalance_g"] is not None and run["imbalance_g"] <= 1.0


def aim(run: dict) -> float:
    return 0.5 if len(run["fills"]) == 1 else 0.3


def summarise(runs: list[dict]) -> str:
    fills = [f for r in runs for f in r["fills"]]
    tubes = sum(len(solver_tubes(r)) for r in runs)
    good = [f for f in fills if not f["error"]]
    offs = [abs(f["off_true_g"]) for f in good]
    lines = [
        f"seeds: {len(runs)}; balance tubes: {tubes}; filled without a refused move: {len(good)}",
        f"within 0.5 g of target: {sum(o <= 0.5 for o in offs)}; within 0.3 g: {sum(o <= 0.3 for o in offs)}",
        f"off target (true), g: median {statistics.median(offs):.2f}, max {max(offs):.2f}" if offs else "",
        f"over target by more than half a ball: {sum(f['over'] for f in good)}",
        f"rotor within 1.0 g (true weights): {sum(balanced(r) for r in runs)} of {len(runs)}",
        f"balls spilled: {sum(r['spilled'] for r in runs)} in all; seeds with none: "
        f"{sum(r['spilled'] == 0 for r in runs)}",
        f"tubes back in their cells: {sum(r['back_ok'] for r in runs)} of {len(runs)} seeds; "
        f"seeds where something else moved: {sum(bool(r['disturbed']) for r in runs)}",
        f"pour rounds per tube: median {statistics.median(len(f['rounds']) for f in good):.0f}"
        if good
        else "",
        f"simulated time per tube, s: median {statistics.median(f['seconds'] for f in good):.0f}"
        if good
        else "",
        f"wall-clock per seed, s: median {statistics.median(r['wall_seconds'] for r in runs):.0f}",
    ]
    for r in runs:
        for f in r["fills"]:
            if f["error"] or abs(f["off_true_g"]) > aim(r):
                what = f["error"] or f"off by {f['off_true_g']:+.2f} g"
                lines.append(f"  seed {r['seed']} {f['balance_tube']}: {what}")
    return "\n".join(line for line in lines if line)


def solver_tubes(run: dict) -> list[str]:
    return [line for line in run["plan"].splitlines() if "fill an empty tube" in line]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--only", type=int, nargs="*", help="just these seeds (not saved)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    seeds = args.only or SEEDS[: args.seeds]
    with ProcessPoolExecutor(args.workers) as pool:
        runs = list(pool.map(run_seed, seeds))
    text = summarise(runs)
    OUT.mkdir(exist_ok=True)
    name = "evaluation_programmed.json" if not args.only else "evaluation_some_seeds.json"
    (OUT / name).write_text(json.dumps(runs, indent=1, default=str))
    if args.only:
        for r in runs:
            for f in r["fills"]:
                print(
                    r["seed"],
                    f["balance_tube"],
                    "target",
                    f["target_g"],
                    "true",
                    f["true_g"],
                    "read",
                    f["read_g"],
                    "rounds",
                    len(f["rounds"]),
                    f["error"],
                )
                for rd in f["rounds"]:
                    print("   ", rd)
    print(text)
    if args.only:
        for r in runs:
            print(r["seed"], "back", r["back"], "disturbed", r["disturbed"], "spilled", r["spilled"])


if __name__ == "__main__":
    main()
