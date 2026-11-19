"""Measure the solver on 100 seeds (25 per job), as results.md reports.

    cd step-2-plan
    PYTHONPATH=$PWD:$PWD/../simulation-environment \
      pixi run --manifest-path ../simulation-environment/pixi.toml python evaluate.py --workers 4

The solver gets each sample's weight as Step 1 would read it (the true weight plus up to 0.05 g of balance
error). The plan is then graded with the true weights (the examiner's side), with each balance tube filled up
to 0.5 g off its target, as Step 3 allows.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from centrifuge_scene import grading, scene
from solver import FILL_ERROR_G, TOLERANCE_G, imbalance, plan_load

SEEDS = range(100)
READ_ERROR_G = 0.05
EXPECTED_BALANCE_TUBES = {"pairs": 0, "odd_one": 1, "triangle": 0, "mismatch": 2}


def one(seed: int) -> dict:
    _, truth = scene.build(seed)
    true = {n: s.mass * 1000 for n, s in truth.samples.items()}
    rng = np.random.default_rng(1000 + seed)
    read = {n: w + float(rng.uniform(-READ_ERROR_G, READ_ERROR_G)) for n, w in true.items()}
    t0 = time.perf_counter()
    plan = plan_load(read)
    ms = (time.perf_counter() - t0) * 1000
    row = {"seed": seed, "job": truth.scenario, "samples": len(true), "ms": round(ms, 1)}
    if plan is None:
        return row | {"found": False}
    # The examiner's number: true sample weights; balance tubes filled to target, then off by up to 0.5 g.
    masses = {t.hole: (t.grams if t.balance else true[t.name]) for t in plan.tubes}
    exact = imbalance(masses)
    filled = []
    for _ in range(200):
        m = dict(masses)
        for t in plan.balance_tubes:
            m[t.hole] += float(rng.uniform(-FILL_ERROR_G, FILL_ERROR_G))
        filled.append(imbalance(m))
    # The same number from the simulation's own grading code (kg), to be sure both count holes the same way.
    same = abs(grading.imbalance({k: g / 1000 for k, g in masses.items()}) * 1000 - exact) < 1e-9
    return row | {
        "found": True,
        "balance_tubes": len(plan.balance_tubes),
        "expected": EXPECTED_BALANCE_TUBES[truth.scenario],
        "planned_g": round(plan.imbalance_g, 3),
        "true_g": round(exact, 3),
        "filled_max_g": round(max(filled), 3),
        "filled_ok": sum(f <= TOLERANCE_G for f in filled) / len(filled),
        "worst_g": round(plan.worst_g, 3),
        "crowded": len(plan.crowded),
        "balance_crowded": sum(t.name in plan.crowded for t in plan.balance_tubes),
        "targets_g": [round(t.grams, 2) for t in plan.balance_tubes],
        "same_as_grading": same,
        "plan": plan.describe(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(one, SEEDS))
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / "evaluation.json").write_text(json.dumps(rows, indent=1))

    by = defaultdict(list)
    for r in rows:
        by[r["job"]].append(r)
    print("| Job | Seeds | Plan found | Right number of balance tubes | Within 1.0 g (true weights) | "
          "Planned imbalance, median / max | Within 1.0 g after filling (±0.5 g) | Worst case | Tubes going in beside a full hole |")
    print("|---|---|---|---|---|---|---|---|---|")
    for job in EXPECTED_BALANCE_TUBES:
        rs = by[job]
        f = [r for r in rs if r["found"]]
        print(
            f"| {job} | {len(rs)} | {len(f)} | {sum(r['balance_tubes'] == r['expected'] for r in f)} | "
            f"{sum(r['true_g'] <= TOLERANCE_G for r in f)} | "
            f"{statistics.median(r['true_g'] for r in f):.2f} / {max(r['true_g'] for r in f):.2f} g | "
            f"{100 * statistics.mean(r['filled_ok'] for r in f):.1f} % of fills | "
            f"{max(r['worst_g'] for r in f):.2f} g | "
            f"{statistics.mean(r['crowded'] for r in f):.1f} (balance tubes: {sum(r['balance_crowded'] for r in f)}) |"
        )
    print("same numbers as the simulation's grading:", all(r.get("same_as_grading", True) for r in rows))
    print("solver time per job, ms: median", statistics.median(r["ms"] for r in rows), "max", max(r["ms"] for r in rows))
    for job in EXPECTED_BALANCE_TUBES:
        r = next(r for r in rows if r["job"] == job and r["found"])
        print(f"\nExample, seed {r['seed']} ({job}):\n{r['plan']}")


if __name__ == "__main__":
    main()
