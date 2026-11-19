"""Measure the programmed Step 1 on test seeds, against the simulator's answers.

For each seed: the arm weighs every sample (the cells come from the simulator, as Step 0 reports them: it
found 120 of 120 tubes with the high rack camera), then the examiner checks with the simulator's own answers:

- **weight**: what the balance read against the tube's true weight (should be within 0.05 g);
- **back in its cell**: the tube's tip within 3 mm of the cell's centre, standing within 3° of upright;
- **nothing disturbed**: every other tube, and both racks, moved less than 2 mm;
- **time** per tube (simulated) and wall-clock time.

    cd step-1-weigh
    P="pixi run --manifest-path ../simulation-environment/pixi.toml"
    PYTHONPATH=$PWD:$PWD/../simulation-environment $P python evaluate.py --seeds 20 --workers 4
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np
from centrifuge_scene import scene
from centrifuge_scene.racks import cell_centres
from centrifuge_scene.settings import LAYOUT

import test_scenes
import weigh

OUT = Path(__file__).parent / "results"
WEIGHT_OK = 0.05  # g
IN_CELL = 0.003  # m
UPRIGHT_DEG = 3.0
DISTURBED = 0.002  # m


def tilt_deg(data: mujoco.MjData, body: int) -> float:
    return math.degrees(math.acos(min(1.0, abs(float(data.xmat[body].reshape(3, 3)[2, 2])))))


def run_seed(seed: int, changed: bool = False) -> dict:
    """One seed: test scene A (as built) or, with ``changed``, test scene B (rack and balance moved)."""
    model, data, truth = test_scenes.build(seed, changed)
    samples = list(truth.samples)
    cells = [truth.cells[n] for n in samples]
    bodies = {n: model.body(n).id for n in truth.tubes} | {r: model.body(r).id for r in scene.RACKS}
    mujoco.mj_forward(model, data)
    start = {n: data.xpos[b].copy() for n, b in bodies.items()}

    results, wall = weigh.run(model, data, cells)

    rows = []
    for name, w in zip(samples, results, strict=False):
        b = bodies[name]
        cell_xy = np.array(cell_centres(getattr(LAYOUT, w.rack))[w.cell])
        off = float(np.linalg.norm(data.xpos[b][:2] - cell_xy))
        true_g = truth.samples[name].mass * 1000
        rows.append(
            {
                "tube": name,
                "rack": w.rack,
                "cell": w.cell,
                "read_g": w.grams,
                "true_g": round(true_g, 3),
                "error_g": None if w.grams is None else round(w.grams - true_g, 3),
                "steady": w.steady,
                "back_off_mm": round(off * 1000, 2),
                "back_tilt_deg": round(tilt_deg(data, b), 2),
                "back_in_cell": off < IN_CELL and tilt_deg(data, b) < UPRIGHT_DEG,
                "seconds": round(w.seconds, 1),
                "turn_rack_deg": w.turn_rack_deg,
                "turn_holder_deg": w.turn_holder_deg,
                "smallest_gap_mm": w.smallest_gap_mm,
                "error": w.error,
            }
        )
    weighed = {r["tube"] for r in rows}
    moved = {
        n: round(float(np.linalg.norm(data.xpos[b] - start[n])) * 1000, 2)
        for n, b in bodies.items()
        if n not in weighed
    }
    return {
        "seed": seed,
        "job": truth.scenario,
        "tubes": rows,
        "not_weighed": len(samples) - len(rows),
        "others_moved_mm": moved,
        "disturbed": [n for n, mm in moved.items() if mm > DISTURBED * 1000],
        "wall_seconds": round(wall, 1),
    }


def ok(row: dict) -> bool:
    return (
        not row["error"]
        and row["steady"]
        and row["error_g"] is not None
        and abs(row["error_g"]) <= WEIGHT_OK
        and row["back_in_cell"]
    )


def summarise(runs: list[dict]) -> str:
    rows = [r for run in runs for r in run["tubes"]]
    total = sum(len(r["tubes"]) + r["not_weighed"] for r in runs)
    errors = [abs(r["error_g"]) for r in rows if r["error_g"] is not None]
    lines = [
        f"seeds: {len(runs)}; sample tubes: {total}",
        f"weighings that worked (steady, within {WEIGHT_OK} g, back in its cell): "
        f"{sum(ok(r) for r in rows)} of {total}",
        f"refused moves: {sum(bool(r['error']) for r in rows)}; "
        f"not reached: {sum(r['not_weighed'] for r in runs)}",
        f"weight error, g: median {statistics.median(errors):.3f}, max {max(errors):.3f}" if errors else "",
        f"steady readings: {sum(r['steady'] for r in rows)} of {len(rows)}",
        f"back in its cell: {sum(r['back_in_cell'] for r in rows)} of {len(rows)}; "
        f"off centre max {max(r['back_off_mm'] for r in rows):.2f} mm, "
        f"tilt max {max(r['back_tilt_deg'] for r in rows):.2f} deg",
        f"seeds where another tube or a rack moved over 2 mm: {sum(bool(r['disturbed']) for r in runs)}",
        f"smallest gap planned, mm: {min(r['smallest_gap_mm'] for r in rows):.1f}",
        f"simulated time per tube, s: median {statistics.median(r['seconds'] for r in rows):.1f}",
        f"wall-clock per seed, s: median {statistics.median(r['wall_seconds'] for r in runs):.0f}",
    ]
    by_job: dict[str, list[dict]] = {}
    for run in runs:
        by_job.setdefault(run["job"], []).append(run)
    for job, js in by_job.items():
        n = sum(len(r["tubes"]) + r["not_weighed"] for r in js)
        good = sum(ok(t) for r in js for t in r["tubes"])
        lines.append(f"  {job}: {good} of {n}")
    bad = [
        (run["seed"], r["tube"], r["error"] or "a check failed")
        for run in runs
        for r in run["tubes"]
        if not ok(r)
    ]
    for b in bad:
        lines.append(f"  failed: seed {b[0]} {b[1]}: {b[2]}")
    return "\n".join(line for line in lines if line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--only", type=int, nargs="*", help="just these seeds (not saved)")
    parser.add_argument("--changed", action="store_true", help="test scenes B: rack and balance moved")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    seeds = args.only or list(range(args.first, args.first + args.seeds))
    with ProcessPoolExecutor(args.workers) as pool:
        runs = list(pool.map(run_seed, seeds, [args.changed] * len(seeds)))
    if args.only:
        print(summarise(runs))
        return
    OUT.mkdir(exist_ok=True)
    name = "evaluation_programmed_changed.json" if args.changed else "evaluation_programmed.json"
    (OUT / name).write_text(json.dumps(runs, indent=1))
    print(summarise(runs))


if __name__ == "__main__":
    main()
