"""Record the scene's state during real v3 jobs, to draw the person into later (phase 2's pictures).

Runs v3's whole job (Steps 1 to 5 of ``step-6-whole-run/run.py``, unchanged) on a seed, and every 0.25 s of
simulated time saves what the pictures need: the joint positions (arm, tubes, liquid, lid), the moved caps
(mocap bodies), and which caps are shown (``geom_group``, changed when a balance tube is capped). Pictures are
then rendered from these snapshots with the person added, without running the physics again.

Step 0 is given the right answer (which cell holds which tube) instead of running SAM 3: the pictures need the
arm's moves, not Step 0's, and SAM 3 is a 3.4 GB model.

The recording wraps v3's ``scene.step`` from outside (every move of the arm goes through it); no v3 file is
changed.

    python record_jobs.py --seeds 0 1 2 3 --workers 2     # one job each, 2 to 7 min a job
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V3 = HERE.parent / "v3-centrifuge-loading"
STEPS = [
    "step-0-find",
    "step-1-weigh",
    "step-2-plan",
    "step-3-fill-balance-tube",
    "step-4-load-rotor",
    "step-5-close-and-report",
    "simulation-environment",
]
OUT = HERE / "data" / "jobs"
EVERY = 250  # simulation steps between snapshots: 0.25 s at v3's 1 ms step
COLOURS = ("red", "yellow", "green", "purple", "orange")


def _paths() -> None:
    for p in [V3 / s for s in STEPS]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def right_step0(seed: int) -> dict:
    """Step 0's answer as run.py expects it, taken from the simulator (always right)."""
    import test_scenes
    from centrifuge_scene.settings import SAMPLE_COLOURS

    _, _, truth = test_scenes.build(seed, changed=False)
    cells = {}
    for name, (rack, cell) in truth.cells.items():
        if name in truth.samples:
            colour = COLOURS[SAMPLE_COLOURS.index(truth.samples[name].colour)]
            cells[f"{rack}/{cell}"] = {"kind": "sample", "colour": colour}
        elif name.startswith("water"):
            cells[f"{rack}/{cell}"] = {"kind": "water", "colour": "blue"}
        else:
            cells[f"{rack}/{cell}"] = {"kind": "empty", "colour": "none"}
    return {"cells": cells, "right": True, "seconds": 0.0}


def record(seed: int) -> dict:
    _paths()
    from centrifuge_scene import scene

    spec = importlib.util.spec_from_file_location("v3_run", V3 / "step-6-whole-run" / "run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)

    snaps: dict[str, list] = {"qpos": [], "mocap_pos": [], "mocap_quat": [], "geom_group": [], "time": []}
    v3_step = scene.step
    count = [0]

    def step(model, data, n=1):
        v3_step(model, data, n)
        count[0] += n
        if count[0] >= EVERY:
            count[0] = 0
            snaps["qpos"].append(data.qpos.astype(np.float32))
            snaps["mocap_pos"].append(data.mocap_pos.astype(np.float32))
            snaps["mocap_quat"].append(data.mocap_quat.astype(np.float32))
            snaps["geom_group"].append(model.geom_group.astype(np.int8))
            snaps["time"].append(float(data.time))

    t0 = time.time()
    scene.step = step
    try:
        result = run.run_seed((seed, right_step0(seed)))
    finally:
        scene.step = v3_step
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"seed{seed}.npz", **{k: np.array(v) for k, v in snaps.items()})
    summary = {
        "seed": seed,
        "scenario": result.get("scenario"),
        "success": result.get("success"),
        "snapshots": len(snaps["time"]),
        "sim_seconds": round(snaps["time"][-1], 1) if snaps["time"] else 0.0,
        "wall_seconds": round(time.time() - t0, 1),
    }
    (OUT / f"seed{seed}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(record, args.seeds))
    print(f"recorded {sum(r['snapshots'] for r in results)} snapshots from {len(results)} jobs")


if __name__ == "__main__":
    main()
