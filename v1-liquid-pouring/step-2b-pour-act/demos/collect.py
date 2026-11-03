"""The demonstrations: the programmed pour recorded from every saved start (demos/record.py).

The starts are Step 2's (step-2-pour/data/starts): Step 1b run on scenes 100 to 159, medium and large,
saved where both tubes reach the pouring point. They are only read. Each start is poured twice:

    demo 0   as programmed
    demo 1   with the slowly wandering nudge (DART noise), so the policy also sees how to get back on course

each to its own target, drawn between 40 and 60 %, so the policy learns to stop at the level it is
asked for, not always at the same one. Scenes are split by seed, as in Step 2, so no scene is in two sets:

    train  seeds 100-139      val  seeds 140-149      test  seeds 150-159

    python -m demos.collect --workers 4
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

STARTS = Path(__file__).resolve().parents[2] / "step-2-pour" / "data" / "starts"
SPLITS = {"train": range(100, 140), "val": range(140, 150), "test": range(150, 160)}
NOISE = (0.0, 1.0)  # demo k is recorded with NOISE[k]
TARGETS = (40.0, 60.0)  # %, drawn evenly


def split_of(seed: int) -> str:
    return next(name for name, seeds in SPLITS.items() if seed in seeds)


def _job(job: tuple[Path, int, Path]) -> dict:
    from . import record

    start_path, k, out = job
    meta = json.loads(start_path.with_suffix(".json").read_text())
    seed = [meta["seed"], int(meta["asked"] * 1000), k]
    target = float(np.random.default_rng(seed).uniform(*TARGETS))
    result, ticks = record.run(start_path, target, noise=NOISE[k], seed=int(np.prod(seed) % 2**31))
    record.save(ticks, result, out / split_of(meta["seed"]) / f"{start_path.stem}_demo{k}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--starts", type=Path, default=STARTS)
    parser.add_argument("--out", type=Path, default=Path("data/demos"))
    parser.add_argument("--workers", type=int, default=4, help="at most 4 on this Mac (CLAUDE.md)")
    args = parser.parse_args()
    jobs = []
    for npy in sorted(args.starts.glob("*.npy")):
        seed = json.loads(npy.with_suffix(".json").read_text())["seed"]
        for k in range(len(NOISE)):
            if not (args.out / split_of(seed) / f"{npy.stem}_demo{k}.json").exists():
                jobs.append((npy, k, args.out))
    print(f"{len(jobs)} demonstrations to record, {args.workers} at a time", flush=True)
    start = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs), start=1):
            print(
                f"{i}/{len(jobs)} {time.time() - start:.0f} s {r['start']} noise {r['noise']} "
                f"target {r['target_pct']:.1f} -> {r['level_pct']} ({r['ticks']} ticks, "
                f"{r['wall_seconds']} s) {r['failure'] or ''}",
                flush=True,
            )


if __name__ == "__main__":
    main()
