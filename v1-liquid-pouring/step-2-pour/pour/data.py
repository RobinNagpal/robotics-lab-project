"""The level network's training pictures: programmed pours from every saved start, with labels.

Each start (pour.start) is poured more than once, each time to a different target and at different
tipping speeds, so the pictures cover levels from empty to about 65 % and many flows. Scenes are
split by seed, so no scene is in two sets:

    train  seeds 100-139      val  seeds 140-149      test  seeds 150-159

Each pour is saved as one .npz: the receiver's pictures (every 0.1 s) and, for each picture, the
time, balls in the receiver / source / air / spilled, the level (%), the tilt and the phase.

    python -m pour.data --starts data/starts --out data/pours
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

SPLITS = {"train": range(100, 140), "val": range(140, 150), "test": range(150, 160)}
POURS_PER_START = 2
TARGETS = (35.0, 65.0)  # %, drawn evenly
FIND_RATES = (1.5, 3.0)  # degrees per second
POUR_RATES = (0.5, 1.2)


def split_of(seed: int) -> str:
    return next(name for name, seeds in SPLITS.items() if seed in seeds)


def _job(job: tuple[Path, int, Path]) -> dict:
    from . import pour

    start_path, k, out = job
    meta = json.loads(start_path.with_suffix(".json").read_text())
    rng = np.random.default_rng([meta["seed"], int(meta["asked"] * 1000), k])
    target = float(rng.uniform(*TARGETS))
    find_rate = float(rng.uniform(*FIND_RATES))
    pour_rate = float(rng.uniform(*POUR_RATES))
    result, frames = pour.run(start_path, target, find_rate=find_rate, pour_rate=pour_rate)
    folder = out / split_of(meta["seed"])
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{start_path.stem}_pour{k}"
    np.savez_compressed(
        folder / f"{name}.npz",
        **{key: np.array(v) for key, v in frames.items()},
    )
    result["frames"] = len(frames["t"])
    (folder / f"{name}.json").write_text(json.dumps(result, default=str))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--starts", type=Path, default=Path("data/starts"))
    parser.add_argument("--out", type=Path, default=Path("data/pours"))
    parser.add_argument("--pours", type=int, default=POURS_PER_START)
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    args = parser.parse_args()
    jobs = []
    for npy in sorted(args.starts.glob("*.npy")):
        seed = json.loads(npy.with_suffix(".json").read_text())["seed"]
        for k in range(args.pours):
            if not (args.out / split_of(seed) / f"{npy.stem}_pour{k}.json").exists():
                jobs.append((npy, k, args.out))
    start = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs), start=1):
            print(
                f"{i}/{len(jobs)} {time.time() - start:.0f} s {r['start']} target {r['target_pct']:.1f} -> "
                f"{r['level_pct']} ({r['frames']} pictures, {r['wall_seconds']} s) {r['failure'] or ''}",
                flush=True,
            )


if __name__ == "__main__":
    main()
