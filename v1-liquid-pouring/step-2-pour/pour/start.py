"""Starting points for the pour: run Step 1b and save the simulation where it ends.

Step 1b ends with both arms holding their tubes at the pouring point, the source 2 cm above the
receiver's rim and beside it. Saving the whole simulation state there lets every pour start from a
real pick and carry (the tubes sit in the fingers as Step 1b left them) without running Step 1b again.

    python -m pour.start --first 100 --seeds 40 --out data/starts
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import mujoco
import numpy as np

# Medium and large only: small tubes are left out of Step 2 for now.
SIZES = (0.15, 0.2)
STATE = mujoco.mjtState.mjSTATE_INTEGRATION


def held_tubes(model: mujoco.MjModel, data: mujoco.MjData, containers: dict) -> tuple[str, str]:
    """The receiver (its rim nearest the pouring point) and the source (nearest the middle of the row)."""
    from carry.episode import POUR

    def rim(n):
        body = data.body(n)
        return body.xpos + body.xmat.reshape(3, 3)[:, 2] * containers[n].height

    receiver = min(
        (n for n in containers if n.startswith("receiver")), key=lambda n: np.linalg.norm(rim(n) - POUR)
    )
    source = min((n for n in containers if n.startswith("source")), key=lambda n: abs(data.body(n).xpos[1]))
    return receiver, source


def _job(job: tuple[int, float, Path]) -> dict:
    seed, size, out = job
    from carry import episode
    from carry.perceive import Perception
    from lab_scene import scene as lab

    model, truth = lab.build(seed)
    data = mujoco.MjData(model)
    episode.prepare(model, data)
    coordinator = episode.Coordinator(model, data, Perception(), episode.load_net())
    coordinator.containers = truth.containers
    sizes = [size] + [s for s in (0.2, 0.15, 0.1) if s != size]
    start = time.time()
    result = coordinator.run(sizes)
    meta = {
        "seed": seed,
        "asked": size,
        "size": result.get("size"),
        "done": bool(result.get("done")),
        "failure": result.get("failure"),
        "gripped_below_rim_cm": result.get("gripped_below_rim_cm"),
        "meet_distance_mm": result.get("meet_distance_mm"),
        "wall_seconds": round(time.time() - start, 1),
    }
    name = f"seed{seed}_size{int(size * 1000)}"
    if meta["done"]:
        meta["receiver"], meta["source"] = held_tubes(model, data, truth.containers)
        state = np.zeros(mujoco.mj_stateSize(model, STATE))
        mujoco.mj_getState(model, data, state, STATE)
        np.save(out / f"{name}.npy", state)
    (out / f"{name}.json").write_text(json.dumps(meta, default=str))
    return meta


def load(path: Path) -> tuple[mujoco.MjModel, mujoco.MjData, dict, dict]:
    """The model, the simulation at the end of Step 1b, the scene's containers, and the run's notes."""
    from lab_scene import scene as lab

    meta = json.loads(path.with_suffix(".json").read_text())
    model, truth = lab.build(meta["seed"])
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, np.load(path.with_suffix(".npy")), STATE)
    mujoco.mj_forward(model, data)
    return model, data, truth.containers, meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--first", type=int, default=100, help="the first scene's seed")
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    parser.add_argument("--out", type=Path, default=Path("data/starts"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [
        (seed, size, args.out)
        for seed in range(args.first, args.first + args.seeds)
        for size in SIZES
        if not (args.out / f"seed{seed}_size{int(size * 1000)}.json").exists()
    ]
    start = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, meta in enumerate(pool.imap_unordered(_job, jobs), start=1):
            print(f"{i}/{len(jobs)} {time.time() - start:.0f} s {json.dumps(meta)}", flush=True)


if __name__ == "__main__":
    main()
