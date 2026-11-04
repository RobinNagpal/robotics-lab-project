"""Camera study, part 1: run Step 1b and save the arms' motion (every 50 ms) and where each run ends.

    PYTHONPATH=$PYTHONPATH:camera_study pixi run python camera_study/record.py --out data/camera_study/runs

Then analyse.py (which camera sees the receiver), plot.py (the map) and render.py (pictures).
See docs/solutions/step-2-pour-camera.md.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

EVERY = 50  # steps (1 ms each)


def _job(args):
    seed, size, out = args
    from carry import episode
    from carry.perceive import Perception
    from lab_scene import liquid

    frames: list[np.ndarray] = []
    keep = {}
    final = {}
    orig_step = episode.Watch.step

    def step(self):
        orig_step(self)
        if "idx" not in keep:
            m = self.model
            ball = set(liquid.ball_body_ids(m).tolist())
            idx = []
            for j in range(m.njnt):
                if m.jnt_bodyid[j] in ball:
                    continue
                a = m.jnt_qposadr[j]
                n = {0: 7, 1: 4, 2: 1, 3: 1}[m.jnt_type[j]]
                idx.extend(range(a, a + n))
            keep["idx"] = np.array(idx)
        if self.steps % EVERY == 0:
            frames.append(self.data.qpos[keep["idx"]].astype(np.float32))
        final["qpos"] = self.data.qpos
        final["qvel"] = self.data.qvel
        final["ctrl"] = self.data.ctrl
        final["time"] = self.data.time

    episode.Watch.step = step
    sizes = [size] + [s for s in (0.2, 0.15, 0.1) if s != size]
    t = time.time()
    r = episode.run_episode(seed, sizes, Perception(), episode.load_net())
    np.savez_compressed(
        out / f"seed{seed}_size{int(size * 1000)}.npz",
        frames=np.array(frames),
        idx=keep.get("idx", np.array([])),
        qpos=final["qpos"].copy(),
        qvel=final["qvel"].copy(),
        ctrl=final["ctrl"].copy(),
        time=final["time"],
    )
    keepr = {
        k: r.get(k)
        for k in (
            "seed",
            "size",
            "done",
            "failure",
            "meet_distance_mm",
            "gripped_below_rim_cm",
            "receiver_offset_mm",
            "source_offset_mm",
            "touched",
            "sizes_tried_first",
        )
    }
    keepr["asked"] = size
    keepr["wall"] = round(time.time() - t, 1)
    (out / f"seed{seed}_size{int(size * 1000)}.json").write_text(json.dumps(keepr, default=str))
    return keepr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--first", type=int, default=10)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = [(s, z, a.out) for s in range(a.first, a.first + a.seeds) for z in (0.15, 0.2)]
    t = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs), 1):
            print(i, len(jobs), round(time.time() - t), json.dumps(r, default=str), flush=True)


if __name__ == "__main__":
    main()
