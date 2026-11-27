"""Convert the recorded episodes (``../data/<split>/episode_NNNN``) to LeRobot's dataset format.

LeRobot's format (v3.0) is what SmolVLA, π0.5 and GR00T all train from: the pictures as MP4 videos (one per
camera), the numbers as Parquet tables, and the sentence ("task") of each episode. One dataset per split:

    outputs/datasets/step1_train        120 episodes
    outputs/datasets/step1_validation   12 episodes

Features of every frame (10 per second):
- ``observation.images.overhead``, ``observation.images.racks_high``, ``observation.images.wrist``: 192 x 256 x 3
- ``observation.state``: 6 joint angles (rad) + gripper (0 open, 1 closed)
- ``action``: the teacher's 6 joint targets for the next 0.1 s + gripper command (0 open, 1 closed)

    cd step-1-weigh/learn
    pixi run python convert.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).parent
DATA = HERE.parent / "data"
OUT = HERE / "outputs" / "datasets"
CAMERAS = ("overhead", "racks_high", "wrist")
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
FPS = 10
# The video encoder would otherwise use every CPU core; the cameras are encoded one after another, 4 threads.
ENCODER_THREADS = 4


def features(height: int, width: int) -> dict:
    names = [*JOINTS, "gripper"]
    out = {
        "observation.state": {"dtype": "float32", "shape": (7,), "names": names},
        "action": {"dtype": "float32", "shape": (7,), "names": names},
    }
    for cam in CAMERAS:
        out[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }
    return out


def convert(split: str, limit: int | None = None) -> Path:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    episodes = sorted((DATA / split).glob("episode_*"))[:limit]
    first = np.asarray(Image.open(episodes[0] / CAMERAS[0] / "0000.jpg"))
    root = OUT / f"step1_{split}"
    if root.exists():
        shutil.rmtree(root)
    ds = LeRobotDataset.create(
        repo_id=f"local/step1_{split}",
        fps=FPS,
        features=features(*first.shape[:2]),
        root=root,
        robot_type="ur5e_robotiq_2f85",
        use_videos=True,
        encoder_threads=ENCODER_THREADS,
    )
    t0 = time.time()
    for i, ep in enumerate(episodes):
        meta = json.loads((ep / "meta.json").read_text())
        arrays = np.load(ep / "data.npz")
        for f in range(meta["frames"]):
            frame = {
                "observation.state": arrays["state"][f].astype(np.float32),
                "action": arrays["action"][f].astype(np.float32),
                "task": meta["task"],
            }
            for cam in CAMERAS:
                frame[f"observation.images.{cam}"] = np.asarray(Image.open(ep / cam / f"{f:04d}.jpg"))
            ds.add_frame(frame)
        ds.save_episode(parallel_encoding=False)
        print(f"{split}: {i + 1}/{len(episodes)} episodes ({time.time() - t0:.0f} s)", flush=True)
    ds.finalize()
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="*", default=["train", "validation"])
    parser.add_argument("--limit", type=int, help="only the first N episodes (a trial)")
    args = parser.parse_args()
    for split in args.splits:
        root = convert(split, args.limit)
        size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) / 1e6
        print(f"{split}: {root} ({size:.0f} MB)")


if __name__ == "__main__":
    main()
