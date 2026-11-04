"""The level network's pictures and labels, from the recorded pours (pour/data.py).

Each example is three pictures of the receiver 0.1 s apart (the newest last), stacked, so the network can
see movement: balls falling and the level rising. Each picture is the pour camera's 96 x 384 strip shrunk
by half to 48 x 192 (1.25 mm per pixel, still about 3 pixels for the 2 % window).

What the network is asked, for the newest picture:

    level   balls in the receiver, as % of the tube (50 % is the mark)
    flow    how fast the level rose over the last 0.3 s, % per second
    air     balls on their way (left the source, not landed yet), in the same % as the level:
            what will still land if the source tips back now

The first build reads the pours once and keeps a cache (data/level/<split>.npz), about 1 GB in all.

    python -m level.data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from lab_scene.settings import RECEIVERS
from torch.utils.data import Dataset

from pour.data import SPLITS
from pour.pour import level_table

FRAMES = 3  # pictures per example, 0.1 s apart
SHRINK = 2
FLOW_BACK = 0.3  # s
OUTPUTS = ("level", "flow", "air")
# Each output's typical size, so that the three count about equally in the loss.
SCALE = np.array([10.0, 5.0, 2.0], np.float32)


def _shrink(images: np.ndarray) -> np.ndarray:
    n, h, w, c = images.shape
    x = images.reshape(n, h // SHRINK, SHRINK, w // SHRINK, SHRINK, c).astype(np.float32).mean(axis=(2, 4))
    return np.round(x).astype(np.uint8)


def build(pours: Path, out: Path, split: str) -> Path:
    """Read every recorded pour of ``split`` once and save the shrunk pictures with their labels."""
    per_ball = {}
    for c in RECEIVERS:
        counts, levels = level_table(c)
        per_ball[round(c.height, 3)] = 50.0 / np.interp(50, levels, counts)  # % of the tube per ball
    images, labels, first, pour_id, size, phase, t_all = [], [], [], [], [], [], []
    names = []
    offset = 0
    for p in sorted((pours / split).glob("*.npz")):
        z = np.load(p)
        if len(z["t"]) == 0:  # stopped before it moved: nothing saved
            continue
        i = len(names)  # this pour's number, its place in ``names``
        meta = json.loads(p.with_suffix(".json").read_text())
        n = len(z["t"])
        t, level = z["t"], z["level"].astype(np.float32)
        flow = (level - np.interp(t - FLOW_BACK, t, level)) / FLOW_BACK
        air = z["air"] * per_ball[round(meta["size"], 3)]
        images.append(_shrink(z["image"]))
        labels.append(np.stack([level, flow, air], 1).astype(np.float32))
        first.append(np.full(n, offset))
        pour_id.append(np.full(n, i))
        size.append(np.full(n, meta["size"], np.float32))
        phase.append(z["phase"])
        t_all.append(t)
        names.append(p.stem)
        offset += n
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{split}.npz"
    np.savez(
        path,
        images=np.concatenate(images),
        labels=np.concatenate(labels),
        first=np.concatenate(first),
        pour=np.concatenate(pour_id),
        size=np.concatenate(size),
        phase=np.concatenate(phase),
        t=np.concatenate(t_all),
        names=np.array(names),
    )
    return path


class Pictures(Dataset):
    """Stacks of three pictures and their (level, flow, air) labels, from a cache made by ``build``."""

    def __init__(self, path: Path, augment: bool = False):
        z = np.load(path)
        self.images = torch.from_numpy(z["images"]).permute(0, 3, 1, 2).contiguous()  # N, 3, H, W uint8
        self.labels = z["labels"]
        self.first = z["first"]
        self.pour, self.size, self.phase, self.t = z["pour"], z["size"], z["phase"], z["t"]
        self.names = z["names"]
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def stack(self, i: int) -> torch.Tensor:
        """The picture and the two before it in the same pour (the first one repeated at the start)."""
        idx = [max(i - k, int(self.first[i])) for k in range(FRAMES - 1, -1, -1)]
        return self.images[idx].reshape(-1, *self.images.shape[2:])

    def __getitem__(self, i: int):
        x = self.stack(i).float() / 255.0
        if self.augment:
            # A little light and contrast change, and a shift sideways of up to 3 pixels. Never up or
            # down: the camera does not move, and the height in the picture is the level.
            x = (x - 0.5) * (1 + 0.2 * (torch.rand(1) - 0.5)) + 0.5 + 0.1 * (torch.rand(1) - 0.5)
            x = torch.roll(x, int(torch.randint(-3, 4, (1,))), dims=2)
        return x, torch.from_numpy(self.labels[i] / SCALE)

    def level_weights(self, step: float = 5.0) -> np.ndarray:
        """Per-example weights that make every 5 % band of level equally likely to be drawn.

        Half of all pictures show an empty receiver (moving in and tipping, before the first ball).
        """
        band = np.floor(self.labels[:, 0] / step).astype(int)
        counts = np.bincount(band)
        return 1.0 / counts[band]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pours", type=Path, default=Path("data/pours"))
    parser.add_argument("--out", type=Path, default=Path("data/level"))
    args = parser.parse_args()
    for split in SPLITS:
        path = build(args.pours, args.out, split)
        z = np.load(path)
        print(f"{split}: {len(z['labels'])} pictures from {len(z['names'])} pours -> {path}", flush=True)


if __name__ == "__main__":
    main()
