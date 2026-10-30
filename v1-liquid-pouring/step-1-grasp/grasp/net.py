"""The grasp network: a GR-ConvNet-style model that scores every pixel at every turn.

GR-ConvNet (Kumra et al., 2020) takes a picture from above and, for every
pixel, says how good a grasp centred there is, at what angle, and how wide.
Ours keeps its shape (shrink, residual blocks, grow back to full size) and
changes the output: instead of one angle per pixel it gives a score for each
of the 18 turns. A tube often has several clear turns, and one angle per
pixel would have to average them into a wrong one. The width is left out:
every tube is 25 mm, so the squeeze is worked out (rules.py).

    height picture (1 x 128 x 128)
      -> 9x9 conv, 32        128 x 128
      -> 4x4 conv /2, 64      64 x 64
      -> 4x4 conv /2, 128     32 x 32
      -> 5 residual blocks    32 x 32
      -> 4x4 up-conv x2, 64   64 x 64
      -> 4x4 up-conv x2, 32  128 x 128
      -> 3x3 conv, 18        scores: one per turn, per pixel
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from . import rules

# Heights are divided by this, so the tallest rims are near 1.
HEIGHT_SCALE = 0.25
# A label is a small bump, this wide (pixels), at the centre of each tube's rim.
LABEL_SIGMA = 1.5
# Pixels this close to a tube's centre count this much more in the loss: they
# are where the answer is read.
NEAR = 4
NEAR_WEIGHT = 20.0


def _block(cin: int, cout: int, k: int, stride: int = 1, up: bool = False) -> nn.Sequential:
    conv = nn.ConvTranspose2d if up else nn.Conv2d
    pad = (k - stride) // 2 if up else k // 2 - (1 if stride == 2 else 0)
    return nn.Sequential(conv(cin, cout, k, stride, pad), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Residual(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.body(x))


class GraspNet(nn.Module):
    def __init__(self, bins: int = rules.BINS, width: int = 32, blocks: int = 5):
        super().__init__()
        self.down = nn.Sequential(
            _block(1, width, 9),
            _block(width, 2 * width, 4, 2),
            _block(2 * width, 4 * width, 4, 2),
        )
        self.middle = nn.Sequential(*[Residual(4 * width) for _ in range(blocks)])
        self.up = nn.Sequential(
            _block(4 * width, 2 * width, 4, 2, up=True),
            _block(2 * width, width, 4, 2, up=True),
        )
        self.head = nn.Conv2d(width, bins, 3, padding=1)

    def forward(self, heights: torch.Tensor) -> torch.Tensor:
        """Scores before the sigmoid, (B, 18, H, W), for heights (B, 1, H, W) in metres."""
        return self.head(self.up(self.middle(self.down(heights / HEIGHT_SCALE))))


def label_maps(tubes: list[dict], size: int, bins: int = rules.BINS) -> tuple[np.ndarray, np.ndarray]:
    """The target scores (bins, size, size) and the loss weights (size, size) for one picture."""
    target = np.zeros((bins, size, size), np.float32)
    weight = np.ones((size, size), np.float32)
    v, u = np.mgrid[0:size, 0:size]
    for t in tubes:
        d2 = (u - t["u"]) ** 2 + (v - t["v"]) ** 2
        bump = np.exp(-d2 / (2 * LABEL_SIGMA**2))
        on = np.array(t["labels"], bool)
        target[on] = np.maximum(target[on], bump)
        weight[d2 <= NEAR**2] = NEAR_WEIGHT
    return target, weight


class Pictures(Dataset):
    """One split's height pictures with their label maps, turned and mirrored at random when training."""

    def __init__(
        self, folder: Path, augment: bool = False, bins: int = rules.BINS, step: float = rules.BIN_STEP
    ):
        self.heights = np.load(folder / "heights.npy")
        self.samples = json.loads((folder / "samples.json").read_text())
        self.augment = augment
        self.bins, self.step = bins, step
        self.rng = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        h = self.heights[i].astype(np.float32)
        target, weight = label_maps(self.samples[i]["tubes"], h.shape[0], self.bins)
        if self.augment:
            # Turning or mirroring the picture turns or mirrors every grasp in it.
            k = int(self.rng.integers(4))
            h, weight = np.rot90(h, k), np.rot90(weight, k)
            target = rules.rotate_bins(np.rot90(target, k, axes=(1, 2)), k, step=self.step)
            if self.rng.random() < 0.5:
                h, weight = h[:, ::-1], weight[:, ::-1]
                target = rules.mirror_bins(target[:, :, ::-1], step=self.step)
            # A depth camera's small noise, about 1 mm.
            h = h + self.rng.normal(0, 0.001, h.shape).astype(np.float32)
        return (
            torch.from_numpy(np.ascontiguousarray(h))[None],
            torch.from_numpy(np.ascontiguousarray(target)),
            torch.from_numpy(np.ascontiguousarray(weight)),
        )


def loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy at every pixel and turn, weighted towards the tubes' centres."""
    per = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (per * weight[:, None]).sum() / (weight.sum() * logits.shape[1])
