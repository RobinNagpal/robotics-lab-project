"""The level network as the pour's eyes: shown the pour camera's picture every 0.1 s, it reads the receiver.

The pour (pour.pour.run, ``reader=``) shows it each new picture of the receiver (the 96 x 384 strip) and
asks for the level, the flow and the balls in the air, the three numbers the network was trained to
give. The pour works in balls, as it does with the true count, so the readings are turned into balls
with the receiver's own table (pour.pour.level_table): only its size is needed, which Step 0.1 gives.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from pour.pour import level_table

from .data import FRAMES, SCALE, _shrink
from .net import make

RUNS = Path(__file__).resolve().parents[1] / "runs"
SETTLED = 5  # readings (0.5 s) averaged for the settled level


class NetReader:
    def __init__(self, name: str = "custom", runs: Path = RUNS):
        settings = json.loads((runs / name / "settings.json").read_text())
        self.net = make(settings["model"])
        self.net.load_state_dict(torch.load(runs / name / "best.pt", map_location="cpu"))
        self.net.eval()
        self.name = name

    def start(self, container) -> None:
        """A new pour, into ``container``."""
        self.counts, self.levels = level_table(container)
        self.per_ball = 50.0 / float(np.interp(50, self.levels, self.counts))  # % of the tube per ball
        self.pictures: deque = deque(maxlen=FRAMES)
        self.history: list[tuple[float, float, float]] = []
        self.last = (0.0, 0.0, 0.0)

    def see(self, crop: np.ndarray) -> None:
        """A new picture of the receiver (the pour camera's 96 x 384 strip, the newest)."""
        self.pictures.append(_shrink(crop[None])[0])
        stack = list(self.pictures)
        stack = [stack[0]] * (FRAMES - len(stack)) + stack  # the first repeated at the start, as in training
        x = torch.from_numpy(np.concatenate(stack, axis=2)).permute(2, 0, 1).float()[None] / 255.0
        with torch.no_grad():
            level, flow, air = (self.net(x)[0].numpy() * SCALE).tolist()
        self.last = (level, flow, max(air, 0.0))
        self.history.append(self.last)

    @property
    def level(self) -> float:
        return self.last[0]

    def to_balls(self, level: float) -> float:
        return float(np.interp(level, self.levels, self.counts))

    @property
    def balls(self) -> float:
        return self.to_balls(self.last[0])

    @property
    def flow_balls(self) -> float:
        return self.last[1] / self.per_ball

    @property
    def air_balls(self) -> float:
        return self.last[2] / self.per_ball

    def settled_level(self) -> float:
        recent = [h[0] for h in self.history[-SETTLED:]] or [0.0]
        return float(np.mean(recent))

    def settled_balls(self) -> float:
        return self.to_balls(self.settled_level())
