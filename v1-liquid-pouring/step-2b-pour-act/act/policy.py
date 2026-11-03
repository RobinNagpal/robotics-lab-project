"""A trained learned pour, ready to drive Arm B: pictures and joints in, the next joint targets out.

Every tick (0.1 s) it is shown what the eyes see (demos/eyes.py) and gives Arm B's joint targets for
the next 10 ticks, and how sure it is that the pour has finished at each. Only the first tick's
targets are used; the next tick it is asked again, with new pictures. Asking every tick keeps the
pour quick to react, which matters most when it must tip back.

``ensemble``: ACT's "temporal ensembling" instead. Every tick's targets are then the average of all
the predictions made for that tick over the last second, the newest weighted most (by exp(-m x age
in ticks)). Smoother, but slower to react.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .data import CHUNK, EXTRA_SCALE, STRIP_FRAMES, TARGET_MID, TARGET_SPAN, Stats
from .model import make

RUNS = Path(__file__).resolve().parents[1] / "runs"


class Policy:
    def __init__(self, name: str, runs: Path = RUNS, weights: str = "best", ensemble: float | None = None):
        run = runs / name
        settings = json.loads((run / "settings.json").read_text())
        self.net = make(
            settings["model"],
            pretrained=not settings.get("no_pretrained", False),
            tune_strip=settings.get("tune_strip", False),
        )
        self.net.load_state_dict(torch.load(run / f"{weights}.pt", map_location="cpu"))
        self.net.eval()
        self.stats = Stats.load(run / "stats.json")
        self.name, self.ensemble = name, ensemble

    def start(self, target_pct: float) -> None:
        """A new pour, to ``target_pct`` % full."""
        self.target = target_pct
        self.strips: deque = deque(maxlen=STRIP_FRAMES)
        self.ahead: deque = deque(maxlen=CHUNK)  # the last CHUNK predictions, newest last
        self.extra = (0.0, 0.0, 0.0)

    @torch.no_grad()
    def act(self, seen: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
        """Arm B's joint targets for the next tick, and how sure it is that the pour has finished."""
        self.strips.append(seen["strip"])
        frames = [self.strips[0]] * (STRIP_FRAMES - len(self.strips)) + list(self.strips)
        joints = (
            np.concatenate([seen["qpos"], seen["ctrl"]]) - self.stats.joints_mean
        ) / self.stats.joints_std

        def picture(x: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(x).permute(2, 0, 1).float()[None] / 255.0

        b = {
            "strip": picture(np.concatenate(frames, axis=2)),
            "wide": picture(seen["wide"]),
            "wrist": picture(seen["wrist"]),
            "joints": torch.from_numpy(joints).float()[None],
            "target": torch.tensor([[(self.target - TARGET_MID) / TARGET_SPAN]], dtype=torch.float32),
        }
        out = self.net(b)
        change = out["actions"][0].numpy() * self.stats.action_std + self.stats.action_mean
        targets = seen["ctrl"][None] + change  # CHUNK x 6
        done = torch.sigmoid(out["done"][0]).numpy()
        self.extra = tuple((out["extra"][0].numpy() * EXTRA_SCALE).tolist())  # level, flow, air
        if self.ensemble is None:
            return targets[0], float(done[0])
        # Prediction made j ticks ago (j = 0 newest) gave this coming tick as its entry j.
        self.ahead.append((targets, done))
        n = len(self.ahead)
        ages = np.arange(n)[::-1]  # the oldest first in ``ahead``
        w = np.exp(-self.ensemble * ages)
        w /= w.sum()
        tick = sum(wi * self.ahead[i][0][ages[i]] for i, wi in enumerate(w))
        fin = sum(wi * self.ahead[i][1][ages[i]] for i, wi in enumerate(w))
        return tick, float(fin)
