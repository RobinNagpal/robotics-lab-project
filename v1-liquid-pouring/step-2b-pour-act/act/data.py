"""The policy's examples, from the recorded demonstrations (demos/collect.py).

One example is one tick (0.1 s) of one demonstration:

    seen        strip    the receiver strip now and 0.1 and 0.2 s before (as the level network sees it)
                wide     the pour camera's whole picture now, shrunk
                wrist    Arm B's wrist camera now
                joints   Arm B's six joint angles and the six joint targets it is being given (12 numbers)
                target   the level to stop at (%)

    to copy     the expert's joint targets for the next CHUNK ticks (1 s), each as a change from the
                targets being given now. Past the end of the pour, the last ones are repeated: hold still.
    done        for each of those ticks: has the pour finished (the expert holds still at the end)?
    extra       the true level, flow and balls in the air now, the level network's three numbers. Asking
                for them as well keeps the pretrained level network reading the receiver (act/model.py).

Every number the policy reads or gives is scaled by its spread over the training examples (``Stats``),
so that the six joints, and the small moves of the slow pour and the large ones of moving in, all count.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from lab_scene.settings import RECEIVERS
from pour.pour import PHASES, level_table

CHUNK = 10  # ticks of actions predicted at once: 1 s
STRIP_FRAMES = 3  # strip pictures per example, 0.1 s apart (as the level network)
FLOW_BACK = 0.3  # s: the flow is the rise of the level over this long (as the level network)
EXTRA = ("level", "flow", "air")
EXTRA_SCALE = np.array([10.0, 5.0, 2.0], np.float32)  # the level network's scales
DONE = PHASES.index("done")
TARGET_MID, TARGET_SPAN = 50.0, 10.0  # the target is given to the policy as (target - 50) / 10


def _per_ball() -> dict[float, float]:
    """% of the tube one ball is, by tube height."""
    out = {}
    for c in RECEIVERS:
        counts, levels = level_table(c)
        out[round(c.height, 3)] = 50.0 / float(np.interp(50, levels, counts))
    return out


@dataclass
class Stats:
    """Means and spreads of the joints (12) and of the actions (CHUNK x 6), from the training examples."""

    joints_mean: np.ndarray
    joints_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    @classmethod
    def of(cls, demos: Demos) -> Stats:
        joints = np.concatenate([demos.qpos, demos.ctrl], 1)
        actions = demos.actions(np.arange(len(demos)))
        return cls(
            joints.mean(0),
            np.maximum(joints.std(0), 1e-3),
            actions.mean(0),
            np.maximum(actions.std(0), 1e-4),
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({k: v.tolist() for k, v in vars(self).items()}))

    @classmethod
    def load(cls, path: Path) -> Stats:
        return cls(**{k: np.array(v, np.float32) for k, v in json.loads(path.read_text()).items()})


class Demos:
    """Every tick of every demonstration in one folder (one split), kept in memory."""

    def __init__(self, folder: Path, noise: tuple[float, ...] | None = None):
        per_ball = _per_ball()
        parts: dict[str, list] = {
            k: []
            for k in [
                "strip",
                "wide",
                "wrist",
                "qpos",
                "ctrl",
                "intent",
                "phase",
                "extra",
                "first",
                "last",
                "demo",
                "target",
                "size",
                "t",
            ]
        }
        self.names, self.results = [], []
        offset = 0
        for p in sorted(folder.glob("*.npz")):
            result = json.loads(p.with_suffix(".json").read_text())
            if noise is not None and result["noise"] not in noise:
                continue
            if result["failure"]:  # stopped part way (Arm B could not reach): not a pour to copy
                continue
            z = np.load(p)
            n = len(z["t"])
            if n == 0:  # stopped before it moved
                continue
            t, level = z["t"], z["level"]
            flow = (level - np.interp(t - FLOW_BACK, t, level)) / FLOW_BACK
            air = z["air"] * per_ball[round(result["size"], 3)]
            for k in ("strip", "wide", "wrist", "qpos", "ctrl", "intent", "phase", "t"):
                parts[k].append(z[k])
            parts["extra"].append(np.stack([level, flow, air], 1).astype(np.float32))
            parts["first"].append(np.full(n, offset))
            parts["last"].append(np.full(n, offset + n - 1))
            parts["demo"].append(np.full(n, len(self.names)))
            parts["target"].append(np.full(n, result["target_pct"], np.float32))
            parts["size"].append(np.full(n, result["size"], np.float32))
            self.names.append(p.stem)
            self.results.append(result)
            offset += n
        if not self.names:
            raise FileNotFoundError(f"no demonstrations in {folder}")
        cat = {k: np.concatenate(v) for k, v in parts.items()}
        # Pictures as (N, 3, H, W) uint8 tensors, to be cut into batches on the GPU's side.
        for k in ("strip", "wide", "wrist"):
            setattr(self, k, torch.from_numpy(cat[k]).permute(0, 3, 1, 2).contiguous())
        for k in ("qpos", "ctrl", "intent", "extra", "target", "size", "t"):
            setattr(self, k, cat[k].astype(np.float32))
        for k in ("phase", "first", "last", "demo"):
            setattr(self, k, cat[k].astype(np.int64))

    def __len__(self) -> int:
        return len(self.t)

    def future(self, idx: np.ndarray) -> np.ndarray:
        """(len(idx), CHUNK): the ticks 1 to CHUNK after each, held at the demonstration's last tick."""
        ahead = idx[:, None] + np.arange(1, CHUNK + 1)[None]
        return np.minimum(ahead, self.last[idx][:, None])

    def actions(self, idx: np.ndarray) -> np.ndarray:
        """(len(idx), CHUNK, 6): the expert's joint targets for the next ticks, less the targets now."""
        return self.intent[self.future(idx)] - self.ctrl[idx][:, None, :]

    def done(self, idx: np.ndarray) -> np.ndarray:
        """(len(idx), CHUNK): 1 where the pour has finished at that tick (or the demonstration has ended)."""
        ahead = idx[:, None] + np.arange(1, CHUNK + 1)[None]
        past_end = ahead > self.last[idx][:, None]
        return ((self.phase[self.future(idx)] == DONE) | past_end).astype(np.float32)

    def strip_stack(self, idx: np.ndarray) -> torch.Tensor:
        """(len(idx), 9, 192, 48) uint8: the strip 0.2 s before, 0.1 s before and now (oldest first)."""
        back = np.arange(STRIP_FRAMES - 1, -1, -1)
        frames = np.maximum(idx[:, None] - back[None], self.first[idx][:, None])
        x = self.strip[torch.from_numpy(frames.reshape(-1))]
        return x.reshape(len(idx), STRIP_FRAMES * 3, *x.shape[2:])

    def batch(self, idx: np.ndarray, stats: Stats) -> dict[str, torch.Tensor]:
        """Everything for the examples ``idx``, scaled; the pictures still as bytes (see ``ready``)."""
        joints = np.concatenate([self.qpos[idx], self.ctrl[idx]], 1)
        idx_t = torch.from_numpy(idx)
        return {
            "strip": self.strip_stack(idx),
            "wide": self.wide[idx_t],
            "wrist": self.wrist[idx_t],
            "joints": torch.from_numpy((joints - stats.joints_mean) / stats.joints_std).float(),
            "target": torch.from_numpy((self.target[idx, None] - TARGET_MID) / TARGET_SPAN).float(),
            "actions": torch.from_numpy((self.actions(idx) - stats.action_mean) / stats.action_std).float(),
            "done": torch.from_numpy(self.done(idx)),
            "extra": torch.from_numpy(self.extra[idx] / EXTRA_SCALE).float(),
        }


PICTURES = ("strip", "wide", "wrist")


def ready(b: dict[str, torch.Tensor], device: str, change: bool = False) -> dict[str, torch.Tensor]:
    """Move a batch to ``device``, the pictures as 0..1, changed a little (``augment``) if ``change``."""
    out = {k: v.to(device) for k, v in b.items()}
    for k in PICTURES:
        out[k] = out[k].float() / 255.0
    return augment(out) if change else out


def augment(b: dict[str, torch.Tensor], gen: torch.Generator | None = None) -> dict[str, torch.Tensor]:
    """Small changes to the pictures, so the policy does not lean on exact pixel values.

    Light and contrast change a little in every picture. The wide and wrist pictures shift by up to
    3 pixels each way. The strip shifts only sideways, never up or down: the camera does not move,
    and the height in the strip is the level (as for the level network).
    """
    out = dict(b)
    n = b["strip"].shape[0]
    for k in PICTURES:
        x = b[k]
        contrast = 1 + 0.2 * (torch.rand(n, 1, 1, 1, generator=gen).to(x.device) - 0.5)
        light = 0.1 * (torch.rand(n, 1, 1, 1, generator=gen).to(x.device) - 0.5)
        x = ((x - 0.5) * contrast + 0.5 + light).clamp(0, 1)
        dx = int(torch.randint(-3, 4, (1,), generator=gen))
        dy = 0 if k == "strip" else int(torch.randint(-3, 4, (1,), generator=gen))
        x = _shift(x, dy, dx)
        out[k] = x
    return out


def _shift(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """Shift the pictures, filling the uncovered edge with the edge's own pixels."""
    if dy == 0 and dx == 0:
        return x
    pad = (max(dx, 0), max(-dx, 0), max(dy, 0), max(-dy, 0))
    x = torch.nn.functional.pad(x, pad, mode="replicate")
    h, w = x.shape[2] - abs(dy), x.shape[3] - abs(dx)
    y0, x0 = max(-dy, 0), max(-dx, 0)
    return x[:, :, y0 : y0 + h, x0 : x0 + w]
