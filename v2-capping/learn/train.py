"""Train ACT (learn/model.py) on the recorded unscrewing runs, for a fixed time.

    python -m learn.train --data runs/demos --name act_A --minutes 20

Examples are drawn at random from every sample of the training runs (scenes 300-469), with sensor noise
added each time. Every ``--eval-every`` steps a fixed set of validation samples (scenes 470-499, with
fixed noise) is scored; the weights with the smallest validation action error are kept as
runs/learn/<name>/best.pt, the scores in results.csv.

Learning rate: warmed up over the first 300 steps, then lowered smoothly to nothing as the time runs out
(by the clock, so it ends on time whatever the speed).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .data import CHUNK, HISTORY, STRIDE, Demos, Stats, split
from .model import ACT, loss

THREADS = 4  # CPU threads: at most 4 cores on this Mac (CLAUDE.md)
WARMUP = 300


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def to(b: dict[str, np.ndarray], dev: str) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v).to(dev) for k, v in b.items()}


@torch.no_grad()
def evaluate(net, val: Demos, stats: Stats, idx: np.ndarray, dev: str, batch: int = 256) -> dict:
    """The action error (scaled; and the next sample's joint targets in degrees, gripper command in its
    own units) and "done" right, on the samples ``idx``, with fixed noise."""
    net.eval()
    rng = np.random.default_rng(1)
    err = joint_deg = grip = done_right = 0.0
    for i in range(0, len(idx), batch):
        j = idx[i : i + batch]
        b = to(val.batch(j, stats, rng), dev)
        out = net(b["obs"])
        e = (out["actions"] - b["actions"]).abs()
        err += e.mean().item() * len(j)
        scale = torch.as_tensor(stats.act_std, device=dev)
        first = e[:, 0] * scale
        joint_deg += first[:, :6].mean().item() * len(j) * 180 / math.pi
        grip += first[:, 6].mean().item() * len(j)
        done_right += ((out["done"] > 0) == (b["done"] > 0.5)).float().mean().item() * len(j)
    net.train()
    n = len(idx)
    return {
        "val_action": err / n,
        "val_next_joint_deg": joint_deg / n,
        "val_next_grip": grip / n,
        "val_done_right": done_right / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=Path("runs/demos"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--minutes", type=float, default=20.0)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--kl", type=float, default=10.0, help="ACT's KL weight")
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--val-samples", type=int, default=6000)
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(THREADS)
    rng = np.random.default_rng(0)
    dev = device()
    run = Path("runs/learn") / args.name
    run.mkdir(parents=True, exist_ok=True)
    t_load = time.time()
    train, val = split(args.data)
    stats = Stats.of(train)
    stats.save(run / "stats.json")
    val_idx = np.sort(rng.choice(len(val), size=min(args.val_samples, len(val)), replace=False))
    print(
        f"{len(train)} training samples from {len(train.names)} runs, {len(val)} validation samples from "
        f"{len(val.names)} runs; loaded in {time.time() - t_load:.0f} s",
        flush=True,
    )

    net = ACT().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    weights = sum(p.numel() for p in net.parameters())
    settings = {
        **vars(args),
        "data": str(args.data),
        "weights": weights,
        "chunk": CHUNK,
        "history": HISTORY,
        "stride": STRIDE,
        "device": dev,
        "train_runs": train.names,
        "val_runs": val.names,
    }
    (run / "settings.json").write_text(json.dumps(settings, indent=1))
    print(f"ACT: {weights / 1e6:.2f} M weights, on {dev}", flush=True)

    start = time.time()
    budget = args.minutes * 60
    best = math.inf
    step, seen = 0, 0
    running: dict[str, float] = {}
    with open(run / "results.csv", "w", newline="") as f:
        writer = None
        while True:
            step += 1
            progress = min(1.0, (time.time() - start) / budget)
            for g in opt.param_groups:
                g["lr"] = args.lr * min(1.0, step / WARMUP) * 0.5 * (1 + math.cos(math.pi * progress))
            b = to(train.batch(rng.integers(0, len(train), size=args.batch), stats, rng), dev)
            out = net(b["obs"], actions=b["actions"])
            value, parts = loss(out, b, kl_weight=args.kl)
            opt.zero_grad()
            value.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + v
            seen += 1
            last = time.time() - start >= budget
            if step % args.eval_every == 0 or last:
                s = evaluate(net, val, stats, val_idx, dev)
                row = {
                    "step": step,
                    **{f"train_{k}": v / seen for k, v in running.items()},
                    **s,
                    "minutes": (time.time() - start) / 60,
                }
                running, seen = {}, 0
                if s["val_action"] < best:
                    best = s["val_action"]
                    torch.save(net.state_dict(), run / "best.pt")
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                f.flush()
                print(
                    f"step {step}  train action {row['train_action']:.3f}  val action {s['val_action']:.3f}  "
                    f"next joints {s['val_next_joint_deg']:.3f} deg  grip {s['val_next_grip']:.2f}  "
                    f"done right {s['val_done_right']:.1%}  {row['minutes']:.1f} min",
                    flush=True,
                )
            if last:
                break
    torch.save(net.state_dict(), run / "last.pt")


if __name__ == "__main__":
    main()
