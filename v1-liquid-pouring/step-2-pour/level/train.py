"""Train a level network (level/net.py) on the recorded pours.

    python -m level.train --model custom --name custom --minutes 15

Training draws pictures evenly across levels (level.data.Pictures.level_weights). After every epoch the
validation pictures are scored as they come (not evened out); the weights with the smallest level error
where the pour is decided (20 to 65 % full) are kept as runs/<name>/best.pt, and the scores per epoch go
to runs/<name>/results.csv. It stops after the last epoch that fits in ``--minutes``.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data import OUTPUTS, SCALE, Pictures
from .net import make

THREADS = 4  # CPU threads: at most 4 cores on this Mac (CLAUDE.md)


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def predict(net: torch.nn.Module, data: Pictures, dev: str, batch: int = 256) -> np.ndarray:
    """(level, flow, air) for every picture, in % and % per second."""
    net.eval()
    out = []
    loader = DataLoader(data, batch_size=batch, shuffle=False)
    with torch.no_grad():
        for x, _ in loader:
            out.append(net(x.to(dev)).cpu().numpy())
    return np.concatenate(out) * SCALE


def scores(pred: np.ndarray, data: Pictures) -> dict[str, float]:
    truth = data.labels
    err = np.abs(pred - truth)
    band = (truth[:, 0] >= 20) & (truth[:, 0] <= 65)
    flowing = truth[:, 1] > 2
    return {
        "level_err_band": float(err[band, 0].mean()),
        "level_err_all": float(err[:, 0].mean()),
        "level_within_2_band": float((err[band, 0] <= 2).mean()),
        "flow_err_flowing": float(err[flowing, 1].mean()),
        "air_err": float(err[:, 2].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=Path("data/level"))
    parser.add_argument("--model", choices=["custom", "resnet18"], default="custom")
    parser.add_argument("--name", required=True)
    parser.add_argument("--minutes", type=float, default=15.0, help="stop after the epoch that passes this")
    parser.add_argument("--epochs", type=int, default=100, help="at most this many")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(THREADS)
    dev = device()
    run = Path("runs") / args.name
    run.mkdir(parents=True, exist_ok=True)
    train = Pictures(args.data / "train.npz", augment=True)
    val = Pictures(args.data / "val.npz")
    sampler = WeightedRandomSampler(train.level_weights(), num_samples=len(train), replacement=True)
    loader = DataLoader(train, batch_size=args.batch, sampler=sampler)
    net = make(args.model).to(dev)
    params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    print(f"{len(train)} training and {len(val)} validation pictures; {params / 1e6:.2f} M weights; on {dev}")
    (run / "settings.json").write_text(
        json.dumps({**vars(args), "data": str(args.data), "weights": params, "outputs": OUTPUTS})
    )

    best, start = np.inf, time.time()
    sched = None
    with open(run / "results.csv", "w", newline="") as f:
        writer = None
        for epoch in range(1, args.epochs + 1):
            net.train()
            total, n = 0.0, 0
            for x, target in loader:
                x, target = x.to(dev), target.to(dev)
                value = torch.nn.functional.smooth_l1_loss(net(x), target)
                opt.zero_grad()
                value.backward()
                opt.step()
                if sched is not None:
                    sched.step()
                total, n = total + value.item(), n + 1
            minutes = (time.time() - start) / 60
            if sched is None:
                # Now the time per epoch is known: plan the epochs that fit, and lower the learning
                # rate smoothly to nothing over them.
                planned = max(2, min(args.epochs, int(args.minutes / minutes)))
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=(planned - 1) * len(loader))
            s = scores(predict(net, val, dev), val)
            row = {"epoch": epoch, "train_loss": total / n, **s, "minutes": (time.time() - start) / 60}
            if s["level_err_band"] < best:
                best = s["level_err_band"]
                torch.save(net.state_dict(), run / "best.pt")
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            print(
                f"epoch {epoch:2d}/{planned}  loss {row['train_loss']:.4f}  "
                f"level error {s['level_err_band']:.2f} "
                f"(20-65 %), {s['level_err_all']:.2f} (all)  within 2 %: {s['level_within_2_band']:.1%}  "
                f"flow {s['flow_err_flowing']:.2f}  air {s['air_err']:.2f}  {row['minutes']:.1f} min",
                flush=True,
            )
            if epoch >= planned:
                break
    torch.save(net.state_dict(), run / "last.pt")


if __name__ == "__main__":
    main()
