"""Train the grasp network (grasp/net.py) on the close-up pictures.

    python -m grasp.train --name grconv --epochs 16

After every epoch it checks the validation pictures the way the grading does
(evaluate.py), without physics: for each target, does the model's choice come
down clear, or does it rightly say there is no clear turn. The weights with
the best validation score are kept as runs/<name>/best.pt; the scores per
epoch go to runs/<name>/results.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from detect_classify.camera import Calibration
from torch.utils.data import DataLoader

from . import pick, rules, side
from .net import GraspNet, Pictures, loss


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def turn_accuracy(scores: np.ndarray, samples: list[dict]) -> dict[str, float]:
    """Without the simulator: is the chosen turn one the labels call clear, near the right pixel?

    A quick stand-in for the grading, good enough to choose the best epoch.
    """
    right = clear_targets = refused_right = blocked_targets = 0
    for s, sc in zip(samples, scores, strict=True):
        calib = Calibration.from_dict(s["camera"])
        target = next(t for t in s["tubes"] if t["name"] == s["target"])
        grasp, (u, v, k) = pick.learned(sc, np.zeros(sc.shape[1:]), calib)
        if any(target["labels"]):
            clear_targets += 1
            near = (u - target["u"]) ** 2 + (v - target["v"]) ** 2 <= 3**2
            right += int(grasp is not None and near and target["labels"][k])
        else:
            blocked_targets += 1
            refused_right += int(grasp is None)
    return {
        "clear_turn_found": right / max(clear_targets, 1),
        "blocked_refused": refused_right / max(blocked_targets, 1),
        "right": (right + refused_right) / max(len(samples), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=None, help="default: data/, data_side/ or data_side2/")
    parser.add_argument("--kind", choices=["top", "side", "side2"], default="top")
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(0)
    dev = device()
    run = Path("runs") / args.name
    run.mkdir(parents=True, exist_ok=True)
    args.data = args.data or Path("data" if args.kind == "top" else f"data_{args.kind}")
    bins, step = (rules.BINS, rules.BIN_STEP) if args.kind == "top" else (side.BINS, side.STEP)
    train = Pictures(args.data / "train", augment=True, bins=bins, step=step)
    val = Pictures(args.data / "val", bins=bins, step=step)
    loader = DataLoader(
        train, batch_size=args.batch, shuffle=True, num_workers=args.workers, persistent_workers=True
    )
    val_loader = DataLoader(val, batch_size=64, num_workers=args.workers)
    net = GraspNet(bins).to(dev)
    params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(loader), pct_start=0.1
    )
    print(f"{len(train)} training and {len(val)} validation pictures; {params / 1e6:.2f} M weights; on {dev}")
    (run / "settings.json").write_text(json.dumps({**vars(args), "data": str(args.data), "weights": params}))

    best, start = -1.0, time.time()
    with open(run / "results.csv", "w", newline="") as f:
        writer = None
        for epoch in range(1, args.epochs + 1):
            net.train()
            total, n = 0.0, 0
            for x, target, weight in loader:
                x, target, weight = x.to(dev), target.to(dev), weight.to(dev)
                value = loss(net(x), target, weight)
                opt.zero_grad()
                value.backward()
                opt.step()
                sched.step()
                total, n = total + value.item(), n + 1
            net.eval()
            val_loss, scores = 0.0, []
            with torch.no_grad():
                for x, target, weight in val_loader:
                    out = net(x.to(dev))
                    val_loss += loss(out, target.to(dev), weight.to(dev)).item() * x.shape[0]
                    scores.append(torch.sigmoid(out).cpu().numpy())
            acc = turn_accuracy(np.concatenate(scores), val.samples)
            row = {
                "epoch": epoch,
                "train_loss": total / n,
                "val_loss": val_loss / len(val),
                **acc,
                "minutes": (time.time() - start) / 60,
            }
            # Keep the epoch with the most targets answered right, of either kind.
            if acc["right"] > best:
                best = acc["right"]
                torch.save(net.state_dict(), run / "best.pt")
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            print(
                f"epoch {epoch:2d}  loss {row['train_loss']:.4f} / {row['val_loss']:.4f}  "
                f"clear turn found {acc['clear_turn_found']:.1%}  "
                f"blocked refused {acc['blocked_refused']:.1%}  "
                f"{row['minutes']:.1f} min",
                flush=True,
            )
    torch.save(net.state_dict(), run / "last.pt")


if __name__ == "__main__":
    main()
