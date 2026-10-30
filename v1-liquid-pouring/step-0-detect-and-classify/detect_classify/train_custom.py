"""Train the custom CNN (detect_classify/custom_net.py) on the same pictures as YOLO.

    python -m detect_classify.train_custom --name custom_rgb --epochs 40

After every epoch it checks the validation pictures: how many objects it finds
(a found object overlaps a true one by at least half) and how many detections
match nothing. The weights with the best validation F1 for tubes are kept as
runs/<name>/weights/best.pt. The scores per epoch go to runs/<name>/results.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

# OpenCV must load before PyTorch (see custom_net).
import cv2  # noqa: F401, I001
import numpy as np
import torch

from .custom_net import CLASSES, CustomNet, NetConfig, Pictures, colour_jitter, decode, loss, normalise

MATCH_IOU = 0.5


def score_picture(found: list[tuple[int, float, np.ndarray]], maps: np.ndarray) -> dict[int, list[int]]:
    """Per class: [true objects, found, extra detections], at output size."""
    out = {}
    for cls in range(len(CLASSES)):
        truth = [maps[cls] == k for k in np.unique(maps[cls]) if k]
        mine = [m for c, _, m in found if c == cls]
        used, hits = set(), 0
        for m in mine:
            best, best_j = 0.0, -1
            for j, t in enumerate(truth):
                if j in used:
                    continue
                union = (m | t).sum()
                score = (m & t).sum() / union if union else 0.0
                if score > best:
                    best, best_j = score, j
            if best >= MATCH_IOU:
                used.add(best_j)
                hits += 1
        out[cls] = [len(truth), hits, len(mine) - hits]
    return out


@torch.no_grad()
def validate(net: CustomNet, loader, device: str) -> dict[str, float]:
    net.eval()
    totals = {c: np.zeros(3, int) for c in range(len(CLASSES))}
    for x, _, maps in loader:
        out = net(normalise(x.to(device).float().div(255)))
        for b in range(x.shape[0]):
            one = {k: v[b : b + 1] for k, v in out.items()}
            for c, s in score_picture(decode(one), maps[b].numpy()).items():
                totals[c] += s
    result = {}
    for c, (true, hits, extra) in totals.items():
        name = CLASSES[c]
        recall = hits / max(true, 1)
        precision = hits / max(hits + extra, 1)
        result[f"{name}_recall"] = recall
        result[f"{name}_extra"] = int(extra)
        result[f"{name}_f1"] = 2 * recall * precision / max(recall + precision, 1e-9)
    net.train()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--limit", type=int, default=0, help="use only this many training pictures (for a quick check)"
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    out_dir = Path("runs") / args.name
    (out_dir / "weights").mkdir(parents=True, exist_ok=True)

    train_set = Pictures(args.data / "train", augment=True)
    if args.limit:
        train_set.images = train_set.images[: args.limit]
    val_set = Pictures(args.data / "val", augment=False)
    train = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        drop_last=True,
    )
    val = torch.utils.data.DataLoader(
        val_set, batch_size=4, num_workers=args.workers, persistent_workers=args.workers > 0
    )

    cfg = NetConfig()
    net = CustomNet(cfg).to(args.device)
    params = sum(p.numel() for p in net.parameters())
    print(f"custom net: {params / 1e6:.2f} M parameters, {len(train_set)} training pictures", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * len(train)
    # A short warm-up, then a cosine fall to 1 % of the starting rate.
    warm = min(300, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (
            (s + 1) / warm
            if s < warm
            else 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * (s - warm) / (steps - warm)))
        ),
    )
    (out_dir / "args.json").write_text(
        json.dumps(vars(args) | {"config": asdict(cfg)}, indent=1, default=str)
    )

    best = -1.0
    rows = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        net.train()
        sums = {"mask": 0.0, "heat": 0.0, "offset": 0.0, "total": 0.0}
        for x, t, _ in train:
            x = colour_jitter(x.to(args.device).float().div(255))
            t = {k: v.to(args.device) for k, v in t.items()}
            losses = loss(net(normalise(x)), t)
            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()
            sched.step()
            for k in sums:
                sums[k] += float(losses[k].detach()) / len(train)
        scores = validate(net, val, args.device)
        row = (
            {"epoch": epoch, "minutes": round((time.time() - start) / 60, 2)}
            | {k: round(v, 4) for k, v in sums.items()}
            | {k: round(v, 4) if isinstance(v, float) else v for k, v in scores.items()}
        )
        rows.append(row)
        print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)
        ckpt = {"model": net.state_dict(), "config": asdict(cfg), "epoch": epoch, "scores": scores}
        torch.save(ckpt, out_dir / "weights" / "last.pt")
        if scores["test_tube_f1"] > best:
            best = scores["test_tube_f1"]
            torch.save(ckpt, out_dir / "weights" / "best.pt")
        with open(out_dir / "results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"done in {(time.time() - start) / 60:.1f} min, best tube F1 on validation {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
