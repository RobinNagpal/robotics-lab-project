"""Train a learned pour (act/model.py) on the demonstrations, for a fixed time.

    python -m act.train --model act --name act --minutes 20
    python -m act.train --model mlp --name mlp --minutes 20

Examples are drawn at random from every tick of the training demonstrations (seeds 100 to 139). Every
``--eval-every`` steps a fixed set of validation ticks (seeds 140 to 149) is scored. The weights with the
smallest validation action error are kept as runs/<name>/best.pt; the scores go to runs/<name>/results.csv.

Learning rate: warmed up over the first 300 steps, then lowered smoothly to nothing as the time runs
out (by the clock, so it ends on time whatever the speed).

The strip's eye is Step 2's level network, already trained: it is frozen (act/model.py). With
``--tune-strip`` it is trained on instead, at a tenth of the rate, to keep what it knows.
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

from .data import CHUNK, EXTRA_SCALE, Demos, Stats, ready
from .model import loss, make

THREADS = 4  # CPU threads: at most 4 cores on this Mac (CLAUDE.md)
WARMUP = 300


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def evaluate(
    net, demos: Demos, stats: Stats, idx: np.ndarray, dev: str, batch: int = 128
) -> dict[str, float]:
    """Scores on the ticks ``idx``: the action error (scaled, and in degrees per joint for the next tick),
    "finished" right, and the level read."""
    net.eval()
    errs, first, done_right, level_err, n = 0.0, 0.0, 0.0, 0.0, 0
    for i in range(0, len(idx), batch):
        j = idx[i : i + batch]
        b = ready(demos.batch(j, stats), dev)
        out = net(b)
        err = (out["actions"] - b["actions"]).abs()
        errs += err.mean().item() * len(j)
        # The next tick's targets, in degrees: what is sent to the arm.
        scale = torch.as_tensor(stats.action_std[0], device=dev)
        first += (err[:, 0] * scale).mean().item() * len(j) * 180 / math.pi
        done_right += ((out["done"] > 0) == (b["done"] > 0.5)).float().mean().item() * len(j)
        level_err += ((out["extra"][:, 0] - b["extra"][:, 0]).abs() * EXTRA_SCALE[0]).mean().item() * len(j)
        n += len(j)
    net.train()
    return {
        "val_action": errs / n,
        "val_next_deg": first / n,
        "val_done_right": done_right / n,
        "val_level_err": level_err / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=Path("data/demos"))
    parser.add_argument("--model", choices=["act", "mlp"], default="act")
    parser.add_argument("--name", required=True)
    parser.add_argument("--minutes", type=float, default=20.0)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--kl", type=float, default=10.0, help="ACT's KL weight")
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--val-ticks", type=int, default=3000)
    parser.add_argument(
        "--tune-strip", action="store_true", help="train the strip's eye on, at a tenth of the rate"
    )
    parser.add_argument("--no-pretrained", action="store_true", help="start the strip's eye from nothing")
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(THREADS)
    rng = np.random.default_rng(0)
    dev = device()
    run = Path("runs") / args.name
    run.mkdir(parents=True, exist_ok=True)
    t_load = time.time()
    train, val = Demos(args.data / "train"), Demos(args.data / "val")
    stats = Stats.of(train)
    stats.save(run / "stats.json")
    val_idx = np.sort(rng.choice(len(val), size=min(args.val_ticks, len(val)), replace=False))
    print(
        f"{len(train)} training ticks from {len(train.names)} demonstrations, {len(val)} validation ticks "
        f"from {len(val.names)}; loaded in {time.time() - t_load:.0f} s",
        flush=True,
    )

    net = make(args.model, pretrained=not args.no_pretrained, tune_strip=args.tune_strip).to(dev)
    eye = [p for p in net.strip.parameters() if p.requires_grad]
    eye_ids = {id(p) for p in net.strip.parameters()}
    rest = [p for p in net.parameters() if id(p) not in eye_ids]
    opt = torch.optim.AdamW(
        [{"params": rest, "lr": args.lr}, {"params": eye, "lr": args.lr * 0.1}], weight_decay=1e-4
    )
    base = [g["lr"] for g in opt.param_groups]
    weights = sum(p.numel() for p in net.parameters())
    trained = sum(p.numel() for p in net.parameters() if p.requires_grad)
    settings = {
        **vars(args),
        "data": str(args.data),
        "weights": weights,
        "trained_weights": trained,
        "chunk": CHUNK,
        "device": dev,
    }
    (run / "settings.json").write_text(json.dumps(settings, indent=1))
    print(f"{args.model}: {weights / 1e6:.2f} M weights, {trained / 1e6:.2f} M trained, on {dev}", flush=True)

    start = time.time()
    budget = args.minutes * 60
    best = math.inf
    step, seen = 0, 0
    running: dict[str, float] = {}
    with open(run / "results.csv", "w", newline="") as f:
        writer = None
        while True:
            step += 1
            # Warm up, then lower the rate smoothly to nothing as the time runs out.
            progress = min(1.0, (time.time() - start) / budget)
            for g, lr in zip(opt.param_groups, base, strict=True):
                g["lr"] = lr * min(1.0, step / WARMUP) * 0.5 * (1 + math.cos(math.pi * progress))
            idx = rng.integers(0, len(train), size=args.batch)
            b = ready(train.batch(idx, stats), dev, change=True)
            out = net(b, actions=b["actions"])
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
                    f"step {step}  train action {row['train_action']:.3f}  "
                    f"val action {s['val_action']:.3f}  next tick {s['val_next_deg']:.3f} deg  "
                    f"finished right {s['val_done_right']:.1%}  level {s['val_level_err']:.2f} %  "
                    f"{row['minutes']:.1f} min",
                    flush=True,
                )
            if last:
                break
    torch.save(net.state_dict(), run / "last.pt")


if __name__ == "__main__":
    main()
