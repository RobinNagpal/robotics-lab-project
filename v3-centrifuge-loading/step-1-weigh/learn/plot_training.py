"""Draw the training error ("loss") of a SmolVLA run from its log, for docs/figures/.

    cd step-1-weigh/learn
    pixi run python plot_training.py outputs/train_smolvla_2pass.log ../docs/figures/smolvla-training-loss.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_EVERY = 50  # lerobot-train's --log_freq in train.py (the log rounds steps to thousands)
INK, MUTED, SERIES, SURFACE = "#0b0b0b", "#52514e", "#2a78d6", "#fcfcfb"


def losses(log: Path) -> list[float]:
    text = log.read_text(errors="ignore").replace("\r", "\n")
    return [float(m) for m in re.findall(r"step:\S+ smpl:\S+ .*?loss:([0-9.]+)", text)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--checkpoints", type=int, nargs="*", default=[800, 1600, 2400, 3200, 4000, 4800, 5272])
    args = parser.parse_args()
    ys = losses(args.log)
    xs = [LOG_EVERY * (i + 1) for i in range(len(ys))]

    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for c in args.checkpoints:
        ax.axvline(c, color="#d9d8d4", lw=1, zorder=0)
    ax.plot(xs, ys, color=SERIES, lw=2)
    ax.set_yscale("log")
    ax.set_xlabel("training step (8 frames each)", color=MUTED)
    ax.set_ylabel("error (log scale)", color=MUTED)
    ax.set_title("SmolVLA training error, 2 passes over 120 episodes", color=INK, loc="left", fontsize=11)
    ax.text(args.checkpoints[0], ax.get_ylim()[1], " grey lines: saved snapshots", va="top", color=MUTED,
            fontsize=8)
    ax.annotate(f"{ys[0]:.2f}", (xs[0], ys[0]), textcoords="offset points", xytext=(6, 0), color=INK, fontsize=8)
    ax.annotate(f"{ys[-1]:.3f}", (xs[-1], ys[-1]), textcoords="offset points", xytext=(-8, 8), ha="right",
                color=INK, fontsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#bdbcb7")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color="#ecebe8", lw=0.8)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, facecolor=SURFACE)
    print(f"{len(ys)} points, first {ys[0]}, last {ys[-1]} -> {args.out}")


if __name__ == "__main__":
    main()
