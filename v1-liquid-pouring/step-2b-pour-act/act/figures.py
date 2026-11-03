"""Pictures for the docs, from the training runs and the evaluation (act/evaluate.py).

training.png    validation error while training, ACT and the CNN + MLP policy
test_pours.png  where every test pour ended, by method, against the 48 to 52 % window
trace.png       one test pour, tip angle and level over time: the programmed pour against the policies
pour.gif        one ACT pour from the pour camera (``--gif``; runs the pour again, about 2 minutes)

python -m act.figures --gif
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from demos.collect import STARTS

from .evaluate import OUT, load

RUNS = Path(__file__).resolve().parents[1] / "runs"
IMAGES = Path(__file__).resolve().parents[1] / "docs" / "images"
COLOURS = {
    "programmed": "#222222",
    "programmed-cnn": "#888888",
    "act": "#d1495b",
    "mlp": "#00798c",
}
NAMES = {
    "programmed": "programmed, true count",
    "programmed-cnn": "programmed, level network",
    "act": "ACT",
    "mlp": "CNN + MLP",
}


def training(runs: list[str], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for name in runs:
        path = RUNS / name / "results.csv"
        if not path.exists():
            continue
        rows = list(csv.DictReader(path.open()))
        minutes = [float(r["minutes"]) for r in rows]
        axes[0].plot(
            minutes,
            [float(r["val_action"]) for r in rows],
            label=NAMES.get(name, name),
            color=COLOURS.get(name),
        )
        axes[1].plot(minutes, [float(r["val_next_deg"]) for r in rows], color=COLOURS.get(name))
    axes[0].set(xlabel="minutes of training", ylabel="action error (scaled)", title="Validation: 10 ticks")
    axes[1].set(xlabel="minutes of training", ylabel="degrees per joint", title="Validation: the next tick")
    axes[0].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "training.png", dpi=130)
    plt.close(fig)


def test_pours(methods: list[str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axhspan(48, 52, color="#7fbf7f", alpha=0.3, label="48 to 52 %")
    for i, m in enumerate(methods):
        results = load(m)
        if not results:
            continue
        y = np.array([r["level_pct"] for r in results])
        x = i + np.linspace(-0.25, 0.25, len(y))
        failed = np.array([bool(r.get("failure")) for r in results])
        ax.scatter(x[~failed], y[~failed], s=22, color=COLOURS.get(m, "C0"))
        ax.scatter(x[failed], y[failed], s=30, marker="x", color=COLOURS.get(m, "C0"))
        inside = sum(r["within_window"] for r in results)
        ax.text(i, 2, f"{inside} of {len(results)}\nin the window", ha="center", fontsize=9)
    ax.set_xticks(range(len(methods)), [NAMES.get(m, m) for m in methods])
    ax.set(ylabel="receiver's final level (%)", ylim=(0, 75), title="Every test pour to 50 % (x: failed)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "test_pours.png", dpi=130)
    plt.close(fig)


def trace(start: str, policies: list[str], out: Path) -> None:
    from demos import record

    result, ticks = record.run(STARTS / f"{start}.npy", 50.0)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    lines = [("programmed", np.array(ticks["t"]), np.array(ticks["tilt"]), np.array(ticks["level"]))]
    for p in policies:
        path = OUT / p / f"{start}_50.npz"
        if path.exists():
            z = np.load(path)
            lines.append((p, z["t"], z["tilt"], z["level"]))
    for name, t, tilt, level in lines:
        c = COLOURS.get(name)
        axes[0].plot(t, tilt, color=c, label=NAMES.get(name, name))
        axes[1].plot(t, level, color=c)
    axes[0].axhspan(81, 88, color="#f2c14e", alpha=0.25, label="where the balls start to flow")
    axes[1].axhspan(48, 52, color="#7fbf7f", alpha=0.3)
    axes[0].set(ylabel="tip angle (degrees)", title=f"One test pour ({start}), target 50 %")
    axes[1].set(ylabel="receiver level (%, counted)", xlabel="seconds")
    axes[0].legend(loc="lower right", fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "trace.png", dpi=130)
    plt.close(fig)


def gif(start: str, policy: str, out: Path) -> None:
    import torch
    from PIL import Image

    from .policy import Policy
    from .rollout import run

    torch.set_num_threads(4)
    movie: list = []
    result, _ = run(STARTS / f"{start}.npy", Policy(policy), 50.0, movie=movie)
    frames = [Image.fromarray(f).resize((320, 240)) for f in movie]
    frames[0].save(out / "pour.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)
    print(json.dumps(result))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--methods", nargs="+", default=["programmed", "programmed-cnn", "act", "mlp"])
    parser.add_argument("--start", default="seed150_size200", help="the test pour drawn in trace.png")
    parser.add_argument("--gif", action="store_true")
    args = parser.parse_args()
    IMAGES.mkdir(parents=True, exist_ok=True)
    policies = [m for m in args.methods if not m.startswith("programmed")]
    training(policies, IMAGES)
    test_pours(args.methods, IMAGES)
    trace(args.start, policies, IMAGES)
    if args.gif:
        gif(args.start, policies[0], IMAGES)


if __name__ == "__main__":
    main()
