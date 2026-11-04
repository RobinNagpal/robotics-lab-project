"""Pictures of the level networks for the docs: what each method reads at three moments of a test pour, and
the validation score of both networks after every epoch.

    python -m level.figures
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import rule
from .data import Pictures
from .net import make
from .train import device, predict, scores


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=Path("data/level"))
    parser.add_argument("--out", type=Path, default=Path("docs/images"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    test = Pictures(args.data / "test.npz")
    dev = device()
    net = make("custom").to(dev)
    net.load_state_dict(torch.load("runs/custom/best.pt", map_location=dev))
    cnn = predict(net, test, dev)
    rl = rule.predict(json.loads(Path("runs/rule/lines.json").read_text()), test)
    truth = test.labels

    # Three moments of one test pour: the first balls on their way, more falling, settled.
    first = np.where((truth[:, 0] < 3) & (truth[:, 2] > 0.4))[0]
    i0 = int(first[0])
    pour_ = test.pour == test.pour[i0]
    falling = np.where(pour_ & (truth[:, 2] > 1) & (truth[:, 0] > 20))[0]
    settled = np.where(pour_ & np.isin(test.phase, (4, 6)) & (truth[:, 0] > 30))[0]
    picks = [
        (i0, "first balls on their way"),
        (int(falling[len(falling) // 2]), "balls falling"),
        (int(settled[-1]), "settled"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(8, 5.6), constrained_layout=True)
    for ax, (i, what) in zip(axes, picks, strict=True):
        ax.imshow(test.images[i].permute(1, 2, 0).numpy(), interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"{what}\ntrue {truth[i, 0]:.0f} %   rule {rl[i, 0]:.0f} %   CNN {cnn[i, 0]:.0f} %\n"
            f"balls in the air: true {truth[i, 2]:.1f} %, CNN {cnn[i, 2]:.1f} %",
            fontsize=8,
        )
    fig.savefig(args.out / "three_moments.png", dpi=110)

    # Validation level error after every epoch.
    fig, ax = plt.subplots(figsize=(6.5, 3.2), constrained_layout=True)
    for name, label in (("custom", "custom CNN"), ("resnet18", "ResNet-18 (pretrained)")):
        rows = list(csv.DictReader((Path("runs") / name / "results.csv").read_text().splitlines()))
        ax.plot(
            [float(r["minutes"]) for r in rows], [float(r["level_err_band"]) for r in rows], "o-", label=label
        )
    val = Pictures(args.data / "val.npz")
    rule_val = scores(rule.predict(json.loads(Path("runs/rule/lines.json").read_text()), val), val)[
        "level_err_band"
    ]
    ax.axhline(rule_val, color="gray", ls="--", label="blue-pixel rule")
    ax.set_xlabel("minutes of training")
    ax.set_ylabel("level error, 20-65 % full\n(points, validation)")
    ax.legend()
    fig.savefig(args.out / "training.png", dpi=110)

    shutil.copy("runs/evaluation/test_pours.png", args.out / "test_pours.png")


if __name__ == "__main__":
    main()
