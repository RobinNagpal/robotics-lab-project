"""Grade level networks and the rule on the test pictures (seeds 150-159), all the same way.

    python -m level.evaluate --runs custom resnet18 --out runs/evaluation

For each method: the level error where the pour is decided (20 to 65 % full), by tube size, while the
balls are falling and once they have settled (what the stop and top-up decisions read), the flow error
while flowing, and the error in the balls still in the air. Also a picture of a few test pours, each
method's level and flow against the truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import rule
from .data import Pictures
from .net import make
from .train import device, predict

SETTLED = (4, 6)  # phases "settle" and "done" (pour.pour.PHASES)


def grade(pred: np.ndarray, data: Pictures) -> dict:
    truth = data.labels
    err = np.abs(pred - truth)
    band = (truth[:, 0] >= 20) & (truth[:, 0] <= 65)
    settled = band & np.isin(data.phase, SETTLED)
    moving = band & ~np.isin(data.phase, SETTLED)
    flowing = truth[:, 1] > 2

    def level(mask):
        return {
            "pictures": int(mask.sum()),
            "mean_error": round(float(err[mask, 0].mean()), 2),
            "within_2": round(float((err[mask, 0] <= 2).mean()), 3),
            "worst": round(float(err[mask, 0].max()), 1),
        }

    out = {
        "level 20-65 %": level(band),
        "level 20-65 %, medium": level(band & np.isclose(data.size, 0.15)),
        "level 20-65 %, large": level(band & np.isclose(data.size, 0.2)),
        "level, settled": level(settled),
        "level, while pouring": level(moving),
        "level, empty receiver (true 0 %)": round(float(err[truth[:, 0] == 0, 0].mean()), 2),
        "flow error while flowing (% per s)": round(float(err[flowing, 1].mean()), 2),
    }
    if not np.isnan(pred[:, 2]).all():
        out["air error (%)"] = round(float(err[:, 2].mean()), 3)
        out["air error while flowing (%)"] = round(float(err[flowing, 2].mean()), 3)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=Path("data/level"))
    parser.add_argument("--runs", nargs="+", default=["custom"])
    parser.add_argument("--out", type=Path, default=Path("runs/evaluation"))
    args = parser.parse_args()
    test = Pictures(args.data / "test.npz")
    dev = device()
    preds = {"rule": rule.predict(json.loads(Path("runs/rule/lines.json").read_text()), test)}
    for name in args.runs:
        settings = json.loads((Path("runs") / name / "settings.json").read_text())
        net = make(settings["model"]).to(dev)
        net.load_state_dict(torch.load(Path("runs") / name / "best.pt", map_location=dev))
        preds[name] = predict(net, test, dev)
    results = {name: grade(p, test) for name, p in preds.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "evaluation.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))

    # A few test pours, medium and large.
    pours = []
    for size in (0.15, 0.2):
        ids = np.unique(test.pour[np.isclose(test.size, size)])
        pours += list(ids[:2])
    fig, axes = plt.subplots(
        2, len(pours), figsize=(4 * len(pours), 6), sharex="col", constrained_layout=True
    )
    for j, pid in enumerate(pours):
        m = test.pour == pid
        t = test.t[m]
        axes[0, j].plot(t, test.labels[m, 0], color="black", lw=2, label="truth")
        axes[1, j].plot(t, test.labels[m, 1], color="black", lw=2, label="truth")
        for name, p in preds.items():
            axes[0, j].plot(t, p[m, 0], lw=1, label=name)
            axes[1, j].plot(t, np.clip(p[m, 1], -10, 60), lw=1, label=name)
        axes[0, j].set_title(
            f"{test.names[pid]} ({'medium' if test.size[m][0] < 0.17 else 'large'})", fontsize=8
        )
        axes[0, j].set_ylabel("level (%)")
        axes[1, j].set_ylabel("flow (% per second)")
        axes[1, j].set_xlabel("seconds")
    axes[0, 0].legend(fontsize=8)
    fig.savefig(args.out / "test_pours.png", dpi=100)


if __name__ == "__main__":
    main()
