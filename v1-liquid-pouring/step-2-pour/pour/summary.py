"""What is in the level network's data, and how the programmed pours that made it did.

python -m pour.summary --pours data/pours --out docs/images
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import SPLITS

FLOW_SPAN = 0.2  # s each side of a picture over which its flow (% per second) is measured


def flow(t: np.ndarray, level: np.ndarray) -> np.ndarray:
    """How fast the level rises at each picture, % per second, from FLOW_SPAN before to FLOW_SPAN after."""
    before = np.interp(t - FLOW_SPAN, t, level)
    after = np.interp(t + FLOW_SPAN, t, level)
    return (after - before) / (2 * FLOW_SPAN)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pours", type=Path, default=Path("data/pours"))
    parser.add_argument("--out", type=Path, default=Path("docs/images"))
    args = parser.parse_args()
    out = {}
    levels, flows = {}, {}
    for split in SPLITS:
        results = [json.loads(p.read_text()) for p in sorted((args.pours / split).glob("*.json"))]
        lv, fl = [], []
        for p in sorted((args.pours / split).glob("*.npz")):
            z = np.load(p)
            if len(z["t"]) == 0:  # stopped before it moved: nothing saved
                continue
            lv.append(z["level"])
            fl.append(flow(z["t"], z["level"]))
        levels[split] = np.concatenate(lv) if lv else np.zeros(0)
        flows[split] = np.concatenate(fl) if fl else np.zeros(0)
        ok = [r for r in results if not r["failure"]]
        out[split] = {
            "pours": len(results),
            "failed": dict(Counter(r["failure"][:50] for r in results if r["failure"])),
            "medium / large": [sum(r["size"] == size for r in results) for size in (0.15, 0.2)],
            "pictures": int(sum(r["frames"] for r in results)),
            "seconds per pour (median)": float(np.median([r["sim_seconds"] for r in ok])) if ok else None,
            "within 2 % of the target": f"{sum(r['within_window'] for r in results)} of {len(results)}",
            "balls spilled (total, most in one pour)": [
                sum(r["balls"]["spilled"] for r in results),
                max((r["balls"]["spilled"] for r in results), default=0),
            ],
            "top-ups": dict(Counter(r["topups"] for r in results)),
            "pictures by level (%)": {
                f"{a}-{a + 10}": int(((levels[split] >= a) & (levels[split] < a + 10)).sum())
                for a in range(0, 80, 10)
            },
            "pictures with flow over 2 % per second": int((flows[split] > 2).sum()),
        }
    print(json.dumps(out, indent=1))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "data_summary.json").write_text(json.dumps(out, indent=1))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)
    for split in SPLITS:
        axes[0].hist(levels[split], bins=np.arange(0, 81, 2.5), histtype="step", density=True, label=split)
        axes[1].hist(flows[split], bins=np.arange(-2, 30, 1), histtype="step", density=True, label=split)
    axes[0].set_xlabel("level in the picture (%)")
    axes[1].set_xlabel("flow in the picture (% per second)")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_ylabel("share of pictures")
        ax.legend()
    fig.savefig(args.out / "data_levels_flows.png", dpi=110)


if __name__ == "__main__":
    main()
