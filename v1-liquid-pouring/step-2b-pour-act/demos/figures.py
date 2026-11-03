"""Pictures of the demonstrations for the docs.

views.png    what the policy sees at four moments of one pour: strip, wide picture, wrist picture
chunk.png    one pour over time: tilt and level, and Arm B's six joint targets with one 1 s chunk
nudge.png    the same start poured as programmed and with the nudge: where the lip went
demos.png    how every demonstration ended against its target, as programmed and nudged

python -m demos.figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pour.pour import PHASES

ROOT = Path(__file__).resolve().parents[1]
DEMOS = ROOT / "data" / "demos"
OUT = ROOT / "docs" / "images"
PHASE_COLOURS = {
    "move in": "#e8eef7",
    "find flow": "#fdf1d6",
    "pour": "#fde0cc",
    "tip back": "#f9d0d6",
    "settle": "#e3f2e1",
    "top up": "#f3e3f7",
    "done": "#eeeeee",
}


def load(name: str) -> tuple[dict, dict]:
    path = next(DEMOS.glob(f"*/{name}.npz"))
    return dict(np.load(path)), json.loads(path.with_suffix(".json").read_text())


def shade_phases(ax, t: np.ndarray, phase: np.ndarray, labels: bool = False) -> None:
    """Colour the background by phase; ``labels`` writes each phase's name at the top."""
    start = 0
    for i in range(1, len(t) + 1):
        if i == len(t) or phase[i] != phase[start]:
            name = PHASES[phase[start]]
            ax.axvspan(t[start] - 0.05, t[i - 1] + 0.05, color=PHASE_COLOURS[name], lw=0)
            if labels and t[i - 1] - t[start] > 0.8:
                ax.text(
                    (t[start] + t[i - 1]) / 2,
                    1.01,
                    name,
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            start = i


def views(name: str, out: Path) -> None:
    z, r = load(name)
    phase = z["phase"]
    moments = [
        (
            "moving in",
            int(np.flatnonzero(phase == PHASES.index("move in"))[len(np.flatnonzero(phase == 0)) // 2]),
        ),
        ("first balls", int(np.flatnonzero(z["receiver"] > 0)[0]) + 3),
        ("pouring", int(np.flatnonzero(z["level"] >= 0.6 * r["target_pct"])[0])),
        ("finished", len(phase) - 1),
    ]
    fig = plt.figure(figsize=(12, 5.2))
    grid = fig.add_gridspec(2, 8, width_ratios=[1, 2.4] * 4, hspace=0.3, wspace=0.08)
    for j, (title, i) in enumerate(moments):
        ax = fig.add_subplot(grid[:, 2 * j])
        ax.imshow(z["strip"][i], interpolation="nearest")
        ax.set_title("strip", fontsize=9)
        ax.axis("off")
        ax = fig.add_subplot(grid[0, 2 * j + 1])
        ax.imshow(z["wide"][i], interpolation="nearest")
        ax.set_title(f"{title}, {z['t'][i]:.1f} s (level {z['level'][i]:.0f} %)\nwide", fontsize=9)
        ax.axis("off")
        ax = fig.add_subplot(grid[1, 2 * j + 1])
        ax.imshow(z["wrist"][i], interpolation="nearest")
        ax.set_title("wrist", fontsize=9)
        ax.axis("off")
    fig.suptitle(
        f"What the policy sees, every 0.1 s (one pour, {r['start']}, target {r['target_pct']:.0f} %)",
        fontsize=12,
    )
    fig.savefig(out / "views.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def chunk(name: str, out: Path) -> None:
    z, r = load(name)
    t, phase = z["t"], z["phase"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
    ax = axes[0]
    shade_phases(ax, t, phase, labels=True)
    ax.plot(t, z["tilt"], color="#2f5fa8", label="tilt of the source (degrees)")
    ax.set_ylabel("tilt (degrees)", color="#2f5fa8")
    ax2 = ax.twinx()
    ax2.plot(t, z["level"], color="#d1495b", label="receiver level (%)")
    ax2.axhline(r["target_pct"], color="#d1495b", ls=":", lw=1)
    ax2.set_ylabel("receiver level (%)", color="#d1495b")
    ax.set_title(f"One programmed pour ({r['start']}, target {r['target_pct']:.0f} %)", pad=16)
    ax = axes[1]
    shade_phases(ax, t, phase)
    i = int(np.flatnonzero(phase == PHASES.index("pour"))[0]) + 15  # a tick during the pour
    change = np.degrees(z["intent"] - z["intent"][0])
    for j in range(6):
        ax.plot(t, change[:, j], lw=1.2, label=f"joint {j + 1}")
    ax.axvspan(t[i], t[min(i + 10, len(t) - 1)], color="#222222", alpha=0.12)
    ax.axvline(t[i], color="#222222", lw=1)
    ax.annotate(
        "at this tick the policy\npredicts the next 1 s\n(10 ticks) of targets",
        xy=(t[i] + 0.5, change[i : i + 10].max() + 2),
        xytext=(t[i] - 9, change.max() * 0.65),
        arrowprops={"arrowstyle": "->"},
        fontsize=9,
    )
    ax.set_ylabel("joint target, change from\nthe start (degrees)")
    ax.set_xlabel("seconds")
    ax.legend(ncol=6, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "chunk.png", dpi=130)
    plt.close(fig)


def nudge(start: str, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for k, colour, label in ((0, "#222222", "as programmed"), (1, "#d1495b", "nudged")):
        z, r = load(f"{start}_demo{k}")
        placed = z["phase"] > 0
        t = z["t"][placed] - z["t"][placed][0]
        axes[0].plot(t, z["lip"][placed, 1] * 1000, color=colour, label=label)
        axes[1].plot(
            t,
            z["tilt"][placed],
            color=colour,
            label=f"{label}: ended {r['level_pct']:.1f} % (target {r['target_pct']:.0f} %)",
        )
    axes[0].axhspan(-4, 4, color="#f2c14e", alpha=0.15)
    axes[0].text(0.3, 3.3, "4 mm sideways spills about 5 %", fontsize=8)
    axes[0].set(
        xlabel="seconds after the lip is in place",
        ylabel="lip, sideways from the rim (mm)",
        title="Where the lip is",
        ylim=(-5, 5),
    )
    axes[1].set(
        xlabel="seconds after the lip is in place", ylabel="tilt (degrees)", title="How far it is tipped"
    )
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(f"The same start ({start}) poured twice", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "nudge.png", dpi=130)
    plt.close(fig)


def demos(out: Path) -> dict:
    rows = [json.loads(p.read_text()) for p in sorted(DEMOS.glob("*/*.json"))]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bins = np.arange(-3.5, 5.0, 0.5)
    for noise, colour, label in ((0.0, "#222222", "as programmed"), (1.0, "#d1495b", "nudged")):
        err = [r["level_pct"] - r["target_pct"] for r in rows if r["noise"] == noise and not r["failure"]]
        ax.hist(
            np.clip(err, -3.5, 4.5),
            bins=bins,
            histtype="step",
            lw=2,
            color=colour,
            label=f"{label} ({len(err)} pours)",
        )
    ax.axvspan(-2, 2, color="#7fbf7f", alpha=0.25, label="about the window (2 to 4 balls)")
    ax.set(
        xlabel="final level minus target (points)",
        ylabel="pours",
        title="How every demonstration ended",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "demos.png", dpi=130)
    plt.close(fig)
    summary = {}
    for noise in (0.0, 1.0):
        ok = [r for r in rows if r["noise"] == noise and not r["failure"]]
        summary[str(noise)] = {
            "pours": len(ok),
            "within_window": sum(r["within_window"] for r in ok),
            "pours_with_spill": sum(r["balls"]["spilled"] > 0 for r in ok),
            "spilled": sum(r["balls"]["spilled"] for r in ok),
            "topups": sum(r["topups"] for r in ok),
            "median_seconds": float(np.median([r["sim_seconds"] for r in ok])),
        }
    for split in ("train", "val", "test"):
        ok = [json.loads(p.read_text()) for p in sorted((DEMOS / split).glob("*.json"))]
        ok = [r for r in ok if not r["failure"]]
        summary[split] = {
            "pours": len(ok),
            "nudged": sum(r["noise"] > 0 for r in ok),
            "medium": sum(r["size"] < 0.17 for r in ok),
            "ticks": sum(r["ticks"] for r in ok),
        }
    (out / "demos_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--demo", default="seed150_size200_demo0", help="the pour drawn in views and chunk")
    parser.add_argument("--nudge-start", default="seed150_size200")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    views(args.demo, OUT)
    chunk(args.demo, OUT)
    nudge(args.nudge_start, OUT)
    print(json.dumps(demos(OUT), indent=1))


if __name__ == "__main__":
    main()
