"""The charts for docs/uncapping-learning.md, from a real recorded run (runs/demos).

    python -m learn.figures --run runs/demos/seed300_size200.npz

1. unscrew-pull.png   the pull Arm B's wrist feels through one whole unscrewing
2. lift-test.png      one lift test with the cap on and one with it off: clean, and with sensor noise
3. turn-twist.png     the twist the wrist feels while turning: quarter 1, 6 and 7, clean and with noise
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .data import DRIFT, NOISE  # noqa: E402

PHASES = ("approach", "grip", "turn", "test", "down", "open", "turn back", "regrip", "lift away")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3 (validated together)
OUT = Path("docs/images/uncapping-learned")

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "font.size": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "lines.solid_capstyle": "round",
    }
)


def load(path: Path) -> dict:
    f = np.load(path)
    meta = json.loads(str(f["meta"]))
    out = {k: f[k] for k in f.files if k != "meta"}
    out["pull"] = -(out["wrist_force"][:, 2] - meta["tare"]["force"][2])  # up on the cap = positive
    out["twist"] = np.abs(out["wrist_torque"][:, 2] - meta["tare"]["torque"][2])
    out["meta"] = meta
    return out


def block(r: dict, phase: str, quarter: int) -> np.ndarray:
    return np.flatnonzero((r["phase"] == PHASES.index(phase)) & (r["quarter"] == quarter))


def noisy(x: np.ndarray, sigma: float, drift: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return x + rng.normal(0, sigma, x.shape) + rng.normal(0, drift)


def label_end(ax, x, y, text, color, dy=0.0) -> None:
    ax.annotate(
        text, (x[-1], y[-1]), xytext=(6, dy), textcoords="offset points", color=INK, va="center", fontsize=9
    )
    ax.plot(x[-1], y[-1], "o", ms=4, color=color, mec=SURFACE, mew=1.5)


def unscrew_pull(r: dict, out: Path) -> None:
    """The pull through one whole unscrewing, each lift test marked."""
    first = int(np.argmax(r["phase"] == PHASES.index("turn")))
    t = r["time"][first:] - r["time"][first]
    pull = r["pull"][first:]
    phase = r["phase"][first:]
    quarter = r["quarter"][first:]
    fig, ax = plt.subplots(figsize=(9, 3.2), dpi=160)
    ax.plot(t, pull, color=BLUE)
    ax.axhline(5, color=INK2, lw=0.9, ls=(0, (4, 3)))
    ax.text(t[-1], 3.4, "decision line: 5 N", ha="right", color=INK2, fontsize=8.5)
    # What happens in one cycle, under the first one.
    cyc = np.flatnonzero(quarter == 1)
    spans = [
        ("turn", ("turn",), -3.0, "center"),
        ("test", ("test",), -5.2, "center"),
        ("let go, turn back, grip again", ("down", "open", "turn back", "regrip"), -3.0, "left"),
    ]
    for text, names, y, ha in spans:
        k = cyc[np.isin(phase[cyc], [PHASES.index(n) for n in names])]
        if len(k):
            a, b = t[k[0]], t[k[-1]]
            ax.annotate("", (a, -1.6), (b, -1.6), arrowprops={"arrowstyle": "<->", "color": INK2, "lw": 0.8})
            x = (a + b) / 2 if ha == "center" else a
            ax.text(x, y, text, ha=ha, va="top", fontsize=8, color=INK2)
    last_q = int(quarter.max())
    for q in range(1, last_q + 1):
        s = np.flatnonzero((phase == PHASES.index("test")) & (quarter == q))
        if not len(s):
            continue
        k = s[-1]
        off = q == last_q
        ax.annotate(
            f"test {q}\n{'cap off' if off else 'still on'}",
            (t[k], pull[k]),
            xytext=(0, 34 if off else 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=ORANGE if off else INK,
            fontweight="bold" if off else "normal",
        )
    ax.set_ylim(-7.5, 20)
    ax.set_xlim(0, t[-1] + 0.5)
    ax.set_xlabel("seconds since the first turn")
    ax.set_ylabel("pull at the wrist (N)\nup +, push down −")
    ax.set_title("What Arm B's wrist feels while unscrewing one cap (scene 300, clean values)")
    fig.tight_layout()
    fig.savefig(out / "unscrew-pull.png")
    plt.close(fig)


def lift_test(r: dict, out: Path) -> None:
    """Test 6 (cap on) and test 7 (cap off), clean and with sensor noise."""
    last_q = int(r["quarter"].max())
    on, off = block(r, "test", last_q - 1), block(r, "test", last_q)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), dpi=160, sharey=True)
    for ax, with_noise in zip(axes, (False, True), strict=True):
        for idx, color, text, seed in ((on, BLUE, "cap still on", 1), (off, ORANGE, "cap off", 2)):
            t = r["time"][idx] - r["time"][idx[0]]
            y = r["pull"][idx]
            if with_noise:
                y = noisy(y, NOISE["force"], DRIFT["force"], seed)
            ax.plot(t, y, color=color, label=text)
            label_end(ax, t, y, text, color)
        ax.axhline(5, color=INK2, lw=0.9, ls=(0, (4, 3)))
        ax.set_xlim(0, 0.85)
        ax.set_xlabel("seconds since the test began (lift 0.4 s, hold 0.3 s)")
    axes[0].set_ylim(-4, 18)
    axes[0].set_ylabel("pull at the wrist (N)")
    axes[0].set_title("Clean (what the simulation gives)")
    axes[1].set_title("With a real sensor's noise (what the policy gets)")
    axes[0].legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / "lift-test.png")
    plt.close(fig)


def turn_twist(r: dict, out: Path) -> None:
    """The twist while turning: quarter 1 (the seal breaks), 6, and the last (the cap comes off)."""
    last_q = int(r["quarter"].max())
    series = (
        (1, BLUE, "quarter 1: the seal breaks"),
        (last_q - 1, AQUA, f"quarter {last_q - 1}: cap stays on"),
        (last_q, ORANGE, f"quarter {last_q}: cap comes off"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), dpi=160, sharey=True)
    for ax, with_noise in zip(axes, (False, True), strict=True):
        for i, (q, color, text) in enumerate(series):
            idx = block(r, "turn", q)
            t = r["time"][idx] - r["time"][idx[0]]
            y = r["twist"][idx]
            if with_noise:
                y = noisy(y, NOISE["torque"], DRIFT["torque"], 10 + i)
            ax.plot(t, y, color=color, label=text)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("seconds into the quarter turn")
    axes[0].set_ylim(-0.3, 0.8)
    axes[0].set_ylabel("twist at the wrist (N m)")
    axes[0].set_title("Clean: only quarter 1 stands out")
    axes[1].set_title("With noise: quarters 6 and 7 can't be told apart")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8.5)
    # Direct labels on the clean panel (the light aqua line needs one: contrast).
    for q, _color, text in series:
        idx = block(r, "turn", q)
        t = r["time"][idx] - r["time"][idx[0]]
        y = r["twist"][idx]
        k = int(len(t) * {1: 0.3, last_q - 1: 0.75, last_q: 0.75}[q])
        dy = {1: 9, last_q - 1: 9, last_q: -10}[q]
        axes[0].annotate(
            text.split(":")[0], (t[k], y[k]), xytext=(0, dy), textcoords="offset points", fontsize=8.5,
            ha="center", color=INK,
        )  # fmt: skip
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "turn-twist.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", type=Path, default=Path("runs/demos/seed300_size200.npz"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    r = load(args.run)
    unscrew_pull(r, args.out)
    lift_test(r, args.out)
    turn_twist(r, args.out)
    print("written to", args.out)


if __name__ == "__main__":
    main()
