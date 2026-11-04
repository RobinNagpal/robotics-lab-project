"""How well each trained policy copies the expert, tick by tick (no closed loop: the policy does not drive).

Every tick of the validation and test demonstrations is shown to the policy, and what it would command
is compared with what the expert commanded. This is "offline" grading: the policy never moves the arm,
so its mistakes cannot add up. Whether it can pour on its own is the closed-loop test (act/evaluate.py).

    runs/offline/summary.json   the scores, overall and phase by phase, for both splits, and for the
                                programmed and nudged pours apart; with "hold still" to compare
    docs/images/training.png    validation error while training
    docs/images/phases.png      the next tick's error, phase by phase
    docs/images/replay.png      one validation pour replayed: the tilt the expert's targets aim for
                                against the policies', and "finished"

    python -m act.analyse --runs act mlp
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
from lab_scene import scene as lab
from pour import start
from pour.pour import PHASES
from pour.tilt import ARM

from demos.collect import STARTS

from .data import CHUNK, Demos, Stats, ready
from .model import make
from .train import device

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
IMAGES = ROOT / "docs" / "images"
COLOURS = {"expert": "#222222", "act": "#d1495b", "mlp": "#00798c", "hold_still": "#b9bec7"}
NAMES = {"expert": "expert (programmed)", "act": "ACT", "mlp": "CNN + MLP", "hold_still": "hold still"}


def load_net(name: str, dev: str):
    settings = json.loads((RUNS / name / "settings.json").read_text())
    net = make(settings["model"], tune_strip=settings.get("tune_strip", False))
    net.load_state_dict(torch.load(RUNS / name / "best.pt", map_location="cpu"))
    return net.to(dev).eval(), Stats.load(RUNS / name / "stats.json")


@torch.no_grad()
def predict(net, stats: Stats, demos: Demos, dev: str, batch: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """For every tick: the joint targets it would give (N, CHUNK, 6; rad) and P(finished) (N, CHUNK)."""
    targets, done = [], []
    for i in range(0, len(demos), batch):
        idx = np.arange(i, min(i + batch, len(demos)))
        out = net(ready(demos.batch(idx, stats), dev))
        change = out["actions"].cpu().numpy() * stats.action_std + stats.action_mean
        targets.append(demos.ctrl[idx][:, None] + change)
        done.append(torch.sigmoid(out["done"]).cpu().numpy())
    return np.concatenate(targets), np.concatenate(done)


def scores(targets: np.ndarray, done: np.ndarray, demos: Demos) -> dict:
    """Overall and per phase: the next tick's error (degrees, mean over joints), the whole second's error,
    and how often "finished" (for the next tick) is right."""
    idx = np.arange(len(demos))
    truth = demos.ctrl[:, None] + demos.actions(idx)
    err = np.degrees(np.abs(targets - truth))  # N, CHUNK, 6
    right = (done[:, 0] > 0.5) == (demos.done(idx)[:, 0] > 0.5)

    def summarise(mask: np.ndarray) -> dict:
        return {
            "ticks": int(mask.sum()),
            "next_tick_deg": round(float(err[mask, 0].mean()), 4),
            "next_tick_worst_joint_deg": round(float(err[mask, 0].max(1).mean()), 4),
            "whole_second_deg": round(float(err[mask].mean()), 4),
            "finished_right": round(float(right[mask].mean()), 4),
        }

    def by_phase(keep: np.ndarray) -> dict:
        out = {"all": summarise(keep)}
        for k, name in enumerate(PHASES):
            mask = keep & (demos.phase == k)
            if mask.any():
                out[name] = summarise(mask)
        return out

    nudged = np.array([r["noise"] > 0 for r in demos.results])[demos.demo]
    everything = np.ones(len(demos), bool)
    return {**by_phase(everything), "clean": by_phase(~nudged), "nudged": by_phase(nudged)}


def hold_still(demos: Demos) -> tuple[np.ndarray, np.ndarray]:
    """The simplest guess, to compare with: keep the joint targets where they are, never say finished."""
    targets = np.repeat(demos.ctrl[:, None], CHUNK, axis=1)
    return targets, np.zeros((len(demos), CHUNK))


def tilt_of(targets: np.ndarray, start_path: Path) -> np.ndarray:
    """The tilt (degrees) Arm B's joint targets aim for, with the tube where Step 1b left it in the fingers.

    The tube turns a little in the fingers while pouring, so this is not the tube's true tilt; but it is
    the same measure for the expert and the policies, and easier to read than six joint angles.
    """
    model, data, _, meta = start.load(start_path)
    joints = lab.arm_joint_ids(model, ARM)
    site = model.site(lab.pinch_site(ARM)).id
    tube = data.body(meta["source"]).xmat.reshape(3, 3)
    rel = data.site_xmat[site].reshape(3, 3).T @ tube
    w = mujoco.MjData(model)
    w.qpos[:] = data.qpos
    shape = targets.shape[:-1]
    out = np.empty(int(np.prod(shape)))
    for i, q in enumerate(targets.reshape(-1, 6)):
        w.qpos[joints] = q
        mujoco.mj_kinematics(model, w)
        axis_z = (w.site_xmat[site].reshape(3, 3) @ rel)[2, 2]
        out[i] = math.degrees(math.acos(np.clip(axis_z, -1, 1)))
    return out.reshape(shape)


def replay(name: str, demos: Demos, preds: dict, out: Path) -> None:
    d = demos.names.index(name)
    idx = np.flatnonzero(demos.demo == d)
    r = demos.results[d]
    start_path = STARTS / f"{r['start']}.npy"
    t = demos.t[idx]
    fig, axes = plt.subplots(
        3, 1, figsize=(10, 8.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.6, 1.2]}
    )
    expert = tilt_of(demos.intent[idx], start_path)
    axes[0].plot(t, expert, color=COLOURS["expert"], lw=2.5, label=NAMES["expert"])
    for m, (targets, done) in preds.items():
        nxt = tilt_of(targets[idx, 0], start_path)
        axes[0].plot(t, np.r_[expert[0], nxt[:-1]], color=COLOURS[m], lw=1.2, label=f"{NAMES[m]}, next tick")
        axes[2].plot(t, done[idx, 0], color=COLOURS[m], label=NAMES[m])
    # ACT's whole 1 s chunks at a few ticks, to show what it plans ahead.
    if "act" in preds:
        for i in range(10, len(idx), 25):
            chunk = tilt_of(preds["act"][0][idx[i]], start_path)
            axes[0].plot(t[i] + 0.1 * np.arange(1, 11), chunk, color=COLOURS["act"], lw=3, alpha=0.35)
        axes[0].plot([], [], color=COLOURS["act"], lw=3, alpha=0.35, label="ACT, its whole 1 s chunk")
    axes[0].set(
        ylabel="tilt the joint targets\naim for (degrees)",
        title=f"One validation pour replayed ({r['start']}, target {r['target_pct']:.0f} %)\n"
        "each tick: what would each policy command?",
    )
    axes[0].legend(fontsize=8, loc="lower right")
    axes[1].plot(t, demos.extra[idx, 0], color="#2f5fa8")
    axes[1].axhline(r["target_pct"], color="#2f5fa8", ls=":")
    axes[1].set(ylabel="receiver level\n(%, counted)")
    finished = (demos.phase[idx] == PHASES.index("done")).astype(float)
    axes[2].fill_between(t, 0, finished, color="#cccccc", step="mid", label="expert has finished")
    axes[2].set(ylabel='P("finished")', xlabel="seconds", ylim=(-0.05, 1.05))
    axes[2].legend(fontsize=8, loc="upper left")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "replay.png", dpi=130)
    plt.close(fig)


def phases_figure(summary: dict, runs: list[str], out: Path) -> None:
    """Validation, by phase: the next tick's error, the whole second's error, and "finished" wrong."""
    names = [p for p in PHASES if p in summary[runs[0]]["val"]]
    methods = ["hold_still", *runs]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    x = np.arange(len(names))
    w = 0.8 / len(methods)
    panels = [
        ("next_tick_deg", "degrees, mean over the 6 joints", "Next tick's joint targets: error"),
        ("whole_second_deg", "degrees, mean over the 6 joints", "The whole 1 s chunk: error"),
        ("finished_right", "% of ticks", '"Finished" wrong'),
    ]
    for ax, (key, ylabel, title) in zip(axes, panels, strict=True):
        for j, m in enumerate(methods):
            if key == "finished_right" and m == "hold_still":
                continue
            val = summary[m]["val"]
            y = [100 * (1 - val[p][key]) if key == "finished_right" else val[p][key] for p in names]
            ax.bar(x + (j - (len(methods) - 1) / 2) * w, y, w, color=COLOURS[m], label=NAMES[m])
        ax.set_xticks(x, [f"{p}\n({summary[runs[0]]['val'][p]['ticks']})" for p in names], fontsize=8)
        ax.set(ylabel=ylabel, title=title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=9)
    fig.suptitle("Validation pours, phase by phase (ticks in brackets)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "phases.png", dpi=130)
    plt.close(fig)


def training_figure(runs: list[str], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for m in runs:
        rows = list(csv.DictReader((RUNS / m / "results.csv").open()))
        minutes = [float(r["minutes"]) for r in rows]
        axes[0].plot(
            minutes,
            [float(r["train_action"]) for r in rows],
            color=COLOURS[m],
            ls="--",
            lw=1,
            label=f"{NAMES[m]}, training",
        )
        axes[0].plot(
            minutes,
            [float(r["val_action"]) for r in rows],
            color=COLOURS[m],
            lw=2,
            label=f"{NAMES[m]}, validation",
        )
        axes[1].plot(
            minutes, [float(r["val_next_deg"]) for r in rows], color=COLOURS[m], lw=2, label=NAMES[m]
        )
    axes[0].set(
        xlabel="minutes of training",
        ylabel="action error (scaled)",
        title="Error over the whole 1 s chunk",
        ylim=(0, 0.5),
    )
    axes[1].set(
        xlabel="minutes of training",
        ylabel="degrees, mean over the 6 joints",
        title="Validation: the next tick's targets",
        ylim=(0, 0.25),
    )
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "training.png", dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", nargs="+", default=["act", "mlp"])
    parser.add_argument(
        "--replay", default=None, help="the validation demonstration replayed (default: chosen)"
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    dev = device()
    IMAGES.mkdir(parents=True, exist_ok=True)
    splits = {s: Demos(ROOT / "data" / "demos" / s) for s in ("val", "test")}
    summary: dict = {}
    preds: dict = {}
    for m in args.runs:
        net, stats = load_net(m, dev)
        summary[m] = {}
        for s, demos in splits.items():
            targets, done = predict(net, stats, demos, dev)
            summary[m][s] = scores(targets, done, demos)
            if s == "val":
                preds[m] = (targets, done)
        print(m, json.dumps({s: summary[m][s]["all"] for s in splits}), flush=True)
    summary["hold_still"] = {s: scores(*hold_still(demos), demos) for s, demos in splits.items()}
    print("hold_still", json.dumps({s: summary["hold_still"][s]["all"] for s in splits}), flush=True)
    (RUNS / "offline").mkdir(exist_ok=True)
    (RUNS / "offline" / "summary.json").write_text(json.dumps(summary, indent=1))
    val = splits["val"]
    name = args.replay
    if name is None:  # a clean pour of a large tube with a top-up: every phase in one picture
        name = next(
            n
            for n, r in zip(val.names, val.results, strict=True)
            if r["noise"] == 0 and r["size"] > 0.17 and r["topups"] == 1
        )
    print("replayed:", name)
    replay(name, val, preds, IMAGES)
    phases_figure(summary, args.runs, IMAGES)
    training_figure(args.runs, IMAGES)


if __name__ == "__main__":
    main()
