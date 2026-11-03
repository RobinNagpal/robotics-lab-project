"""Grade the side-grasp network on its test pictures, as evaluate.py grades the top-down one.

    python -m grasp.evaluate_side --weights runs/side_grconv/best.pt
    python -m grasp.evaluate_side --weights runs/side_v2/best.pt --data data_side2 --version 2
    python -m grasp.evaluate_side --weights runs/side_v3/best.pt --data data_side3 --version 3

Version 2 (docs/side/version2.md): the fingers come in barely open, directions are counted
from the rack's side (each sample's frame_yaw), and the scores are smoothed over neighbouring
directions before choosing. ``--smooth`` alone tries only the last on a version 1 network.
Version 3 (docs/side/version3.md): as version 2, and the network scores each direction at each of
side.HEIGHTS; the chosen output says both where the gripper comes from and how low it grips.

1. The cut-off under which the network says "no clear direction" is chosen on
   the validation pictures (the most targets answered right).
2. Fast grading, every test target: the network's grasp is fast-checked where
   it chose to grasp. Good = clear, and the pinch point within 5 mm of the axis.
3. Physics, 60 random targets: the network's grasp with the real arm (the
   nearer one). On the same targets, one random direction each, at the true
   place, checks the fast check against physics.

Everything goes to runs/<name>/evaluation.json and summary.json.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch
from detect_classify.camera import Calibration

from . import pick, scenes, side
from .net import GraspNet
from .train import device

CENTRED = 0.005
PHYSICS_TARGETS = 60


def target_of(sample: dict) -> dict:
    return next(t for t in sample["tubes"] if t["name"] == sample["target"])


def scores_of(net, heights: np.ndarray, dev: str, smooth: bool = False) -> np.ndarray:
    out = np.concatenate(
        [pick.model_scores(net, heights[i : i + 64], dev) for i in range(0, len(heights), 64)]
    )
    return side.smoothed(out) if smooth else out


def choose(scores: np.ndarray, heights: np.ndarray, sample: dict, threshold: float) -> dict | None:
    calib = Calibration.from_dict(sample["camera"])
    best, (u, v, k) = pick.learned(scores, heights, calib, threshold=0.0)
    if best is None or best.score < threshold:
        return None
    return {
        "xy": [float(x) for x in best.xy],
        "k": int(k),
        "direction": float(sample.get("frame_yaw", 0.0) + side.direction(k % side.BINS)),
        "below": side.HEIGHTS[k // side.BINS] if scores.shape[0] > side.BINS else side.BELOW_RIM,
        "top": best.top,
        "score": best.score,
    }


def best_threshold(net, folder: Path, dev: str, smooth: bool = False) -> tuple[float, float]:
    samples = json.loads((folder / "samples.json").read_text())
    heights = np.load(folder / "heights.npy").astype(np.float32)
    scores = scores_of(net, heights, dev, smooth)
    best = (0.0, 0.5)
    for threshold in np.arange(0.05, 0.71, 0.05):
        right = 0
        for s, sc, h in zip(samples, scores, heights, strict=True):
            target = target_of(s)
            calib = Calibration.from_dict(s["camera"])
            _, (u, v, k) = pick.learned(sc, h, calib)
            if any(target["labels"]):
                near = (u - target["u"]) ** 2 + (v - target["v"]) ** 2 <= 9
                right += int(sc[k, v, u] >= threshold and near and target["labels"][k])
            else:
                right += int(sc[k, v, u] < threshold)
        if right / len(samples) > best[0]:
            best = (right / len(samples), float(threshold))
    return best[1], best[0]


def _fast_job(args):
    seed, choices, preshape = args
    model, data, _ = scenes.build(seed)
    checker = side.SideChecker(model, data, preshape=preshape)
    return [
        (i, checker.clear(np.array(g["xy"]), g["direction"], g["top"], target, g["below"]))
        for i, target, g in choices
    ]


def _trial_job(args):
    key, seed, target, xy, direction, top, preshape, below = args
    return key, side.run(seed, target, np.array(xy), direction, top, preshape=preshape, below=below)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=Path("data_side"))
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    parser.add_argument("--physics", type=int, default=PHYSICS_TARGETS)
    parser.add_argument("--version", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--smooth", action="store_true", help="smooth the scores (on in version 2)")
    parser.add_argument("--out", type=Path, default=None, help="where to write; default: next to the weights")
    args = parser.parse_args()
    preshape = args.version >= 2
    smooth = args.smooth or args.version >= 2
    outputs = side.BINS * (len(side.HEIGHTS) if args.version == 3 else 1)
    start = time.time()
    dev = device()
    net = GraspNet(outputs).to(dev)
    net.load_state_dict(torch.load(args.weights, map_location=dev))
    net.eval()
    threshold, val_right = best_threshold(net, args.data / "val", dev, smooth)
    print(f"cut-off {threshold:.2f}, chosen on validation ({val_right:.1%} of targets right)", flush=True)

    samples = json.loads((args.data / "test" / "samples.json").read_text())
    heights = np.load(args.data / "test" / "heights.npy").astype(np.float32)
    scores = scores_of(net, heights, dev, smooth)
    rows = []
    for i, s in enumerate(samples):
        target = target_of(s)
        rows.append(
            {
                "i": i,
                "seed": s["seed"],
                "target": s["target"],
                "kind": s["kind"],
                "graspable": bool(any(target["labels"])),
                "clear_directions": int(sum(target["labels"])),
                "truth": {"xy": target["xy"], "top": target["top"]},
                "model": choose(scores[i], heights[i], s, threshold),
            }
        )

    by_seed: dict[int, list] = {}
    for r in rows:
        if r["model"] is not None:
            by_seed.setdefault(r["seed"], []).append((r["i"], r["target"], r["model"]))
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for result in pool.imap_unordered(_fast_job, [(s, c, preshape) for s, c in by_seed.items()]):
            for i, clear in result:
                g = rows[i]["model"]
                g["clear"] = bool(clear)
                g["error"] = float(np.linalg.norm(np.array(g["xy"]) - rows[i]["truth"]["xy"]))
                g["good"] = bool(clear and g["error"] < CENTRED)
    print(f"fast grading done, {time.time() - start:.0f} s", flush=True)

    rng = np.random.default_rng(0)
    chosen = rng.choice(len(rows), size=min(args.physics, len(rows)), replace=False)
    jobs = []
    for i in chosen:
        r = rows[i]
        g = r["model"]
        if g is not None:
            jobs.append(
                (
                    (int(i), "model"),
                    r["seed"],
                    r["target"],
                    g["xy"],
                    g["direction"],
                    g["top"],
                    preshape,
                    g["below"],
                )
            )
        k = int(rng.integers(outputs))
        r["check_direction"] = k
        t = r["truth"]
        a = float(samples[i].get("frame_yaw", 0.0) + side.direction(k % side.BINS))
        below = side.HEIGHTS[k // side.BINS] if outputs > side.BINS else side.BELOW_RIM
        jobs.append(((int(i), "check"), r["seed"], r["target"], t["xy"], a, t["top"], preshape, below))
    print(f"{len(jobs)} physics trials", flush=True)
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for n, ((i, what), result) in enumerate(pool.imap_unordered(_trial_job, jobs), start=1):
            r = rows[i]
            if what == "model":
                r["model"]["physics"] = result
            else:
                r["check"] = {
                    "label": bool(target_of(samples[i])["labels"][r["check_direction"]]),
                    "physics": result,
                }
            if n % 20 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} trials, {time.time() - start:.0f} s", flush=True)

    for r in rows:
        if r["model"] is not None and r["graspable"]:
            # Which of the target's own labels the chosen direction has: a wrong direction, or a right
            # one blocked only because the aim was off.
            labels = target_of(samples[r["i"]])["labels"]
            r["model"]["direction_labelled_clear"] = bool(labels[r["model"]["k"]])
    out = args.out or args.weights.parent
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation.json").write_text(json.dumps({"threshold": threshold, "rows": rows}, default=_plain))
    summary = summarise(rows, threshold, val_right)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"done in {(time.time() - start) / 60:.1f} min")


def _plain(x):
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(type(x))


def summarise(rows: list[dict], threshold: float, val_right: float) -> dict:
    clear = [r for r in rows if r["graspable"]]
    blocked = [r for r in rows if not r["graspable"]]
    picked = [r["model"] for r in rows if r["model"] is not None]
    errors = np.array([g["error"] for g in picked])
    out = {
        "threshold": threshold,
        "validation_right": val_right,
        "targets": len(rows),
        "graspable": len(clear),
        "good_on_graspable": float(np.mean([r["model"] is not None and r["model"]["good"] for r in clear])),
        "refused_graspable": float(np.mean([r["model"] is None for r in clear])),
        "refused_blocked": float(np.mean([r["model"] is None for r in blocked])) if blocked else None,
        # The pickable tubes, split by what the network did: the four add up to all of them.
        "on_graspable": {
            "good_grasp": sum(1 for r in clear if r["model"] is not None and r["model"]["good"]),
            "wrongly_refused": sum(1 for r in clear if r["model"] is None),
            "tried_a_blocked_direction": sum(
                1
                for r in clear
                if r["model"] is not None
                and not r["model"]["good"]
                and not r["model"]["clear"]
                and not r["model"]["direction_labelled_clear"]
            ),
            "aim_off": sum(
                1
                for r in clear
                if r["model"] is not None
                and not r["model"]["good"]
                and (r["model"]["clear"] or r["model"]["direction_labelled_clear"])
            ),
        },
        "boxed_in_tried": sum(1 for r in blocked if r["model"] is not None),
        # Version 3: how often each grip height was chosen (cm below the rim), and for which sizes.
        "heights_chosen": {
            size: {
                f"{round(b * 100)}cm": sum(
                    1
                    for r in rows
                    if r["model"] is not None and r["kind"].endswith(size) and r["model"]["below"] == b
                )
                for b in sorted({r["model"]["below"] for r in rows if r["model"] is not None})
            }
            for size in ("100", "150", "200")
        },
        "error_median_mm": float(np.median(errors) * 1000),
        "error_95_mm": float(np.percentile(errors, 95) * 1000),
    }
    for size in ("100", "150", "200"):
        some = [r for r in clear if r["kind"].endswith(size)]
        out[f"graspable_{size}mm"] = float(len(some) / max(1, sum(r["kind"].endswith(size) for r in rows)))
        out[f"good_{size}mm"] = float(np.mean([r["model"] is not None and r["model"]["good"] for r in some]))
    sample = [r for r in rows if "check" in r]
    tried = [r for r in sample if r["model"] is not None and "physics" in r["model"]]
    reachable = [r for r in tried if not r["model"]["physics"].get("unreachable")]
    out["physics"] = {
        "sample": len(sample),
        "graspable_in_sample": sum(r["graspable"] for r in sample),
        "tried": len(tried),
        "unreachable_for_the_nearer_arm": len(tried) - len(reachable),
        "success_of_reachable": float(np.mean([r["model"]["physics"]["success"] for r in reachable]))
        if reachable
        else None,
        "boxed_in_refused": sum(1 for r in sample if not r["graspable"] and r["model"] is None),
        "boxed_in": sum(1 for r in sample if not r["graspable"]),
    }
    checks = [r["check"] for r in sample if not r["check"]["physics"].get("unreachable")]
    out["fast_check_vs_physics"] = {
        "trials": len(checks),
        "agree": float(np.mean([c["label"] == c["physics"]["success"] for c in checks])) if checks else None,
        "label_clear_physics_clean": sum(c["label"] and c["physics"]["success"] for c in checks),
        "label_clear_physics_failed": sum(c["label"] and not c["physics"]["success"] for c in checks),
        "label_blocked_physics_failed": sum(not c["label"] and not c["physics"]["success"] for c in checks),
        "label_blocked_physics_clean": sum(not c["label"] and c["physics"]["success"] for c in checks),
        "unreachable": sum(1 for r in sample if r["check"]["physics"].get("unreachable")),
    }
    return out


if __name__ == "__main__":
    main()
