"""Grade the programmed rule and the model on the same test pictures, in three parts.

    python -m grasp.evaluate --weights runs/grconv/best.pt

1. Fast grading, every test target: each method's grasp is tried with the
   fast check at the place it chose (not the true place). A grasp is good
   when the way down is clear and the pinch point is within 5 mm of the
   tube's axis. A target with no clear turn at all should be refused.
2. Physics, a sample of targets: each method's grasp is run with the real arm
   (trial.py). On the same sample, one random turn per target checks how
   often the fast check agrees with full physics.
3. Squeeze: the heaviest tubes, full of liquid, are lifted with the fingers told to close
   0.5 to 4 mm under the tube's width, to see how hard they press and
   whether the tube slips.

The programmed rule's footprint is fitted on the training pictures first.
Everything goes to runs/<name>/evaluation.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch
from detect_classify.camera import Calibration

from . import pick, rules, scenes, trial
from .net import GraspNet
from .train import device

CENTRED = 0.005
PHYSICS_TARGETS = 60
SQUEEZES = (0.0005, 0.001, 0.002, 0.004)
SQUEEZE_TUBES = 3


def target_of(sample: dict) -> dict:
    return next(t for t in sample["tubes"] if t["name"] == sample["target"])


def fit_footprint(samples: list[dict]) -> pick.Footprint:
    """The rectangle size that makes the rule agree most with the training labels."""
    best, best_right = None, -1
    for reach, half in itertools.product(np.arange(0.040, 0.0801, 0.004), np.arange(0.006, 0.0301, 0.002)):
        fp = pick.Footprint(float(reach), float(half))
        right = 0
        for s in samples:
            labels = target_of(s)["labels"]
            g = pick.programmed(s["step0"], s["scene_step0"], fp)
            right += int(not any(labels)) if g is None else int(bool(labels[rules.yaw_bin(g.yaw)]))
        if right > best_right:
            best, best_right = fp, right
    print(
        f"footprint fitted on {len(samples)} training pictures: {best}, {best_right / len(samples):.1%} agree"
    )
    return best


def _fast_job(args):
    """Fast-check every choice in one scene."""
    seed, choices = args
    model, data, _ = scenes.build(seed)
    checker = scenes.Checker(model, data)
    out = []
    for i, method, target, g in choices:
        clear = checker.clear(np.array(g["xy"]), g["yaw"], g["top"], target)
        out.append((i, method, clear))
    return out


def _trial_job(args):
    key, seed, target, xy, yaw, top, squeeze, full = args
    return key, trial.run(seed, target, np.array(xy), yaw, top, squeeze, liquid_in=full)


def _grasp_dict(g: pick.Grasp | None) -> dict | None:
    if g is None:
        return None
    return {"xy": [float(v) for v in g.xy], "yaw": float(g.yaw), "top": float(g.top), "score": float(g.score)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    parser.add_argument("--physics", type=int, default=PHYSICS_TARGETS, help="targets run with full physics")
    args = parser.parse_args()
    start = time.time()

    train = json.loads((args.data / "train" / "samples.json").read_text())
    footprint = fit_footprint(train[::3])
    samples = json.loads((args.data / "test" / "samples.json").read_text())
    heights = np.load(args.data / "test" / "heights.npy").astype(np.float32)

    dev = device()
    net = GraspNet().to(dev)
    net.load_state_dict(torch.load(args.weights, map_location=dev))
    net.eval()
    scores = np.concatenate(
        [pick.model_scores(net, heights[i : i + 64], dev) for i in range(0, len(heights), 64)]
    )

    rows = []
    for i, s in enumerate(samples):
        target = target_of(s)
        calib = Calibration.from_dict(s["camera"])
        learned, _ = pick.learned(scores[i], heights[i], calib)
        rows.append(
            {
                "i": i,
                "seed": s["seed"],
                "target": s["target"],
                "kind": s["kind"],
                "graspable": bool(any(target["labels"])),
                "clear_turns": int(sum(target["labels"])),
                "truth": {"xy": target["xy"], "top": target["top"]},
                "programmed": _grasp_dict(pick.programmed(s["step0"], s["scene_step0"], footprint)),
                "model": _grasp_dict(learned),
            }
        )

    # 1. Fast grading.
    by_seed: dict[int, list] = {}
    for r in rows:
        for method in ("programmed", "model"):
            if r[method] is not None:
                by_seed.setdefault(r["seed"], []).append((r["i"], method, r["target"], r[method]))
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for result in pool.imap_unordered(_fast_job, list(by_seed.items())):
            for i, method, clear in result:
                g = rows[i][method]
                g["clear"] = bool(clear)
                g["error"] = float(np.linalg.norm(np.array(g["xy"]) - rows[i]["truth"]["xy"]))
                g["good"] = bool(clear and g["error"] < CENTRED)
    print(f"fast grading done, {time.time() - start:.0f} s", flush=True)

    # 2 and 3. Physics.
    rng = np.random.default_rng(0)
    chosen = rng.choice(len(rows), size=min(args.physics, len(rows)), replace=False)
    jobs = []
    for i in chosen:
        r = rows[i]
        for method in ("programmed", "model"):
            g = r[method]
            if g is not None:
                jobs.append(
                    ((int(i), method), r["seed"], r["target"], g["xy"], g["yaw"], g["top"], 0.004, False)
                )
        k = int(rng.integers(rules.BINS))
        r["check_bin"] = k
        t = r["truth"]
        jobs.append(
            ((int(i), "check"), r["seed"], r["target"], t["xy"], rules.bin_yaw(k), t["top"], 0.004, False)
        )
    # The heaviest tubes: sources in the tallest tubes, at their first clear turn.
    heavy = [
        r for r in rows if r["kind"].endswith("200") and r["target"].startswith("source") and r["graspable"]
    ]
    for r in heavy[:SQUEEZE_TUBES]:
        k = int(np.argmax(target_of(samples[r["i"]])["labels"]))
        for sq in SQUEEZES:
            t = r["truth"]
            jobs.append(
                (
                    (r["i"], f"squeeze_{sq}"),
                    r["seed"],
                    r["target"],
                    t["xy"],
                    rules.bin_yaw(k),
                    t["top"],
                    sq,
                    True,
                )
            )
    print(f"{len(jobs)} physics trials", flush=True)
    physics: dict[str, dict] = {}
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for n, ((i, what), result) in enumerate(pool.imap_unordered(_trial_job, jobs), start=1):
            physics[f"{i}/{what}"] = result
            if n % 20 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} trials, {time.time() - start:.0f} s", flush=True)
    for key, result in physics.items():
        i, what = key.split("/")
        r = rows[int(i)]
        if what in ("programmed", "model"):
            r[what]["physics"] = result
        elif what == "check":
            target = target_of(samples[int(i)])
            r["check"] = {"label": bool(target["labels"][r["check_bin"]]), "physics": result}
        else:
            r.setdefault("squeeze", {})[what.split("_")[1]] = result

    out = args.weights.parent / "evaluation.json"
    out.write_text(json.dumps({"footprint": vars(footprint), "rows": rows}, default=_plain))
    summary = summarise(rows)
    (args.weights.parent / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"done in {(time.time() - start) / 60:.1f} min; details in {out}")


def _plain(x):
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(type(x))


def summarise(rows: list[dict]) -> dict:
    out: dict = {"targets": len(rows), "graspable": sum(r["graspable"] for r in rows)}
    for method in ("programmed", "model"):
        clear = [r for r in rows if r["graspable"]]
        blocked = [r for r in rows if not r["graspable"]]
        picked = [r[method] for r in rows if r[method] is not None]
        errors = np.array([g["error"] for g in picked])
        m = {
            "good_on_graspable": np.mean([r[method] is not None and r[method]["good"] for r in clear]),
            "clear_on_graspable": np.mean([r[method] is not None and r[method]["clear"] for r in clear]),
            "refused_graspable": np.mean([r[method] is None for r in clear]),
            "refused_blocked": np.mean([r[method] is None for r in blocked]) if blocked else None,
            "error_median_mm": float(np.median(errors) * 1000),
            "error_95_mm": float(np.percentile(errors, 95) * 1000),
            "within_5mm": float(np.mean(errors < CENTRED)),
        }
        for size in ("100", "150", "200"):
            some = [r for r in clear if r["kind"].endswith(size)]
            m[f"good_{size}mm"] = np.mean([r[method] is not None and r[method]["good"] for r in some])
        ran = [r[method]["physics"] for r in rows if r[method] is not None and "physics" in r[method]]
        in_sample = [r for r in rows if "check" in r]
        m["physics_trials"] = len(ran)
        m["physics_success_of_sample"] = sum(
            r[method] is not None and r[method].get("physics", {}).get("success", False) for r in in_sample
        ) / max(len(in_sample), 1)
        m["physics_success_of_tried"] = float(np.mean([p["success"] for p in ran])) if ran else None
        out[method] = {k: (float(v) if v is not None else None) for k, v in m.items()}
    checks = [r["check"] for r in rows if "check" in r]
    agree = [c["label"] == c["physics"]["success"] for c in checks]
    out["fast_check_vs_physics"] = {
        "trials": len(checks),
        "agree": float(np.mean(agree)) if checks else None,
        "label_clear_physics_clean": sum(c["label"] and c["physics"]["success"] for c in checks),
        "label_clear_physics_touched": sum(c["label"] and not c["physics"]["success"] for c in checks),
        "label_blocked_physics_touched": sum(not c["label"] and not c["physics"]["success"] for c in checks),
        "label_blocked_physics_clean": sum(not c["label"] and c["physics"]["success"] for c in checks),
    }
    squeeze = {}
    for r in rows:
        for sq, p in r.get("squeeze", {}).items():
            squeeze.setdefault(sq, []).append(p)
    out["squeeze"] = {
        sq: {
            "lifted": float(np.mean([p.get("lifted", False) for p in ps])),
            "force_N": float(np.mean([p.get("squeeze", np.nan) for p in ps])),
            "slip_mm": float(np.mean([p.get("slip", np.nan) for p in ps]) * 1000),
            "weight_N": float(np.mean([p.get("weight", np.nan) for p in ps]) * 9.81),
        }
        for sq, ps in sorted(squeeze.items())
    }
    return out


if __name__ == "__main__":
    main()
