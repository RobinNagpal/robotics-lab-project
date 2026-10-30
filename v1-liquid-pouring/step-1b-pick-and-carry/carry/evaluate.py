"""Run Step 1b on many task scenes, for each tube size, and sum up.

    python -m carry.evaluate --seeds 10

Each run asks for one size first (100, 150 or 200 mm), so every size is
tried in every scene. Results go to runs/evaluation.json; the summary is
printed and written to runs/summary.json.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

SIZES = (0.1, 0.15, 0.2)
# A run counts as a success when everything below holds.
POSITION = 5.0  # mm, each tube's rim from where it should be
UPRIGHT = 5.0  # degrees


def _job(args):
    seed, size = args
    import torch

    torch.set_num_threads(1)
    from .episode import load_net, run_episode
    from .perceive import Perception

    global _tools
    if "_tools" not in globals():
        _tools = (Perception(), load_net())
    order = [size] + [s for s in SIZES if s != size]
    return run_episode(seed, order, *_tools)


def success(r: dict) -> bool:
    return bool(
        r.get("done")
        and r["pair_matches"]
        and r["picked_from_right_racks"]
        and r["receiver_error_mm"] < POSITION
        and r["source_error_mm"] < POSITION
        and max(r["lean_deg"].values()) < UPRIGHT
        and r["balls_lost"] == 0
        and not any(r["touched"].values())
    )


def times(r: dict) -> dict[str, float]:
    """When each message came, by what it means: the receiver arm's and the source arm's."""
    from .episode import RECEIVER_ARM, SOURCE_ARM

    at = {(arm, msg): t for t, arm, msg in r["log"]}
    out = {}
    for key, (arm, msg) in {
        "source_picked": (SOURCE_ARM, "picked"),
        "receiver_picked": (RECEIVER_ARM, "picked"),
        "receiver_ready": (RECEIVER_ARM, "ready"),
        "source_in_place": (SOURCE_ARM, "in place"),
    }.items():
        if (arm, msg) in at:
            out[key] = at[(arm, msg)]
    return out


def summarise(results: list[dict]) -> dict:
    ok = [success(r) for r in results]
    done = [r for r in results if r.get("done")]
    out = {
        "runs": len(results),
        "success": int(sum(ok)),
        "failures": [
            {k: r.get(k) for k in ("seed", "size", "failure", "touched", "lean_deg", "balls_lost")}
            for r, good in zip(results, ok, strict=True)
            if not good
        ],
        "by_size": {
            str(s): f"{sum(g for r, g in zip(results, ok, strict=True) if r.get('size') == s)}"
            f" of {sum(r.get('size') == s for r in results)}"
            for s in SIZES
        },
        "asked_size_used": int(sum(not r.get("sizes_tried_first") for r in results)),
    }
    if done:
        out["receiver_error_mm"] = _spread([r["receiver_error_mm"] for r in done])
        out["source_error_mm"] = _spread([r["source_error_mm"] for r in done])
        out["meet_distance_mm"] = _spread([r["meet_distance_mm"] for r in done])
        out["lean_deg_max"] = round(max(max(r["lean_deg"].values()) for r in done), 2)
        out["balls_lost"] = int(sum(r["balls_lost"] for r in done))
        for key in ("source_picked", "receiver_picked", "receiver_ready", "source_in_place"):
            out[f"{key}_s"] = _spread([times(r)[key] for r in done if key in times(r)])
        out["wall_seconds_per_run"] = round(float(np.mean([r["wall_seconds"] for r in done])), 1)
    return out


def _spread(values: list[float]) -> dict[str, float]:
    v = np.array(values, float)
    return {"median": round(float(np.median(v)), 2), "max": round(float(v.max()), 2)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    parser.add_argument("--out", type=Path, default=Path("runs"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(seed, size) for seed in range(args.seeds) for size in SIZES]
    start, results = time.time(), []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs), start=1):
            results.append(r)
            print(
                f"{i}/{len(jobs)} runs, {sum(map(success, results))} successful, {time.time() - start:.0f} s",
                flush=True,
            )
    results.sort(key=lambda r: (r["seed"], r.get("size") or 0))
    (args.out / "evaluation.json").write_text(json.dumps(results, indent=1, default=str))
    summary = summarise(results)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
