"""Grade every pour the same way: from the same saved starts, to the same target, by the same count.

    python -m act.evaluate --methods programmed programmed-cnn act mlp --workers 4

Methods
    programmed       Step 2's programmed pour, reading the true ball count (a perfect scale): the expert
                     the policies copy, and the best they can hope to match
    programmed-cnn   the same pour, reading Step 2's level network instead (step-2-pour/level/reader.py)
    <run name>       a learned pour from runs/<name> (act/rollout.py); ``<name>@ens<m>`` uses ACT's
                     temporal ensembling with weight m

Starts: the test scenes (seeds 150 to 159, medium and large), never used to train or choose anything.
Each pour's result goes to runs/evaluation/<method>/<start>_<target>.json (and, for the learned ones,
what it did every tick, .npz), so an interrupted evaluation carries on where it stopped.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from demos.collect import SPLITS, STARTS

OUT = Path(__file__).resolve().parents[1] / "runs" / "evaluation"


def _job(job: tuple[str, Path, float, Path]) -> dict:
    import torch

    torch.set_num_threads(1)
    method, start_path, target, out = job
    if method.startswith("programmed"):
        from pour import pour

        reader = None
        if method == "programmed-cnn":
            from level.reader import NetReader

            reader = NetReader("custom")
        r, _ = pour.run(start_path, target, record=False, reader=reader)
        r = {k: v for k, v in r.items() if k != "log"}
        r["error_balls"] = r["balls"]["receiver"] - r["target_balls"]
        r["policy"] = method
        trace = None
    else:
        from .policy import Policy
        from .rollout import run

        name, _, ens = method.partition("@ens")
        r, trace = run(start_path, Policy(name, ensemble=float(ens) if ens else None), target)
        r["policy"] = method
    stem = f"{start_path.stem}_{target:g}"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stem}.json").write_text(json.dumps(r, default=str))
    if trace is not None:
        np.savez_compressed(out / f"{stem}.npz", **{k: np.array(v) for k, v in trace.items()})
    return r


def summary(results: list[dict]) -> dict:
    ok = [r for r in results if not r.get("failure")]
    err = np.array([r["error_balls"] for r in results], float)
    level_err = np.array([abs(r["level_pct"] - r["target_pct"]) for r in results])
    spilled = np.array([r["balls"]["spilled"] for r in results])
    return {
        "pours": len(results),
        "within_window": int(sum(r["within_window"] for r in results)),
        "failed": len(results) - len(ok),
        "mean_abs_error_balls": round(float(np.abs(err).mean()), 2),
        "mean_abs_error_points": round(float(level_err.mean()), 2),
        "mean_error_balls": round(float(err.mean()), 2),
        "pours_with_spill": int((spilled > 0).sum()),
        "spilled": int(spilled.sum()),
        "median_seconds": round(float(np.median([r["sim_seconds"] for r in results])), 1),
        "refused": int(sum(r.get("refused") or 0 for r in results)),
    }


def load(method: str, out: Path = OUT) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((out / method).glob("*.json"))]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--methods", nargs="+", default=["programmed", "programmed-cnn", "act", "mlp"])
    parser.add_argument("--split", default="test", choices=list(SPLITS))
    parser.add_argument("--targets", type=float, nargs="+", default=[50.0])
    parser.add_argument("--starts", type=Path, default=STARTS)
    parser.add_argument("--workers", type=int, default=4, help="at most 4 on this Mac (CLAUDE.md)")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    starts = [
        p
        for p in sorted(args.starts.glob("*.npy"))
        if json.loads(p.with_suffix(".json").read_text())["seed"] in SPLITS[args.split]
    ]
    jobs = [
        (m, p, t, args.out / m)
        for m in args.methods
        for p in starts
        for t in args.targets
        if not (args.out / m / f"{p.stem}_{t:g}.json").exists()
    ]
    print(f"{len(jobs)} pours to run ({len(starts)} starts), {args.workers} at a time", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs), start=1):
            print(
                f"{i}/{len(jobs)} {time.time() - t0:.0f} s {r['policy']:>16} {r['start']} "
                f"target {r['target_pct']:g} -> {r['level_pct']} ({r['error_balls']:+d} balls, "
                f"spilled {r['balls']['spilled']}) {r.get('failure') or ''}",
                flush=True,
            )
    table = {m: summary(load(m, args.out)) for m in args.methods if load(m, args.out)}
    (args.out / "summary.json").write_text(json.dumps(table, indent=1))
    keys = list(next(iter(table.values())))
    print(f"\n{'':>16} " + " ".join(f"{k[:14]:>14}" for k in keys))
    for m, row in table.items():
        print(f"{m:>16} " + " ".join(f"{row[k]!s:>14}" for k in keys))


if __name__ == "__main__":
    main()
