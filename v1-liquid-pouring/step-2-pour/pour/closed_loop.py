"""The pour with the level network reading the receiver, against the same pour reading the true count.

Every saved start of the test scenes (seeds 150 to 159, never used to train or choose the network) is
poured to 50 % three ways, each graded by the true count:

    true      the programmed pour: reads the true ball count, as a perfect scale would
    cnn-flow  reads the custom CNN; tips back when level + flow x 0.6 s reaches the mark (the same rule)
    cnn-air   reads the custom CNN; tips back when level + balls in the air reaches the mark

    python -m pour.closed_loop --workers 4
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from . import pour

WAYS = {"true": (None, "flow"), "cnn-flow": ("custom", "flow"), "cnn-air": ("custom", "air")}


def _job(args) -> dict:
    import torch

    torch.set_num_threads(1)
    start, way, target = args
    net, rule = WAYS[way]
    reader = None
    if net is not None:
        from level.reader import NetReader

        reader = NetReader(net)
    result, _ = pour.run(start, target, record=False, reader=reader, rule=rule)
    result.pop("settings", None)
    result["way"] = way
    return result


def summary(results: list[dict]) -> dict:
    out = {}
    for way in WAYS:
        rs = [r for r in results if r["way"] == way]
        if not rs:
            continue
        err = np.array([r["level_pct"] - r["target_pct"] for r in rs if r["failure"] is None])
        out[way] = {
            "pours": len(rs),
            "failed": sum(r["failure"] is not None for r in rs),
            "within 2 points": int(sum(r["within_window"] for r in rs)),
            "mean error (points)": round(float(np.abs(err).mean()), 2) if len(err) else None,
            "worst error (points)": round(float(np.abs(err).max()), 2) if len(err) else None,
            "spilled balls": int(sum(r["balls"]["spilled"] for r in rs)),
            "top-ups": int(sum(r["topups"] for r in rs)),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--starts", type=Path, default=Path("data/starts"))
    parser.add_argument("--seeds", type=int, nargs=2, default=(150, 159), help="first and last seed")
    parser.add_argument("--ways", nargs="+", default=list(WAYS), choices=list(WAYS))
    parser.add_argument("--target", type=float, default=50.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("runs/closed_loop"))
    args = parser.parse_args()
    starts = sorted(
        p
        for p in args.starts.glob("seed*_size*.npy")
        if args.seeds[0] <= int(p.stem.split("_")[0][4:]) <= args.seeds[1]
    )
    jobs = [(s, way, args.target) for s in starts for way in args.ways]
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for r in pool.imap_unordered(_job, jobs):
            results.append(r)
            print(
                f"{len(results)}/{len(jobs)} {time.time() - t0:.0f} s {r['way']:8s} {r['start']} "
                f"-> {r['level_pct']} % ({r['failure'] or 'ok'})",
                flush=True,
            )
    (args.out / "pours.json").write_text(json.dumps(results, indent=1, default=str))
    s = summary(results)
    (args.out / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
