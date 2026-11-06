"""Run the programmed uncapping on many scenes and sum it up.

    python -m uncap.evaluate --seeds 150-159 --sizes 0.15 0.2 --workers 4

Each run is written to runs/uncap/seed<seed>_size<mm>.json; the summary to runs/uncap/summary.json.
"""

from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing as mp
import time
from pathlib import Path


def seeds(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def _job(args) -> dict:
    # The simulation's libraries first, then PyTorch: the other way round, macOS loads two copies of
    # OpenMP and the worker stops ("OMP: Error #15"). (A single run imports them in this order.)
    from .episode import run_episode

    importlib.import_module("torch").set_num_threads(1)

    seed, size, out, vary, vary_arms, record_to, test_from, pause = args
    name = f"seed{seed}_size{round(size * 1000)}"
    try:
        result = run_episode(
            seed,
            [size],
            vary=seed if vary else None,
            vary_arms=vary_arms,
            record_to=record_to,
            test_from=test_from,
            pause=pause,
        )
    except Exception as e:  # one crash should not stop the others
        result = {"seed": seed, "size": size, "failure": f"crashed: {e!r}", "done": False}
    result["run"] = name
    (out / f"{name}.json").write_text(json.dumps(result, indent=1, default=str))
    return result


def _stats(values: list) -> dict:
    if not values:
        return {}
    return {"mean": round(sum(values) / len(values), 2), "min": min(values), "max": max(values)}


def summary(results: list[dict]) -> dict:
    n = len(results)
    off = [r for r in results if r.get("cap_off")]
    quarters = [r["quarters"] for r in off]
    return {
        "runs": n,
        "done": sum(bool(r.get("done")) for r in results),
        "cap_off": len(off),
        "cap_on_holder": sum(bool(r.get("cap_on_holder")) for r in results),
        "tube_in_rack": sum(bool((r.get("tube_in_rack") or {}).get("in_cell")) for r in results),
        "nothing_touched": sum(not r.get("touched") for r in results if "touched" in r),
        "tube_held": sum(bool(r.get("tube_held")) for r in results),
        "off_said_too_early": sum(bool(r.get("off_said_too_early")) for r in results),
        "quarters": {"min": min(quarters, default=None), "max": max(quarters, default=None)},
        "unscrew_seconds": _stats([r["unscrew_seconds"] for r in results if r.get("unscrew_seconds")]),
        "lift_tests": _stats([sum(f is not None for f in r.get("lift_forces_N") or []) for r in off]),
        "failures": sorted({r["run"]: r.get("failure") for r in results if r.get("failure")}.items()),
        "touched": sorted({r["run"]: r.get("touched") for r in results if r.get("touched")}.items()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seeds", default="150-159")
    parser.add_argument("--sizes", type=float, nargs="+", default=[0.15, 0.2])  # no 100 mm (episode.SIZES)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("runs/uncap"))
    parser.add_argument("--vary", action="store_true", help="stress test: tube placement and cap tightness")
    parser.add_argument("--vary-arms", action="store_true", help="with --vary: the arms' start pose too")
    parser.add_argument("--record", type=Path, default=None, help="save Arm B's unscrewing here (record.py)")
    parser.add_argument("--test-from", type=int, default=1, help="first quarter with a lift test (6: safe)")
    parser.add_argument("--pause", type=float, default=0.0, help="s held still after untested quarters")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [
        (s, size, args.out, args.vary, args.vary_arms, args.record, args.test_from, args.pause)
        for s in seeds(args.seeds)
        for size in args.sizes
    ]
    t0 = time.time()
    results = []
    with mp.get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
        for r in pool.imap_unordered(_job, jobs):
            results.append(r)
            line = {k: r.get(k) for k in ("run", "done", "cap_off", "cap_on_holder", "quarters", "failure")}
            line["in_rack"] = (r.get("tube_in_rack") or {}).get("in_cell")
            line["touched"] = r.get("touched")
            print(f"{time.time() - t0:.0f} s {json.dumps(line, default=str)}", flush=True)
    s = summary(results)
    (args.out / "summary.json").write_text(json.dumps(s, indent=1, default=str))
    print(json.dumps(s, indent=1, default=str))


if __name__ == "__main__":
    main()
