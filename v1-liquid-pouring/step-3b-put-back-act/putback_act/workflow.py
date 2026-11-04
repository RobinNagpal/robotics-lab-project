"""The whole task on one scene with the learned pour: Step 0's table, Step 1b's pick and carry, the pour
by a policy from Step 2b (ACT by default), the source back in its cell (Step 3), the receiver in the
destination rack (Step 4), and the check (Step 5).

Everything but the pour is Step 3's code, unchanged (step-3-put-back/putback). As there, Step 1b is not
run again: each scene starts from a saved Step 1b run (step-2-pour/data/starts), and Step 0 is run again
on the scene as it was before anything moved.

The one difference after the pour: Step 3 takes the source out the way the programmed pour brought it
in, backwards. The learned pour has no planned way in (the policy moved the arm). So the programmed
pour's way in is planned here for the same start, with no physics, and the source leaves by that,
after the same straight lift. The policy copies that way in closely, and every move out is checked
for collisions before Arm B makes it, as in Step 3.

    python -m putback_act.workflow --starts seed150_size200 seed151_size150 --policy act --workers 4
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

from act import rollout
from act.policy import Policy
from carry.motion import Blocked
from carry.perceive import Perception
from pour import start as starts
from pour.pour import move_in_path
from pour.tilt import Tilter
from putback.workflow import STARTS, put_back, step0


def way_in(path: Path) -> list:
    """The programmed pour's way in for this start (Arm B's joints, from the end of Step 1b), planned
    on a fresh copy of the start with no physics. Step 3 leaves the pour by it, backwards."""
    model, data, containers, meta = starts.load(path)
    source, receiver = meta["source"], meta["receiver"]
    tilter = Tilter(model, data, source, containers[source].height, containers[source].radius)
    path_q, _ = move_in_path(tilter, data, source, receiver, containers[receiver].height)
    return [data.qpos[tilter.joints].copy(), *path_q]


def run_one(start: Path, policy: Policy, target: float = 50.0, perception=None) -> dict:
    """The whole task from one saved Step 1b run, with ``policy`` pouring."""
    wall = time.time()
    meta = json.loads(start.with_suffix(".json").read_text())
    perception = perception or Perception()
    seen = step0(meta["seed"], meta["source"], perception)
    sim: dict = {}
    poured, _ = rollout.run(start, policy, target, sim=sim)
    try:
        sim["move_in"] = way_in(start)
    except Blocked as e:  # never seen: the programmed pour plans this same way in from every start
        poured["failure"] = poured["failure"] or f"no way out: {e}"
    result = put_back(start, meta, seen, poured, sim, wall=wall)
    result["method"] = policy.name
    return result


def _job(args) -> dict:
    import torch

    torch.set_num_threads(1)
    start, target, name, out = args
    try:
        result = run_one(start, Policy(name), target)
    except Exception as e:  # a crash in one scene should not stop the others
        result = {"start": start.stem, "failure": f"crashed: {e!r}", "done": False}
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{start.stem}.json").write_text(json.dumps(result, indent=1, default=str))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--starts", nargs="+", required=True, help="saved Step 1b runs, e.g. seed150_size200")
    parser.add_argument("--policy", default="act", help="a trained run in step-2b-pour-act/runs/")
    parser.add_argument("--target", type=float, default=50.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("runs/workflow"))
    args = parser.parse_args()
    jobs = [(STARTS / f"{s}.npy", args.target, args.policy, args.out / args.policy) for s in args.starts]
    t0 = time.time()
    with mp.get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
        for r in pool.imap_unordered(_job, jobs):
            line = {k: r.get(k) for k in ("start", "done", "level_pct", "failure", "spilled")}
            line["pour"] = {k: r.get("pour", {}).get(k) for k in ("failure", "end_tilt_deg", "sim_seconds")}
            line["source"] = {k: r.get("source", {}).get(k) for k in ("in_cell", "own_cell", "off_mm")}
            line["receiver"] = {k: r.get("receiver", {}).get(k) for k in ("in_cell", "off_mm")}
            print(f"{time.time() - t0:.0f} s {json.dumps(line)}", flush=True)


if __name__ == "__main__":
    main()
