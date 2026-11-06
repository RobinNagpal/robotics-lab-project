"""The learned unscrew in the full uncapping run, on the stress scenes the teacher was tested on.

    python -m learn.evaluate --run runs/learn/act_A --seeds 200-229 --workers 4

Each run: the result as uncap/evaluate.py writes it (graded by the simulator: cap off and on the holder,
tube back in the rack, nothing touched, tube held), and Arm B's recording. Then both are measured the same
way (learn/measure.py) and set beside the teacher's recordings of the same scenes (--teacher).
Written to <run>/eval/.
"""

from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing as mp
import time
from pathlib import Path

from uncap.evaluate import seeds
from uncap.evaluate import summary as graded

_POLICY = None


def _job(args) -> dict:
    global _POLICY
    # The simulation first, then PyTorch (uncap/evaluate.py: "OMP: Error #15").
    from . import rollout

    importlib.import_module("torch").set_num_threads(1)
    run, seed, out, noise = args
    if _POLICY is None:
        _POLICY = rollout.Policy(run)
    name = f"seed{seed}_size200"
    try:
        result = rollout.run(_POLICY, seed, out, noise=noise)
    except Exception as e:  # one crash should not stop the others
        result = {"seed": seed, "size": 0.2, "failure": f"crashed: {e!r}", "done": False}
    result["run"] = name
    result.pop("log", None)
    (out / f"{name}.json").write_text(json.dumps(result, indent=1, default=str))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--seeds", default="200-229")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-noise", action="store_true")
    parser.add_argument("--teacher", type=Path, default=Path("runs/teacher_A_rec"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.run / "eval"
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(args.run, s, out, not args.no_noise) for s in seeds(args.seeds)]
    t0 = time.time()
    results = []
    with mp.get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
        for r in pool.imap_unordered(_job, jobs):
            results.append(r)
            line = {k: r.get(k) for k in ("run", "done", "cap_off", "cap_on_holder", "failure")}
            line["policy_s"] = r.get("policy_seconds")
            line["touched"] = r.get("touched")
            print(f"{time.time() - t0:.0f} s {json.dumps(line, default=str)}", flush=True)

    from . import measure

    heights = measure.Heights()
    learned = measure.folder(out, heights)
    report = {"graded": graded(results), "learned": measure.summary(learned), "runs": learned}
    wanted = {r["run"] for r in learned}
    if args.teacher.exists():
        teacher = [r for r in measure.folder(args.teacher, heights) if r["run"] in wanted]
        report["teacher"] = measure.summary(teacher)
    (out / "summary.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}, indent=1, default=str))


if __name__ == "__main__":
    main()
