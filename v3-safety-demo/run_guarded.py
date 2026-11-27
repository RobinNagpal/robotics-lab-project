"""v3's whole job with the safety layer on: Step 6's ``run_seed``, unchanged, with its arm going through the
guard.

v3's ``step-6-whole-run/run.py`` is loaded as it is. Two things are wrapped from outside, without changing a
v3
file: ``scene.step`` (``guard.install``) and ``test_scenes.build``, so the scene ``run_seed`` builds gets its
guard (a truth camera, the perfect detector, a safety log) the moment it is made.

Step 0 (SAM 3 finding the tubes) moves nothing, so the guard has nothing to do there. Rather than run the 3.4
GB
model again, its answer is taken as the bench's truth, which is what SAM 3 gave on all 20 seeds
(``step0_right`` in v3's results).

    --hand none      no hand: the run must match v3's own results exactly
    --hand scripted  a hand beside the gripper every 30 s: 4 s in the warning zone, then 2 s in the stop zone
    --limits loose   the normal layer's limits wide open (to measure what v3's jobs need, to set the limits)

    cd v3-safety-demo
    PY=../v3-centrifuge-loading/simulation-environment/.pixi/envs/default/bin/python
    $PY run_guarded.py --seeds 14 6 --workers 2 --hand none --limits loose --name check
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
V3 = HERE.parent / "v3-centrifuge-loading"
STEPS = [
    "step-0-find",
    "step-1-weigh",
    "step-2-plan",
    "step-3-fill-balance-tube",
    "step-4-load-rotor",
    "step-5-close-and-report",
    "simulation-environment",
]
for p in [HERE, *(V3 / s for s in STEPS)]:
    if str(p) not in sys.path:
        sys.path.append(str(p))

OUT = HERE / "results"
V3_RESULTS = V3 / "step-6-whole-run" / "results" / "whole_run.json"
# Fields that differ from run to run whatever happens (wall-clock times).
NOT_COMPARED = {"wall_seconds", "step0_seconds", "seed"}
VISIT_EVERY, VISIT_FIRST = 30.0, 10.0


def v3_whole_run():
    spec = importlib.util.spec_from_file_location("v3_whole_run", V3 / "step-6-whole-run" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def step0_from_truth(seed: int) -> dict:
    """Step 0's answer in run.py's form, from the bench as built (what SAM 3 found on every seed)."""
    from centrifuge_scene import scene
    from centrifuge_scene.settings import SAMPLE_COLOURS

    names = ("red", "yellow", "green", "purple", "orange")  # as run.find_tubes
    _, truth = scene.build(seed)
    cells = {}
    for name, (rack, cell) in truth.cells.items():
        if name in truth.samples:
            cells[f"{rack}/{cell}"] = {
                "kind": "sample",
                "colour": names[SAMPLE_COLOURS.index(truth.samples[name].colour)],
            }
        elif name.startswith("water"):
            cells[f"{rack}/{cell}"] = {"kind": "water", "colour": "blue"}
        else:
            cells[f"{rack}/{cell}"] = {"kind": "empty", "colour": "none"}
    return {"cells": cells, "seconds": 0.0, "right": True}


def job(args: tuple[int, str, str, str]) -> dict:
    seed, hand_kind, limits_kind, name = args
    import test_scenes

    import guard as G
    from detector import PerfectDetector, TruthCamera
    from safety_log import SafetyLog, read, verify
    from standin import ScriptedHand, repeating_visits

    run = v3_whole_run()
    limits = G.loose() if limits_kind == "loose" else G.Limits()
    hand = (
        ScriptedHand(repeating_visits(VISIT_FIRST, VISIT_EVERY, 3600.0)) if hand_kind == "scripted" else None
    )
    log_path = OUT / name / f"log_seed{seed:02d}.jsonl"
    log = SafetyLog(log_path)
    made: dict = {}
    plain_build = test_scenes.build

    def build(seed_: int, changed: bool):
        model, data, truth = plain_build(seed_, changed)
        camera = TruthCamera(hand)
        made["camera"] = camera
        made["guard"] = G.attach(
            G.Guard(model, data, camera, PerfectDetector(), log, limits=limits, trace=True)
        )
        return model, data, truth

    t0 = time.time()
    test_scenes.build = build
    G.install()
    try:
        out = run.run_seed((seed, step0_from_truth(seed)))
    except G.SafetyStop as e:
        out = {"seed": seed, "error": f"SafetyStop: {e}", "success": False}
        made["guard"].hold(1.0)
    except Exception as e:  # noqa: BLE001 - recorded, so one bad seed does not end the batch
        out = {
            "seed": seed,
            "error": f"{type(e).__name__}: {e}",
            "success": False,
            "trace": traceback.format_exc(),
        }
    finally:
        test_scenes.build = plain_build
        G.uninstall()
        log.close()
    g = made.get("guard")
    out["wall_seconds"] = round(time.time() - t0, 1)
    if g is not None:
        out["guard"] = g.summary()
        cam = made["camera"]
        out["guard"]["min_hand_distance"] = (
            None if cam.min_distance == float("inf") else round(cam.min_distance, 4)
        )
        entries = read(log_path)
        out["guard"]["log_problems"] = verify(entries, head=g.log.head)
        out["guard"]["replay_disagrees"] = G.replay(entries)[:5]
        (OUT / name / f"trace_seed{seed:02d}.json").write_text(json.dumps(g.trace))
    return out


def compare(row: dict, v3_rows: dict[int, dict]) -> list[str]:
    """Where a guarded run's result differs from v3's own run of the same seed."""
    ref = v3_rows.get(row["seed"])
    if ref is None:
        return ["no v3 result for this seed"]
    row = json.loads(json.dumps(row, default=str))  # as v3's row went through JSON (dict keys become text)
    diffs = []
    for k, v in ref.items():
        if k in NOT_COMPARED:
            continue
        if row.get(k) != v:
            diffs.append(f"{k}: v3 {v!r}, guarded {row.get(k)!r}")
    return diffs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    p.add_argument("--workers", type=int, default=1, help="at most 2 while Track A runs too")
    p.add_argument("--hand", choices=("none", "scripted"), default="none")
    p.add_argument("--limits", choices=("set", "loose"), default="set")
    p.add_argument("--name", default=None)
    a = p.parse_args()
    name = a.name or f"guarded_{a.hand}"
    (OUT / name).mkdir(parents=True, exist_ok=True)
    v3_rows = {r["seed"]: r for r in json.loads(V3_RESULTS.read_text())}
    rows = []
    with ProcessPoolExecutor(a.workers) as pool:
        for r in pool.map(job, [(s, a.hand, a.limits, name) for s in a.seeds]):
            if a.hand == "none":
                r["same_as_v3"] = compare(r, v3_rows)
            rows.append(r)
            g = r.get("guard", {})
            same = (
                ""
                if a.hand != "none"
                else ("same as v3" if not r["same_as_v3"] else f"DIFFERS {r['same_as_v3'][:3]}")
            )
            ok = r.get("success") or (r.get("all_samples_loaded") and r.get("balanced"))
            print(
                f"{'ok ' if ok else 'BAD'} seed {r['seed']:2d} {r.get('scenario', ''):8s} "
                f"held {g.get('held_seconds')} s slow {g.get('slow_seconds')} s "
                f"stops {len(g.get('stops', []))} log ok {not g.get('log_problems')} "
                f"replay ok {not g.get('replay_disagrees')} | {same} {r.get('error', '')} "
                f"{r['wall_seconds']:.0f} s",
                flush=True,
            )
    (OUT / name / "results.json").write_text(json.dumps(rows, indent=1, default=str))
    if a.hand == "none":
        same = sum(not r["same_as_v3"] for r in rows)
        print(f"\nsame as v3's own results: {same} of {len(rows)}")
    done = sum(bool(r.get("all_samples_loaded") and r.get("balanced") and not r.get("error")) for r in rows)
    print(f"loaded and balanced: {done} of {len(rows)}")


if __name__ == "__main__":
    main()
