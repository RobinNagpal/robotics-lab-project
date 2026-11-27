"""Step 6, programmed: the whole job on one scene, every step feeding the next, and the examiner's grade.

    Step 0  find: SAM 3 outlines the tubes in the high rack camera's picture; our rules say what each is
            (sample + colour, water, empty) and which rack cell it stands in. (Gemini, planned here, is
            blocked: no quota.)
    Step 1  weigh every sample Step 0 found, on the balance, and put it back.
    Step 2  plan from those readings: which hole for each tube, balance tubes' weights, the loading order.
    Step 3  fill each balance tube the plan asks for: an empty tube Step 0 found, water from a water tube;
            then put a spare cap on it (step-3-fill-balance-tube/cap.py).
    Step 4  load every tube into its hole, in the plan's order.
    Step 5  report: the robot's balance number from its own readings, and the examiner's. The lid stays open
            (out of the arm's reach as the centrifuge stands: step-5-close-and-report/docs/lid-reach.md).

What each step uses is only what the earlier steps found and the bench layout; the simulator's answers are
read only by the examiner at the end (and to grade Step 0's answer).

Step 0 runs first for every seed in this process (SAM 3 is a 3.4 GB model: one copy, not one per worker);
Steps 1 to 5 then run in up to ``--workers`` processes.

    cd step-6-whole-run
    PYTHONPATH=$PWD/../step-0-find:$PWD/../step-1-weigh:$PWD/../step-2-plan:$PWD/../step-3-fill-balance-tube:$PWD/../step-4-load-rotor:$PWD/../step-5-close-and-report:$PWD/../simulation-environment \\
        pixi run -e sam --manifest-path ../simulation-environment/pixi.toml python run.py --seeds 0 1 --workers 2
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

HERE = Path(__file__).parent
OUT = HERE / "results"
CAMERA = "racks_high"
# Step 3 stops pouring once this close under a balance tube's target (g): one more round can bring 10 balls
# (3 g) when one was wanted (step-3-fill-balance-tube/docs/results.md). Step 3 allows 0.5 g either way.
CLOSE_ENOUGH_G = 0.4


def step3_module():
    """Step 3's evaluate.py (its second balance tube's target), loaded by path: Steps 0 and 1 have an
    evaluate.py too."""
    path = HERE.parent / "step-3-fill-balance-tube" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("step3_evaluate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- Step 0 ------------------------------------------------------------------------------------------------


def pictures(model, data, names):
    """Colour and depth pictures from the scene as it is now (as step-0-find/bench.take_pictures)."""
    from bench import View, cameras
    from centrifuge_scene.settings import IMAGE_SIZE

    cams = cameras(model, data, names)
    views = {}
    with mujoco.Renderer(model, height=IMAGE_SIZE[1], width=IMAGE_SIZE[0]) as r:
        for n in names:
            r.update_scene(data, camera=n)
            buf = io.BytesIO()
            Image.fromarray(r.render()).save(buf, format="PNG")
            r.enable_depth_rendering()
            r.update_scene(data, camera=n)
            depth = r.render().copy()
            r.disable_depth_rendering()
            views[n] = View(cams[n], buf.getvalue(), depth)
    return views


def find_tubes(seeds: list[int]) -> dict[int, dict]:
    """Step 0 for each seed: {seed: {"cells": {"rack/index": {"kind", "colour"}}, "seconds", "right"}}."""
    import test_scenes
    from centrifuge_scene.settings import SAMPLE_COLOURS
    from find import find_separate
    from sam import Sam3

    names = ("red", "yellow", "green", "purple", "orange")
    sam = Sam3()
    out = {}
    for seed in seeds:
        model, data, truth = test_scenes.build(seed, changed=False)
        views = pictures(model, data, [CAMERA])
        sam.camera_of = {v.png: n for n, v in views.items()}
        t0 = time.time()
        found, _, _ = find_separate(views, sam)
        cells = {k: {"kind": f.kind, "colour": f.colour} for k, f in found.items()}
        # Graded against the simulator (the examiner's side only).
        true = {}
        for name, (rack, cell) in truth.cells.items():
            if name in truth.samples:
                true[f"{rack}/{cell}"] = {
                    "kind": "sample",
                    "colour": names[SAMPLE_COLOURS.index(truth.samples[name].colour)],
                }
            elif name.startswith("water"):
                true[f"{rack}/{cell}"] = {"kind": "water", "colour": "blue"}
            else:
                true[f"{rack}/{cell}"] = {"kind": "empty", "colour": "none"}
        out[seed] = {
            "cells": cells,
            "seconds": round(time.time() - t0, 1),
            "right": cells == true,
        }
        print(
            f"Step 0 seed {seed:2d}: {len(cells)} tubes found, {'all right' if cells == true else 'WRONG'}",
            flush=True,
        )
    return out


# --- Steps 1 to 5 -----------------------------------------------------------------------------------------


def run_seed(job: tuple[int, dict]) -> dict:
    seed, step0 = job
    import solver
    import test_scenes
    from centrifuge_scene import grading, liquid, scene
    from centrifuge_scene.racks import cell_centres
    from centrifuge_scene.settings import LAYOUT
    from fill import Filler
    from motion import Arm, Blocked
    from report import camera_check, robot_report
    from rotor import Loader
    from weigh import Weigher, tube_at

    step3 = step3_module()
    t0 = time.time()
    model, data, truth = test_scenes.build(seed, changed=False)
    cells = step0["cells"]
    out: dict = {
        "seed": seed,
        "scenario": truth.scenario,
        "step0_right": step0["right"],
        "error": "",
    }

    def split(key: str) -> tuple[str, int]:
        rack, idx = key.split("/")
        return rack, int(idx)

    samples = sorted(k for k, c in cells.items() if c["kind"] == "sample")
    waters = sorted(
        (k for k, c in cells.items() if c["kind"] == "water"), key=lambda k: split(k)[1]
    )
    empties = sorted(
        (k for k, c in cells.items() if c["kind"] == "empty"), key=lambda k: split(k)[1]
    )
    spilled_before = liquid.spilled(model, data, truth.tubes)
    arm = Arm(model, data)
    loaded: dict[int, float] = {}
    capped_of: dict[int, bool] = {}
    planned = 0
    try:
        # Step 1: weigh each sample.
        weigher = Weigher(arm)
        weighings = weigher.weigh_all([split(k) for k in samples])
        if any(w.error for w in weighings) or len(weighings) < len(samples):
            raise Blocked("Step 1: " + "; ".join(w.error for w in weighings if w.error))
        readings = {k: w.grams for k, w in zip(samples, weighings, strict=True)}
        out["step1_readings_g"] = readings
        out["step1_error_g"] = {
            k: round(
                readings[k]
                - truth.samples[
                    model.body(
                        tube_at(
                            arm,
                            np.array(
                                cell_centres(getattr(LAYOUT, split(k)[0]))[split(k)[1]]
                            ),
                        )
                    ).name
                ].mass
                * 1000,
                3,
            )
            for k in samples
        }
        # Step 2: plan from the readings.
        plan = solver.plan_load(readings)
        planned = len(plan.tubes)
        out["step2_plan"] = [
            {
                "hole": t.hole,
                "name": t.name,
                "grams": round(t.grams, 2),
                "balance": t.balance,
            }
            for t in plan.tubes
        ]
        # Step 3: fill each balance tube.
        filler = Filler(arm)
        filler.close_enough_g = CLOSE_ENOUGH_G
        arm.wait(0.3)
        filler.balance.tare()
        fills, balance_cell, balance_read, balance_capped = [], {}, {}, {}
        for i, bt in enumerate(plan.balance_tubes):
            if i >= len(empties) or i >= len(waters):
                raise Blocked("Step 3: not enough empty or water tubes found")
            target = (
                bt.grams
                if i == 0
                else step3.second_target(
                    plan, readings, plan.balance_tubes[0].name, fills[0]
                )
            )
            f = filler.fill(split(empties[i])[1], split(waters[i])[1], target, cap=i)
            if f.error:
                raise Blocked(f"Step 3: {f.error}")
            fills.append(f.final_g)
            balance_cell[bt.name], balance_read[bt.name] = empties[i], f.final_g
            balance_capped[bt.name] = f.capped
        out["step3_fills_g"] = balance_read
        filler.go_home()
        # Step 4: load every tube in the plan's order.
        loader = Loader(arm)
        for i, t in enumerate(plan.tubes):
            key = balance_cell[t.name] if t.balance else t.name
            rack, idx = split(key)
            xy = np.array(cell_centres(getattr(LAYOUT, rack))[idx])
            name = model.body(tube_at(arm, xy)).name  # what Step 0 found standing there
            if i:
                loader.return_home()
            capped = balance_capped[t.name] if t.balance else True
            capped_of[t.hole] = capped  # tried: the camera looks even if it failed
            loader.load(name, rack, idx, t.hole, capped=capped)
            loaded[t.hole] = balance_read[t.name] if t.balance else readings[key]
        loader.return_home()
    except Blocked as e:
        out["error"] = str(e)
    scene.settle(model, data, 0.5)
    # Step 5: the rotor camera checks every loaded tube is seated; the report; the examiner.
    # (Not after an error: the arm may be in the camera's view.)
    seen = (
        camera_check(model, data, capped_of) if capped_of and not out["error"] else {}
    )
    out["step5_seen_mm"] = seen
    g = grading.grade(model, data, truth)
    rep = robot_report(loaded, round(g.imbalance * 1000, 3), seen=seen)
    out.update(
        report=rep.text(),
        robot_g=rep.robot_g,
        examiner_g=round(g.imbalance * 1000, 3),
        tubes_planned=planned,
        tubes_seated=len(g.seated),
        all_samples_loaded=g.samples_loaded,
        balanced=g.balanced,
        balls_spilled=int(liquid.spilled(model, data, truth.tubes) - spilled_before),
        sim_blew_up=any(
            np.abs(data.xpos[model.body(n).id]).max() > 2.0 for n in truth.tubes
        ),
        sim_seconds=round(arm.time, 1),
        wall_seconds=round(time.time() - t0, 1),
    )
    out["robot_says_done"] = bool(
        rep.balanced and len(rep.holes) == planned and planned > 0
    )
    out["success"] = bool(
        not out["error"]
        and g.samples_loaded
        and g.balanced
        and out["balls_spilled"] == 0
        and not out["sim_blew_up"]
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    p.add_argument("--workers", type=int, default=1, help="at most 4 on this Mac")
    p.add_argument("--name", default="whole_run")
    a = p.parse_args()
    step0 = find_tubes(a.seeds)
    rows = []
    with ProcessPoolExecutor(a.workers) as pool:
        for r in pool.map(run_seed, [(s, step0[s]) for s in a.seeds]):
            r["step0_seconds"] = step0[r["seed"]]["seconds"]
            rows.append(r)
            mark = "ok " if r["success"] else "BAD"
            print(
                f"{mark} seed {r['seed']:2d} {r['scenario']:8s} {r['report']} | spilled {r['balls_spilled']} "
                f"{r['wall_seconds']:.0f} s {r['error']}",
                flush=True,
            )
    OUT.mkdir(exist_ok=True)
    (OUT / f"{a.name}.json").write_text(json.dumps(rows, indent=1, default=str))
    ok = [
        r
        for r in rows
        if not r["error"]
        and r["all_samples_loaded"]
        and r["balanced"]
        and not r["sim_blew_up"]
    ]
    print(
        f"\nloaded and balanced: {len(ok)} of {len(rows)}; "
        f"and no ball spilled as well (the simulator's strict rule): {sum(r['success'] for r in rows)} of {len(rows)}"
    )


if __name__ == "__main__":
    main()
