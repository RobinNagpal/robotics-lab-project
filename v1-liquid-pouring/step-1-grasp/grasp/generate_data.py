"""Close-up depth pictures of crowded racks, labelled by trying every turn with the fast check.

    python -m grasp.generate_data --split train --count 300
    python -m grasp.generate_data --split train --count 300 --kind side   # side grasps, into data_side/
    python -m grasp.generate_data --split train --count 300 --kind side2  # side grasps v2, into data_side2/
    python -m grasp.generate_data --split train --count 300 --kind side3  # v2 + 3 grip heights, data_side3/

For each scene, every tube is a target once:

1. Step 0's answer for it is made up from the truth plus Step 0's own errors
   (rules.step0_estimate), as is the list of all tubes Step 0 would report
   (a few are missed).
2. The close-up camera looks down from above that answer (scenes.LookCamera).
3. Every tube in the picture gets its 18 labels: for each turn, whether the
   gripper gets down to the grip height and closes without touching anything
   else (scenes.Checker), tried at the tube's true place: its rim's centre
   once the scene has settled.

Written under data/<split>/:
    heights.npy   (N, 128, 128) float16, height above the table of each pixel, m
    samples.json  per picture: the target, Step 0's answers, the camera, and
                  every labelled tube in view (pixel, true place, 18 labels)

Seeds of different splits never overlap, so test scenes are never seen in training.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from . import rules, scenes, side

SPLIT_SEEDS = {"train": 100_000, "val": 200_000, "test": 300_000}
# A tube counts as in the picture when its rim's centre is this far inside the edge.
EDGE = 4


def scene_samples(seed: int, kind: str = "top") -> tuple[np.ndarray, list[dict]]:
    """Top-down grasps (18 turns) or side grasps (24 directions, grasp/side.py).

    ``side2`` (version 2): the fingers come in barely open, and each picture is turned to the
    rack under it (side.rack_yaw), with the tubes in it labelled in that picture's directions.
    ``side3`` (version 3): as side2, labelled at each of side.HEIGHTS (72 labels a tube).
    """
    rack_turned = kind in ("side2", "side3")
    model, data, scene = scenes.build(seed)
    if kind == "top":
        checker = scenes.Checker(model, data)
    else:
        checker = side.SideChecker(model, data, preshape=rack_turned)
    height = scenes.LOOK_HEIGHT if kind == "top" else side.LOOK_HEIGHT
    camera = scenes.LookCamera(model, data)
    rng = np.random.default_rng(seed + 7_000_000)
    tubes = scenes.tubes(scene)
    truth = {t.name: scenes.rim(data, t) for t in tubes}
    # side2: made per picture instead, in that picture's directions.
    labels = {} if rack_turned else {t.name: checker.labels(*truth[t.name], t.name) for t in tubes}
    # What Step 0 reports for every tube in the scene, missing a few.
    step0 = []
    for t in tubes:
        xy, top = rules.step0_estimate(rng, *truth[t.name])
        step0.append(
            {"name": t.name, "xy": xy.tolist(), "top": top, "missed": bool(rng.random() < rules.STEP0_MISSED)}
        )
    pictures, samples = [], []
    for t, seen in zip(tubes, step0, strict=True):
        if seen["missed"]:
            continue  # the arm never goes to a tube it does not know about
        frame_yaw = 0.0
        if rack_turned:
            first, _ = camera.look(np.array(seen["xy"]), seen["top"], height)
            frame_yaw = side.rack_yaw(first)
        heights, calib = camera.look(np.array(seen["xy"]), seen["top"], height, yaw=frame_yaw)
        in_view = []
        for other in tubes:
            xy, top = truth[other.name]
            u, v = calib.project(np.array([[xy[0], xy[1], top]]))[0]
            if EDGE <= u < scenes.LOOK_SIZE - EDGE and EDGE <= v < scenes.LOOK_SIZE - EDGE:
                if rack_turned:
                    lab = checker.labels(xy, top, other.name, frame_yaw, heights=kind == "side3")
                else:
                    lab = labels[other.name]
                in_view.append(
                    {
                        "name": other.name,
                        "u": float(u),
                        "v": float(v),
                        "xy": xy.tolist(),
                        "top": top,
                        "labels": lab.astype(int).tolist(),
                    }
                )
        pictures.append(heights.astype(np.float16))
        samples.append(
            {
                "seed": seed,
                "target": t.name,
                "kind": t.kind,
                "step0": seen,
                "scene_step0": [s for s in step0 if not s["missed"]],
                "camera": calib.to_dict(),
                "frame_yaw": frame_yaw,
                "tubes": in_view,
            }
        )
    camera.close()
    return np.stack(pictures) if pictures else np.zeros((0, 128, 128), np.float16), samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", choices=list(SPLIT_SEEDS), required=True)
    parser.add_argument("--kind", choices=["top", "side", "side2", "side3"], default="top")
    parser.add_argument("--count", type=int, required=True, help="how many scenes")
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    args = parser.parse_args()

    out = (args.out if args.kind == "top" else Path(f"{args.out}_{args.kind}")) / args.split
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(SPLIT_SEEDS[args.split], SPLIT_SEEDS[args.split] + args.count))
    start = time.time()
    results = {}
    with mp.get_context("spawn").Pool(args.workers) as pool:
        jobs = [(s, args.kind) for s in seeds]
        for i, (seed, result) in enumerate(pool.imap_unordered(_job, jobs, chunksize=2), start=1):
            results[seed] = result
            if i % 50 == 0 or i == len(seeds):
                n = sum(len(r[1]) for r in results.values())
                print(f"{i}/{len(seeds)} scenes, {n} pictures, {time.time() - start:.0f} s", flush=True)
    pictures = np.concatenate([results[s][0] for s in seeds])
    samples = [x for s in seeds for x in results[s][1]]
    np.save(out / "heights.npy", pictures)
    (out / "samples.json").write_text(json.dumps(samples))
    targets = [next(t for t in x["tubes"] if t["name"] == x["target"]) for x in samples]
    graspable = np.mean([any(t["labels"]) for t in targets])
    turns = np.mean([np.mean(t["labels"]) for t in targets])
    print(
        f"{len(samples)} pictures; {graspable:.0%} of targets have a clear turn; "
        f"{turns:.0%} of turns are clear"
    )


def _job(args: tuple[int, str]):
    seed, kind = args
    return seed, scene_samples(seed, kind)


if __name__ == "__main__":
    main()
