"""
False alarms with nobody there: the detector over every recorded moment of real v3 jobs, with the cell empty.


The closed-loop run found what the picture test had missed: in one pose, the robot's own gripper (dark fingers
around a tube) looked like a dark glove to YOLO v2. The arm held, the scene stood still, the detector kept
seeing the same "hand", and the job never finished (docs/integration.md). Every moment the robot can be in
must be checked with nobody there.

For each recorded job snapshot (``record_jobs.py``, every ``--every``-th one), with a random lighting level
and the camera's sensor noise, it runs a registry version of the detector and the fixed rule, and counts the
pictures where the zone is not clear. Two uses:

- **a test** (seeds 4 to 7, never trained on): the false-alarm rate with nobody there, with a 95% upper bound;
- **mining** (seeds 0 to 3, the training jobs, ``--save``): every false alarm is written out as a training
  picture with no hand in it (and its depth, for the record), for the next version to learn from.

    python nobody_scan.py --version v2 --jobs 4 5 6 7 --name test_v2
    python nobody_scan.py --version v2 --jobs 0 1 2 3 --save train_negatives --name mine_v2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

import dataset
import detector as D
import hand
import keys
import scene_plus
import sensing
from evaluate import upper_bound
from registry import Registry
from safety_log import SafetyLog

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
MODELS = HERE / "models"


def scan(version: str, jobs: list[int], every: int, rng_seed: int, save: str | None, name: str) -> dict:
    key = keys.load()
    registry = Registry(MODELS / "registry.json", key=key)
    det = D.load(MODELS / registry.models[version]["file"], registry, SafetyLog(), sensing.yolo_builder())
    rng = np.random.default_rng(rng_seed)
    examples = RESULTS / f"nobody_{name}"
    examples.mkdir(parents=True, exist_ok=True)
    if save:
        for sub in ("images", "labels", "depth"):
            (dataset.OUT / sub / save).mkdir(parents=True, exist_ok=True)
    rows, alarms = [], []
    for seed in jobs:
        model, _ = scene_plus.build(seed, awake=True)
        data = mujoco.MjData(model)
        job = dataset.load_job(seed)
        person = sensing.Person(model, data, rng, reaches=False)
        cam = sensing.SafetyCamera(person, rng)
        for i in range(0, len(job["time"]), every):
            dataset.set_snapshot(model, data, job, i)
            level = str(rng.choice(list(scene_plus.LIGHTING), p=[0.25, 0.5, 0.25]))
            scene_plus.set_lighting(model, float(rng.uniform(*scene_plus.LIGHTING[level])))
            hand.place(model, data, None, np.array([1.0, 0, 0, 0]))
            mujoco.mj_forward(model, data)
            frame = cam.capture(model, data, len(rows) + 1)
            d = det.detect(frame)
            row = {
                "seed": seed,
                "snapshot": i,
                "t": float(job["time"][i]),
                "lighting": level,
                "zone": d.zone,
                "score": round(d.score, 3),
                "boxes": [[round(v, 1) for v in b] for b in d.boxes],
            }
            rows.append(row)
            if d.zone != "clear":
                alarms.append(row)
                if len(alarms) <= 30:
                    im = Image.fromarray(frame.image)
                    draw = ImageDraw.Draw(im)
                    for b in d.boxes:
                        draw.rectangle(b, outline=(255, 0, 255), width=2)
                    im.save(examples / f"seed{seed}_snap{i:05d}_{d.zone}.jpg", quality=90)
            if save and (d.boxes or d.zone != "clear"):
                # A picture with no hand in it that the detector got wrong: a training picture with no box.
                stem = f"{save}_s{seed}_{i:05d}"
                Image.fromarray(frame.image).save(dataset.OUT / "images" / save / f"{stem}.jpg", quality=92)
                (dataset.OUT / "labels" / save / f"{stem}.txt").write_text("")
                dataset.save_depth(dataset.OUT / "depth" / save / f"{stem}.png", frame.depth)
        cam.close()
    n, k = len(rows), len(alarms)
    boxed = sum(bool(r["boxes"]) for r in rows)
    result = {
        "version": version,
        "jobs": jobs,
        "every": every,
        "pictures": n,
        "pictures with a box": boxed,
        "false alarms (zone not clear)": {
            "k": k,
            "n": n,
            "rate": k / n if n else None,
            "upper95": upper_bound(k, n),
        },
        "by job": {str(s): sum(r["seed"] == s for r in alarms) for s in jobs},
        "alarms": alarms,
    }
    (RESULTS / f"nobody_{name}.json").write_text(json.dumps(result, indent=1) + "\n")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--version", required=True)
    p.add_argument("--jobs", type=int, nargs="+", required=True)
    p.add_argument("--every", type=int, default=2, help="use every n-th snapshot (they are 0.25 s apart)")
    p.add_argument("--rng", type=int, default=7)
    p.add_argument("--save", default=None, help="write the false alarms as hand-free training pictures here")
    p.add_argument("--name", required=True)
    a = p.parse_args()
    r = scan(a.version, a.jobs, a.every, a.rng, a.save, a.name)
    f = r["false alarms (zone not clear)"]
    print(
        f"{a.version} on jobs {a.jobs}: {f['k']} false alarms in {f['n']} pictures with nobody there "
        f"({f['rate']:.2%}, 95% upper bound {f['upper95']:.2%}); {r['pictures with a box']} with a box; "
        f"by job {r['by job']}"
    )


if __name__ == "__main__":
    main()
