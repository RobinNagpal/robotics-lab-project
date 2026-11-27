"""Record demonstrations of Step 1 for the learned models (SmolVLA, later π0.5).

Each weighing gives two episodes, one sentence each:

- "put the red sample tube in the balance holder"  (rack cell -> holder; from wherever the arm is)
- "put the red sample tube back in row 1, column 3 of the sample rack"  (holder -> its cell)

The balance reading in between stays programmed (wait for a steady display) and is not recorded.

**Clean** weighings follow the teacher exactly. **Nudged** weighings push the arm 2 to 5 mm (and up to 3°)
off the teacher's step now and then, away from the caps, and record the teacher's own action, so the model
sees how to get back on course. A weighing whose tube does not end where it should, or that moves anything
else more than 2 mm, is dropped whole (both episodes): failures are never recorded.

Every 0.1 s (one frame) an episode stores:
- three pictures, 256 x 192: ``overhead`` (the whole bench, the balance), ``racks_high`` (every tube and its
  liquid's colour: from overhead the caps hide it), ``wrist`` (the fingers and the cap);
- ``state``: the 6 joint angles (rad) and the gripper's opening (0 open, 1 closed);
- ``action``: the teacher's 6 joint targets for the next 0.1 s and its gripper command (0 open, 1 closed).

Saved as plain files (``data/<split>/episode_NNNN/``: JPEG pictures, ``data.npz``, ``meta.json``),
converted to LeRobot's format when training starts.

Seeds: training 100+ (clean) and 200+ (nudged), validation 300+ and 400+. Seeds 0 to 19 are the test scenes
and are never recorded.

    cd step-1-weigh
    P="pixi run --manifest-path ../simulation-environment/pixi.toml"
    PYTHONPATH=$PWD:$PWD/../simulation-environment $P python record.py --workers 4
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from centrifuge_scene import scene
from centrifuge_scene.racks import RACK, cell_centres
from centrifuge_scene.settings import LAYOUT
from PIL import Image

import teacher as T
from motion import Arm, Blocked
from weigh import Weigher, tube_at

DATA = Path(__file__).parent / "data"
CAMERAS = ("overhead", "racks_high", "arm/wrist")
SIZE = (256, 192)  # width, height
JPEG_QUALITY = 85
SETTLE = 1.5
COLOUR_NAMES = ("red", "yellow", "green", "purple", "orange")
# A tube counts as placed if it stands within these of where it should be.
PLACED_XY = 0.004
PLACED_TILT_DEG = 3.0
DISTURBED = 0.002

# (split, kind, first seed, weighings)
PLAN = (
    ("train", "clean", 100, 36),
    ("train", "nudged", 200, 24),
    ("validation", "clean", 300, 3),
    ("validation", "nudged", 400, 3),
)


@dataclass
class Job:
    split: str
    kind: str
    seed: int
    weighings: int  # how many of the seed's samples to weigh (the first ones)


def sentence_to_holder(colour: str) -> str:
    return f"put the {colour} sample tube in the balance holder"


def sentence_back(colour: str, cell: int) -> str:
    row, col = divmod(cell, RACK.columns)
    return f"put the {colour} sample tube back in row {row + 1}, column {col + 1} of the sample rack"


def jobs_for(split: str, kind: str, first: int, weighings: int) -> tuple[list[Job], int]:
    """Seeds from ``first`` on, weighing every sample, until ``weighings`` are planned (the last seed fewer).

    Returns the jobs and the next unused seed.
    """
    out, seed, left = [], first, weighings
    while left > 0:
        _, truth = scene.build(seed)
        n = min(len(truth.samples), left)
        out.append(Job(split, kind, seed, n))
        left -= n
        seed += 1
    return out, seed


class Recorder:
    """Collects one episode's frames: pictures from the three cameras with the teacher's tick."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model, self.data = model, data
        self.renderer = mujoco.Renderer(model, height=SIZE[1], width=SIZE[0])
        self.frames: list[dict[str, np.ndarray]] = []
        self.ticks: list[T.Tick] = []

    def __call__(self, tick: T.Tick) -> None:
        pics = {}
        for cam in CAMERAS:
            self.renderer.update_scene(self.data, camera=cam)
            pics[cam] = self.renderer.render().copy()
        self.frames.append(pics)
        self.ticks.append(tick)

    def reset(self) -> None:
        self.frames, self.ticks = [], []

    def save(self, folder: Path, meta: dict) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        for cam in CAMERAS:
            sub = folder / cam.replace("arm/", "")
            sub.mkdir(exist_ok=True)
            for i, pics in enumerate(self.frames):
                Image.fromarray(pics[cam]).save(sub / f"{i:04d}.jpg", quality=JPEG_QUALITY)
        np.savez_compressed(
            folder / "data.npz",
            state=np.array([t.state for t in self.ticks], dtype=np.float32),
            action=np.array([t.action for t in self.ticks], dtype=np.float32),
            nudged=np.array([t.nudged for t in self.ticks]),
        )
        (folder / "meta.json").write_text(
            json.dumps({**meta, "frames": len(self.ticks), "fps": 10}, indent=1)
        )


def tilt_deg(data: mujoco.MjData, body: int) -> float:
    return math.degrees(math.acos(min(1.0, abs(float(data.xmat[body].reshape(3, 3)[2, 2])))))


def placed(data: mujoco.MjData, body: int, xy: np.ndarray) -> bool:
    return bool(
        np.linalg.norm(data.xpos[body][:2] - xy) < PLACED_XY and tilt_deg(data, body) < PLACED_TILT_DEG
    )


def holder_turn_near(w: Weigher, t_rack: float) -> float:
    """The clear finger turn at the holder nearest the rack's (no needless wrist spin)."""
    arm = w.arm
    saved = arm.turn
    arm.turn = t_rack
    try:
        return w.holder_turn()[0]
    finally:
        arm.turn = saved


def record_job(job: Job) -> list[dict]:
    """Record one seed's weighings into a staging folder; returns a summary per weighing."""
    model, truth = scene.build(job.seed, awake=True)
    data = mujoco.MjData(model)
    scene.reset(model, data)
    scene.settle(model, data, SETTLE)
    arm = Arm(model, data)
    w = Weigher(arm)
    teach = T.Teacher(arm)
    rec = Recorder(model, data)
    rng = np.random.default_rng(10_000 + job.seed)
    arm.wait(0.3)
    w.balance.tare()
    names = list(truth.tubes)
    out = []
    stage = DATA / "_staging" / f"{job.split}_{job.kind}_seed{job.seed}"
    if stage.exists():
        shutil.rmtree(stage)
    for i, name in enumerate(list(truth.samples)[: job.weighings]):
        colour = COLOUR_NAMES[int(name.removeprefix("sample"))]
        rack, cell = truth.cells[name]
        xy = np.array(cell_centres(getattr(LAYOUT, rack))[cell])
        summary = {
            "seed": job.seed,
            "kind": job.kind,
            "split": job.split,
            "tube": name,
            "cell": cell,
            "ok": False,
        }
        try:
            tube = tube_at(arm, xy)
            t_rack, _ = w.rack_turn(xy, tube)
            t_hold = holder_turn_near(w, t_rack)
        except Blocked as e:
            summary["error"] = str(e)
            out.append(summary)
            break
        before = {n: data.body(n).xpos.copy() for n in names if n != name}
        nudger = (lambda: T.Nudger(rng)) if job.kind == "nudged" else (lambda: None)

        rec.reset()
        ep1 = teach.run(T.to_holder(w, xy, tube, t_rack, t_hold), rec, nudger())
        ok1 = not ep1.error and placed(data, tube, w.holder_xy)
        if ok1:
            rec.save(
                stage / f"w{i}_a",
                {
                    "task": sentence_to_holder(colour),
                    "seed": job.seed,
                    "kind": job.kind,
                    "tube": name,
                    "colour": colour,
                    "rack": rack,
                    "cell": cell,
                    "nudges": ep1.nudges,
                    "part": "to_holder",
                },
            )
        grams, steady, _ = w.read_steady() if ok1 else (None, False, [])

        rec.reset()
        ep2 = teach.run(T.back_to_cell(w, xy, tube, t_rack, t_hold), rec, nudger()) if ok1 else T.Episode()
        ok2 = ok1 and not ep2.error and placed(data, tube, xy)
        moved = {n: float(np.linalg.norm(data.body(n).xpos - p)) for n, p in before.items()}
        calm = all(v < DISTURBED for v in moved.values())
        if ok2:
            rec.save(
                stage / f"w{i}_b",
                {
                    "task": sentence_back(colour, cell),
                    "seed": job.seed,
                    "kind": job.kind,
                    "tube": name,
                    "colour": colour,
                    "rack": rack,
                    "cell": cell,
                    "nudges": ep2.nudges,
                    "part": "back",
                },
            )
        summary.update(
            ok=bool(ok1 and ok2 and calm),
            grams=grams,
            steady=steady,
            frames=len(ep1.ticks) + len(ep2.ticks),
            nudges=ep1.nudges + ep2.nudges,
            error=ep1.error or ep2.error or ("" if ok1 and ok2 else "tube not placed") or "",
            most_moved_mm=round(max(moved.values(), default=0.0) * 1000, 2),
            tube_in_holder=ok1,
        )
        if not summary["ok"]:
            for part in ("a", "b"):
                if (stage / f"w{i}_{part}").exists():
                    shutil.rmtree(stage / f"w{i}_{part}")
            out.append(summary)
            break  # the scene may now be disturbed: do not record more from it
        out.append(summary)
    rec.renderer.close()
    return out


def gather(summaries: list[dict]) -> dict[str, int]:
    """Move the kept episodes from staging into data/<split>/episode_NNNN, numbered in order."""
    counts: dict[str, int] = {}
    for split in ("train", "validation"):
        target = DATA / split
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        n = 0
        for folder in sorted((DATA / "_staging").glob(f"{split}_*")):
            for ep in sorted(folder.iterdir()):
                shutil.move(str(ep), target / f"episode_{n:04d}")
                n += 1
        counts[split] = n
    shutil.rmtree(DATA / "_staging")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--only", type=int, nargs="*", help="just these training seeds (a trial: not gathered)"
    )
    args = parser.parse_args()
    t0 = time.time()
    if args.only:
        jobs = [Job("trial", "nudged" if s >= 200 else "clean", s, 99) for s in args.only]
        with ProcessPoolExecutor(args.workers) as pool:
            results = [s for r in pool.map(record_job, jobs) for s in r]
    else:
        # Dropped weighings are made up from further seeds, until every part of the plan has its count.
        wanted = {(split, kind): (first, n) for split, kind, first, n in PLAN}
        results = []
        with ProcessPoolExecutor(args.workers) as pool:
            while wanted:
                jobs, next_seed = [], {}
                for (split, kind), (first, n) in wanted.items():
                    js, next_seed[split, kind] = jobs_for(split, kind, first, n)
                    jobs += js
                print(f"{len(jobs)} seeds, {sum(j.weighings for j in jobs)} weighings to record")
                round_results = [s for r in pool.map(record_job, jobs) for s in r]
                results += round_results
                short = {}
                for (split, kind), (_, n) in wanted.items():
                    kept = sum(r["ok"] for r in round_results if r["split"] == split and r["kind"] == kind)
                    if kept < n:
                        short[split, kind] = (next_seed[split, kind], n - kept)
                wanted = short
    seconds = time.time() - t0
    kept = sum(r["ok"] for r in results)
    print(f"kept {kept} of {len(results)} weighings in {seconds / 60:.1f} min")
    for r in results:
        if not r["ok"]:
            print("  dropped:", r)
    if args.only:
        for r in results:
            print(r)
        return
    counts = gather(results)
    DATA.mkdir(exist_ok=True)
    (DATA / "summary.json").write_text(
        json.dumps({"weighings": results, "episodes": counts, "minutes": round(seconds / 60, 1)}, indent=1)
    )
    print("episodes:", counts)


if __name__ == "__main__":
    main()
