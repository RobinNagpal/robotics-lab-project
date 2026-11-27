"""Render the hand detector's pictures: the safety camera's view of real v3 jobs with a person reaching in.

Each picture takes a moment of a recorded v3 job (``record_jobs.py``: the arm, the tubes, the liquid and the
lid where they really were), adds the person at a random moment of a random reach, with a random look and
lighting, and renders the safety camera. About one picture in seven has nobody in it.

For each picture it writes, in YOLO's format:

- ``images/<split>/<name>.jpg``: the picture;
- ``labels/<split>/<name>.txt``: one line per visible hand, ``0 cx cy w h`` (box centre and size, 0 to 1). The
  box is taken from MuJoCo's segmentation picture, so it covers exactly the hand's pixels that the camera sees
  (the robot may hide part of it). A hand with fewer than ``MIN_PIXELS`` pixels showing gets no box;
- ``meta/<split>.jsonl``: the truth and the conditions (zone, how close, gap to the robot, look, lighting,
  hand height, pixels showing), for the per-condition results in Package 2.

Splits never share a job: the test pictures come from other seeds' jobs (other tubes, other moves) than the
training and validation pictures.

    python dataset.py --split train --jobs 0 1 2 3 --n 1500 --workers 2
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

import cameras
import hand
import scene_plus
import zones

HERE = Path(__file__).resolve().parent
JOBS = HERE / "data" / "jobs"
OUT = HERE / "data" / "hands"
NOBODY = 1 / 7  # share of pictures with nobody in them
MIN_PIXELS = 12  # fewer hand pixels than this showing: no box (too little to find)
JPEG_QUALITY = 92


def load_job(seed: int) -> dict[str, np.ndarray]:
    with np.load(JOBS / f"seed{seed}.npz") as f:
        return {k: f[k] for k in f.files}


def set_snapshot(model: mujoco.MjModel, data: mujoco.MjData, job: dict, i: int) -> None:
    """Put the scene in the state of snapshot ``i`` of a recorded job.

    v3's own parts come first in every list, so the recorded arrays fit the start of ours.
    """
    data.qpos[:] = job["qpos"][i]
    n = job["mocap_pos"].shape[1]
    data.mocap_pos[:n] = job["mocap_pos"][i]
    data.mocap_quat[:n] = job["mocap_quat"][i]
    g = job["geom_group"].shape[1]
    model.geom_group[:g] = job["geom_group"][i]


def hand_box(seg: np.ndarray, hand_ids: np.ndarray) -> tuple[int, tuple[int, int, int, int] | None]:
    """The hand's pixels in a segmentation picture: (count, (x0, y0, x1, y1)) or (count, None) if too few."""
    is_geom = seg[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM
    mask = is_geom & np.isin(seg[:, :, 0], hand_ids)
    count = int(mask.sum())
    if count < MIN_PIXELS:
        return count, None
    ys, xs = np.nonzero(mask)
    return count, (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def render_chunk(job: tuple[str, int, list[int], int, int, str, str]) -> list[dict]:
    """Render pictures ``start`` to ``start + count`` of a split, from the recorded jobs of ``seeds``.

    ``mode``: "all" writes the pictures and labels; "depth" writes only the depth pictures (the same moments,
    drawn from the same random numbers, so they fit pictures rendered before); "both" writes everything.

    ``focus``: "none", or "near_robot" for v2's extra pictures: most reaches go to where the robot is (so it
    partly covers the hand), and most hands wear dark gloves.
    """
    split, start, seeds, count, rng_seed, mode, focus = job
    rng = np.random.default_rng([rng_seed, start])
    w, h = cameras.SAFETY.width, cameras.SAFETY.height
    for sub in ("images", "labels", "depth"):
        (OUT / sub / split).mkdir(parents=True, exist_ok=True)
    rows = []
    worlds = {}
    for k in range(count):
        seed = int(seeds[(start + k) % len(seeds)])
        if seed not in worlds:
            model, _ = scene_plus.build(seed, awake=True)
            data = mujoco.MjData(model)
            rgb = mujoco.Renderer(model, height=h, width=w)
            seg = mujoco.Renderer(model, height=h, width=w)
            seg.enable_segmentation_rendering()
            dep = mujoco.Renderer(model, height=h, width=w)
            dep.enable_depth_rendering()
            hand_ids = np.array([model.geom(hand.PREFIX + n).id for n in hand.SKIN_SHAPES])
            worlds[seed] = (model, data, rgb, seg, dep, hand_ids, load_job(seed))
        model, data, rgb, seg, dep, hand_ids, rec = worlds[seed]
        snap = int(rng.integers(len(rec["time"])))
        set_snapshot(model, data, rec, snap)
        lighting = str(rng.choice(list(scene_plus.LIGHTING), p=[0.25, 0.5, 0.25]))
        scale = float(rng.uniform(*scene_plus.LIGHTING[lighting]))
        scene_plus.set_lighting(model, scale)
        look = hand.Look.draw(rng, dark_gloves=focus == "near_robot")
        hand.set_look(model, look)
        mujoco.mj_forward(model, data)
        row = {
            "split": split,
            "seed": seed,
            "snapshot": snap,
            "lighting": lighting,
            "light_scale": round(scale, 3),
        }
        if rng.random() < NOBODY:
            hand.place(model, data, None, np.array([1.0, 0, 0, 0]))
            row.update(goal="nobody", phase="none")
        else:
            wanted = hand.NEAR_ROBOT if focus == "near_robot" and rng.random() < 0.6 else None
            goal, reach = hand.draw_reach(rng, model, data, 0.0, 0.0, goal=wanted)
            phase = str(rng.choice(["in", "pause", "out"], p=[0.4, 0.4, 0.2]))
            t = {
                "in": rng.uniform(reach.t0, reach.t_arrive),
                "pause": rng.uniform(reach.t_arrive, reach.t_leave),
                "out": rng.uniform(reach.t_leave, reach.t_end),
            }[phase]
            wrist = reach.wrist(t)
            hand.place(model, data, wrist, reach.quat())
            row.update(goal=goal, phase=phase, hand_height=round(float(wrist[2]), 3), **asdict(look))
        mujoco.mj_forward(model, data)
        truth = zones.truth(model, data)
        gap = zones.distance_to_robot(model, data) if truth.visible else 0.5
        rgb.update_scene(data, camera=cameras.SAFETY.name)
        picture = rgb.render()
        seg.update_scene(data, camera=cameras.SAFETY.name)
        pixels, box = hand_box(seg.render(), hand_ids)
        name = f"{split}_{start + k:05d}"
        if mode in ("all", "both"):
            Image.fromarray(picture).save(OUT / "images" / split / f"{name}.jpg", quality=JPEG_QUALITY)
            label = ""
            if box is not None:
                x0, y0, x1, y1 = box
                cx, cy, bw, bh = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h
                label = f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
            (OUT / "labels" / split / f"{name}.txt").write_text(label)
        if mode in ("depth", "both"):
            dep.update_scene(data, camera=cameras.SAFETY.name)
            save_depth(OUT / "depth" / split / f"{name}.png", dep.render())
        row.update(
            name=name,
            zone=truth.zone.name,
            reach=None if math.isinf(truth.reach) else round(truth.reach, 4),
            gap=round(gap, 4),
            hand_pixels=pixels,
            box=box,
        )
        rows.append(row)
    for world in worlds.values():
        for renderer in world[2:5]:
            renderer.close()
    return rows


def save_depth(path: Path, depth: np.ndarray) -> None:
    """A depth picture (metres from the camera, along its view axis) as a 16-bit PNG in millimetres."""
    Image.fromarray(np.clip(np.round(depth * 1000), 0, 65535).astype(np.uint16)).save(path)


def load_depth(path: Path) -> np.ndarray:
    """The depth picture in metres."""
    return np.asarray(Image.open(path), dtype=np.float32) / 1000.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--split", required=True, choices=["train", "val", "test", "train_extra"])
    p.add_argument("--jobs", type=int, nargs="+", required=True, help="recorded job seeds to draw from")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--rng", type=int, default=None, help="default: a different one per split")
    p.add_argument(
        "--mode",
        choices=["all", "depth", "both"],
        default="all",
        help="depth: only the depth pictures, for pictures already rendered (checked against their truth)",
    )
    p.add_argument(
        "--focus",
        choices=["none", "near_robot"],
        default="none",
        help="near_robot: v2's extra pictures, hands where the robot is and mostly dark gloves",
    )
    args = p.parse_args()
    rng_seed = (
        args.rng if args.rng is not None else {"train": 1, "val": 2, "test": 3, "train_extra": 4}[args.split]
    )
    chunk = 50
    work = [
        (args.split, s, args.jobs, min(chunk, args.n - s), rng_seed, args.mode, args.focus)
        for s in range(0, args.n, chunk)
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = [r for part in pool.map(render_chunk, work) for r in part]
    if args.mode == "depth":
        # The same moments as the pictures already there: same names, truth and boxes.
        before = [
            json.loads(line) for line in (OUT / "meta" / f"{args.split}.jsonl").read_text().splitlines()
        ]
        same = len(before) == len(rows) and all(
            a["name"] == b["name"]
            and a["zone"] == b["zone"]
            and a["box"] == (list(b["box"]) if b["box"] else None)
            for a, b in zip(before, rows, strict=True)
        )
        print(f"{args.split}: {len(rows)} depth pictures; {'same moments as before' if same else 'MISMATCH'}")
        return
    (OUT / "meta").mkdir(parents=True, exist_ok=True)
    (OUT / "meta" / f"{args.split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    zones_seen = {z: sum(r["zone"] == z for r in rows) for z in ("CLEAR", "WARNING", "STOP")}
    boxes = sum(r["box"] is not None for r in rows)
    print(f"{args.split}: {len(rows)} pictures, {boxes} with a hand box; zones {zones_seen}")


if __name__ == "__main__":
    main()
