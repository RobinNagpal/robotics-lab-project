"""Render overhead pictures of random scenes, with YOLO labels made from MuJoCo's segmentation.

    python -m detect_classify.generate_data --split train --first 0 --count 2000

For every seed this writes, under data/<split>/:
    images/<seed>.jpg   the colour picture the detector sees
    labels/<seed>.txt   one line per object: class, then its outline as x y pairs from 0 to 1
    meta/<seed>.json    the camera calibration and the true objects, for grading only
    depth/<seed>.png    depth along the view axis in 0.1 mm steps (only with --depth)

Seeds of different splits never overlap, so test scenes are never seen in training.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import mujoco
import numpy as np
from lab_scene import liquid
from PIL import Image

from .camera import Calibration
from .classes import CLASS_ID, CLASSES
from .random_scene import IMAGE_SIZE, build

CAMERA = "overhead"
# Smaller than this many pixels, an object is too hidden to be worth labelling.
MIN_PIXELS = 40
# The outline may be this many pixels off the mask's true edge.
OUTLINE_TOLERANCE = 0.7
CLOSE_KERNEL = np.ones((5, 5), np.uint8)
DEPTH_STEP = 1e-4  # metres per unit in the depth PNG

SPLIT_SEEDS = {"train": 0, "val": 1_000_000, "test": 2_000_000}


def render(seed: int) -> tuple[np.ndarray, np.ndarray, list[dict], Calibration, list[str]]:
    """Colour, depth, one record per labelled object, the calibration and the label lines."""
    model, data, scene = build(seed)
    with mujoco.Renderer(model, height=IMAGE_SIZE[1], width=IMAGE_SIZE[0]) as r:
        r.update_scene(data, camera=CAMERA)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        depth = r.render().copy()
        r.disable_depth_rendering()
        r.enable_segmentation_rendering()
        seg = r.render().copy()
    calib = Calibration.from_model(model, data, CAMERA, IMAGE_SIZE)

    # Which object each pixel shows: the body its geom belongs to.
    geom_ids, kinds = seg[..., 0], seg[..., 1]
    body_of_pixel = np.full(geom_ids.shape, -1)
    is_geom = (kinds == mujoco.mjtObj.mjOBJ_GEOM) & (geom_ids >= 0)
    body_of_pixel[is_geom] = model.geom_bodyid[geom_ids[is_geom]]

    records, lines = [], []
    for inst in scene.instances:
        body = model.body(inst.name).id
        # A container is its glass and the liquid in it: from above the liquid
        # hides most of the glass.
        first, count = inst.balls
        bodies = [body] + [model.body(f"{liquid.BALL_PREFIX}{i}").id for i in range(first, first + count)]
        mask = np.isin(body_of_pixel, bodies).astype(np.uint8)
        record = asdict(inst) | {"pos": data.xpos[body].tolist(), "pixels": int(mask.sum()), "label": None}
        if record["pixels"] >= MIN_PIXELS:
            outline = _outline(mask)
            if outline is not None:
                record["label"] = len(lines)
                coords = " ".join(f"{x / IMAGE_SIZE[0]:.5f} {y / IMAGE_SIZE[1]:.5f}" for x, y in outline)
                lines.append(f"{CLASS_ID[inst.cls]} {coords}")
        records.append(record)
    return rgb, depth, records, calib, lines


def _outline(mask: np.ndarray) -> np.ndarray | None:
    """The object's outer edge, with anything seen through its opening filled in.

    From above, the middle of an open container shows the far wall or the
    table through the glass; the object is everything inside its outer edge.
    Thin glass can show as a broken ring, so gaps of a pixel or two are
    closed first. If something in front cuts it in two, the biggest piece is kept.
    """
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, CLOSE_KERNEL)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    outline = cv2.approxPolyDP(biggest, OUTLINE_TOLERANCE, closed=True)[:, 0, :]
    return outline if len(outline) >= 3 else None


def write(seed: int, out: Path, with_depth: bool) -> int:
    rgb, depth, records, calib, lines = render(seed)
    Image.fromarray(rgb).save(out / "images" / f"{seed}.jpg", quality=95)
    (out / "labels" / f"{seed}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    meta = {"seed": seed, "camera": calib.to_dict(), "objects": records}
    (out / "meta" / f"{seed}.json").write_text(json.dumps(meta, indent=1))
    if with_depth:
        steps = np.clip(np.round(depth / DEPTH_STEP), 0, 65535).astype(np.uint16)
        Image.fromarray(steps).save(out / "depth" / f"{seed}.png")
    return len(lines)


def load_depth(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=np.float64) * DEPTH_STEP


def _job(args: tuple[int, str, bool]) -> int:
    seed, out, with_depth = args
    return write(seed, Path(out), with_depth)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", choices=list(SPLIT_SEEDS), required=True)
    parser.add_argument("--first", type=int, default=0, help="first scene number within the split")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--depth", action="store_true", help="also save depth (needed for grading positions)")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    args = parser.parse_args()

    out = args.out / args.split
    for sub in ("images", "labels", "meta") + (("depth",) if args.depth else ()):
        (out / sub).mkdir(parents=True, exist_ok=True)
    seeds = range(SPLIT_SEEDS[args.split] + args.first, SPLIT_SEEDS[args.split] + args.first + args.count)
    jobs = [(s, str(out), args.depth) for s in seeds]

    start = time.time()
    total = 0
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, n in enumerate(pool.imap_unordered(_job, jobs, chunksize=4), start=1):
            total += n
            if i % 100 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} scenes, {total} objects, {time.time() - start:.0f} s", flush=True)

    dataset = args.out / "dataset.yaml"
    dataset.write_text(
        f"path: {args.out.resolve()}\ntrain: train/images\nval: val/images\ntest: test/images\n"
        + "names:\n"
        + "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASSES))
    )


if __name__ == "__main__":
    main()
