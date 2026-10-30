"""Grade Step 0 on scenes the detector never saw: count, class and position.

    python -m detect_classify.evaluate --weights runs/yolo11s_640_800/weights/best.pt
    python -m detect_classify.evaluate --custom runs/custom_rgb/weights/best.pt   # the custom CNN
    python -m detect_classify.evaluate --oracle    # the true masks: grades only the locating maths

Each detection is paired with the true object whose outline it overlaps most
(at least half, by intersection over union). Then, over all test pictures:
    count     true objects found, and detections that match nothing
    class     share of found objects given the right class
    position  distance in x-y between found and true position, and error of the top height

Results go to runs/<name>/step0_results.json.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .camera import Calibration
from .classes import CLASSES
from .generate_data import load_depth
from .locate import locate, own_pixels

MATCH_IOU = 0.5


def true_masks(label_file: Path, size: tuple[int, int]) -> list[tuple[str, np.ndarray]]:
    """The labelled outlines, filled, as (class, mask)."""
    w, h = size
    out = []
    for line in label_file.read_text().splitlines():
        cls, *xy = line.split()
        points = (np.array(xy, dtype=float).reshape(-1, 2) * [w, h]).round().astype(np.int32)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [points], 1)
        out.append((CLASSES[int(cls)], mask.astype(bool)))
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def pair(found: list[np.ndarray], truth: list[np.ndarray]) -> list[tuple[int, int]]:
    """Greedy pairing, best overlap first; each side used once."""
    scores = [(iou(f, t), i, j) for i, f in enumerate(found) for j, t in enumerate(truth)]
    pairs, used_f, used_t = [], set(), set()
    for score, i, j in sorted(scores, reverse=True):
        if score < MATCH_IOU:
            break
        if i not in used_f and j not in used_t:
            pairs.append((i, j))
            used_f.add(i)
            used_t.add(j)
    return pairs


def detections(model, image: Path, conf: float) -> list[tuple[str, np.ndarray]]:
    result = model.predict(str(image), conf=conf, retina_masks=True, verbose=False)[0]
    if result.masks is None:
        return []
    masks = result.masks.data.cpu().numpy() > 0.5
    classes = result.boxes.cls.cpu().numpy().astype(int)
    return [(CLASSES[c], m) for c, m in zip(classes, masks, strict=True)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=Path("data/test"))
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--custom", type=Path, help="the custom CNN's weights, instead of YOLO's")
    parser.add_argument("--oracle", action="store_true", help="use the true masks instead of a detector")
    parser.add_argument("--conf", type=float, default=0.5, help="YOLO's confidence cut-off")
    parser.add_argument("--peak", type=float, default=0.3, help="the custom CNN's centre-peak cut-off")
    args = parser.parse_args()
    if sum(map(bool, (args.oracle, args.weights, args.custom))) != 1:
        parser.error("give one of --weights, --custom or --oracle")

    detector = None
    if args.weights:
        from ultralytics import YOLO

        model = YOLO(str(args.weights))

        def detector(image):
            return detections(model, image, args.conf)

    elif args.custom:
        from .custom_net import Detector

        detector = Detector(args.custom, threshold=args.peak)

    per_class = defaultdict(lambda: {"true": 0, "found": 0, "right_class": 0, "xy_mm": [], "top_mm": []})
    extra = 0
    confusion = defaultdict(int)
    images = sorted((args.data / "images").glob("*.jpg"), key=lambda p: int(p.stem))
    for image in images:
        meta = json.loads((args.data / "meta" / f"{image.stem}.json").read_text())
        calib = Calibration.from_dict(meta["camera"])
        depth = load_depth(args.data / "depth" / f"{image.stem}.png")
        truth = true_masks(args.data / "labels" / f"{image.stem}.txt", (calib.width, calib.height))
        by_label = {o["label"]: o for o in meta["objects"] if o["label"] is not None}
        found = truth if args.oracle else detector(image)

        pairs = pair([m for _, m in found], [m for _, m in truth])
        own = own_pixels(found)
        extra += len(found) - len(pairs)
        for cls, _ in truth:
            per_class[cls]["true"] += 1
        for i, j in pairs:
            true_cls, obj = truth[j][0], by_label[j]
            stats = per_class[true_cls]
            stats["found"] += 1
            found_cls = found[i][0]
            confusion[(true_cls, found_cls)] += 1
            if found_cls != true_cls:
                continue
            stats["right_class"] += 1
            where = locate(found_cls, own[i], depth, calib)
            if where is None:
                continue
            stats["xy_mm"].append(1000 * float(np.hypot(where.x - obj["pos"][0], where.y - obj["pos"][1])))
            stats["top_mm"].append(1000 * abs(where.top - obj["top"]))

    summary = {"images": len(images), "extra_detections": extra, "classes": {}}
    for cls in CLASSES:
        s = per_class[cls]
        xy, top = np.array(s["xy_mm"]), np.array(s["top_mm"])
        summary["classes"][cls] = {
            "true": s["true"],
            "found": s["found"],
            "right_class": s["right_class"],
            "xy_mm_median": float(np.median(xy)) if len(xy) else None,
            "xy_mm_p95": float(np.percentile(xy, 95)) if len(xy) else None,
            "xy_under_10mm": float((xy < 10).mean()) if len(xy) else None,
            "top_mm_median": float(np.median(top)) if len(top) else None,
        }
    summary["confusion"] = {f"{t} -> {f}": n for (t, f), n in sorted(confusion.items()) if t != f}
    _print(summary)

    name = "oracle" if args.oracle else (args.weights or args.custom).parent.parent.name
    out = Path("runs") / name / "step0_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1))
    print(out)


def _print(summary: dict) -> None:
    print(f"{summary['images']} pictures, {summary['extra_detections']} detections matching nothing")
    print(f"{'class':16s} {'true':>5s} {'found':>6s} {'class ok':>9s}", end=" ")
    print(f"{'xy med':>7s} {'xy p95':>7s} {'<1cm':>6s} {'top med':>8s}")
    for cls, s in summary["classes"].items():

        def mm(v):
            return f"{v:7.1f}" if v is not None else "      -"

        under = f"{100 * s['xy_under_10mm']:5.1f}%" if s["xy_under_10mm"] is not None else "     -"
        print(
            f"{cls:16s} {s['true']:5d} {s['found']:6d} {s['right_class']:9d} {mm(s['xy_mm_median'])} "
            f"{mm(s['xy_mm_p95'])} {under} {mm(s['top_mm_median'])}"
        )
    for k, n in summary["confusion"].items():
        print(f"  mistaken: {k}: {n}")


if __name__ == "__main__":
    main()
