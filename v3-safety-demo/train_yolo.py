"""Train the hand detector: YOLO11n (Ultralytics), from its COCO weights, on ``dataset.py``'s pictures.

One class, "hand". Small on purpose: the smallest YOLO11 (2.6 million weights) trains in minutes on the Mac's
GPU and finds a hand in a picture in a few milliseconds.

    python train_yolo.py --name v1 --epochs 20                      # the real run
    python train_yolo.py --name speed --epochs 1 --fraction 0.2     # a short run, to time an epoch

Writes ``runs/<name>/`` (Ultralytics' own output: weights, curves, results.csv) and
``runs/<name>/timing.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "hands"
RUNS = HERE / "runs"


def data_yaml(train: list[str]) -> Path:
    """Ultralytics' dataset file: where the pictures are, and the one class."""
    path = DATA / f"hands_{'_'.join(train)}.yaml"
    lines = (
        [f"path: {DATA}", "train:"]
        + [f"  - images/{t}" for t in train]
        + ["val: images/val", "names:", "  0: hand"]
    )
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--name", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--fraction", type=float, default=1.0, help="share of the training pictures to use")
    p.add_argument("--train", nargs="+", default=["train"], help="training splits (v2 adds train_extra)")
    p.add_argument("--weights", default="yolo11n.pt", help="start from these (v2 can start from v1's)")
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    t0 = time.time()
    model.train(
        data=str(data_yaml(args.train)),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        fraction=args.fraction,
        device="mps",
        workers=args.workers,
        project=str(RUNS),
        name=args.name,
        exist_ok=True,
        seed=0,
        deterministic=True,
        # The pictures are always the same way up (a fixed camera looking down): no upside-down flips.
        # Left-right flips are fine (people come from both sides). Colour changes help with gloves, lighting.
        flipud=0.0,
        fliplr=0.5,
        plots=True,
        verbose=False,
    )
    minutes = (time.time() - t0) / 60
    n_train = sum(len(list((DATA / "images" / t).glob("*.jpg"))) for t in args.train)
    timing = {
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "pictures": round(n_train * args.fraction),
        "minutes": round(minutes, 2),
        "minutes_per_epoch": round(minutes / args.epochs, 2),
        "pictures_per_second": round(n_train * args.fraction * args.epochs / (minutes * 60), 1),
    }
    (RUNS / args.name / "timing.json").write_text(json.dumps(timing, indent=1))
    print(json.dumps(timing))


if __name__ == "__main__":
    main()
