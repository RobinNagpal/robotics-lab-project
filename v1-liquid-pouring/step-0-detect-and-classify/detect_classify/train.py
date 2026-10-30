"""Train YOLO11-seg on the rendered pictures.

    python -m detect_classify.train --name yolo11s_800
    python -m detect_classify.train --fraction 0.25 --name yolo11s_200   # first 25 % of the training set

The settings are the ones in docs/results.md; change them there and here together.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default="yolo11s-seg.pt", help="COCO-trained weights to start from")
    parser.add_argument("--data", type=Path, default=Path("data/dataset.yaml"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10, help="stop after this many epochs without gain")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--fraction", type=float, default=1.0, help="share of the training pictures to use")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(args.data.resolve()),
        imgsz=args.imgsz,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        fraction=args.fraction,
        device=args.device,
        workers=args.workers,
        project=str(Path("runs").resolve()),
        name=args.name,
        exist_ok=True,
        seed=0,
        deterministic=True,
        # The scenes already change colour, so push colour much further than
        # the default: the detector must learn shapes, not colours.
        hsv_h=0.5,
        hsv_s=0.7,
        hsv_v=0.4,
        # A straight-down camera has no up or down; both flips are real pictures.
        fliplr=0.5,
        flipud=0.5,
        plots=True,
    )


if __name__ == "__main__":
    main()
