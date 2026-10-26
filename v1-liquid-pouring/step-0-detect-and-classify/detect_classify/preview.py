"""Draw the labels over the pictures, to check them by eye.

python -m detect_classify.preview --split train --count 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .classes import CLASSES

# One colour per class (BGR), picked to be told apart.
COLOURS = [
    (60, 180, 75),
    (48, 130, 245),
    (25, 225, 255),
    (180, 30, 145),
    (230, 50, 240),
    (128, 128, 0),
    (0, 0, 230),
]


def draw(image: Path, labels: Path) -> np.ndarray:
    picture = cv2.imread(str(image))
    h, w = picture.shape[:2]
    for line in labels.read_text().splitlines():
        cls, *xy = line.split()
        points = (np.array(xy, dtype=float).reshape(-1, 2) * [w, h]).astype(np.int32)
        colour = COLOURS[int(cls)]
        cv2.polylines(picture, [points], isClosed=True, color=colour, thickness=2)
        x, y = points.min(axis=0)
        cv2.putText(
            picture, CLASSES[int(cls)], (int(x), int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1
        )
    return picture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--out", type=Path, default=Path("data/preview"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    images = sorted((args.data / args.split / "images").glob("*.jpg"), key=lambda p: int(p.stem))
    for image in images[: args.count]:
        labels = args.data / args.split / "labels" / f"{image.stem}.txt"
        path = args.out / f"{args.split}_{image.stem}.jpg"
        cv2.imwrite(str(path), draw(image, labels))
        print(path)


if __name__ == "__main__":
    main()
