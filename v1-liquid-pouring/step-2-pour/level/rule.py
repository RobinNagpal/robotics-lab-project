"""The programmed rule to beat: find the blue balls in the picture, read the level from how high they reach.

1. A pixel is a ball if it is clearly bluer than it is red (empty glass is grey-blue: blue at most about
   15 above red; the balls 25 to 110 above).
2. Each row of the tube's middle, below Arm A's fingers, counts as full if enough of it is ball.
3. From the bottom of the tube up, the full rows are followed (a gap of a few rows is allowed, between
   balls); where they end is the top of the liquid. Balls still falling, higher up, do not count, and
   nothing counts unless it starts at the bottom of the tube.
4. That row is turned into a level by a straight line fitted on the training pictures, one per tube size
   (Step 0 tells the size).

The flow is the change in that level over the stack's 0.2 s. The rule gives no estimate of the balls in
the air.

    python -m level.rule
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import Pictures

BLUER = 22  # blue minus red, 0-255
COLUMNS = slice(14, 34)  # the tube's middle in the 48-pixel-wide picture
# Only below Arm A's fingers (rows 34 to 56; the rim is at about row 24). Above them the source's own
# balls show at the lip, and without this the rule read 100 % in every test pour while it swung in.
# The liquid never reaches this high: 65 % of a medium tube is at about row 66.
BELOW = 56
# The bottom of each tube in the picture (row), measured on empty training pictures. The liquid must
# start there: a ball still falling through an empty tube is not a level.
BOTTOM = {0.15: 141, 0.2: 180}
NEAR_BOTTOM = 8  # rows
FULL = 0.3  # share of a row that must be ball
GAP = 3  # rows


def top_row(image: np.ndarray, size: float) -> float:
    """How many rows up from the bottom of the picture the liquid reaches (0 if none)."""
    x = image[BELOW:, COLUMNS].astype(int)
    ball = (x[..., 2] - x[..., 0]) > BLUER
    full = ball.mean(axis=1) >= FULL
    rows = np.where(full)[0]
    floor = BOTTOM[round(float(size), 2)] - BELOW
    if len(rows) == 0 or rows[-1] < floor - NEAR_BOTTOM:
        return 0.0
    bottom = rows[-1]
    top = bottom
    for r in rows[::-1][1:]:
        if top - r > GAP:
            break
        top = r
    return float(len(full) - top)


def heights(data: Pictures) -> np.ndarray:
    return np.array(
        [top_row(data.images[i].permute(1, 2, 0).numpy(), data.size[i]) for i in range(len(data))]
    )


def fit(data: Pictures) -> dict[str, list[float]]:
    """One straight line per tube size, from rows to level, on pictures with liquid in them."""
    h = heights(data)
    lines = {}
    for size in np.unique(data.size):
        use = (data.size == size) & (data.labels[:, 0] > 3) & (h > 0)
        lines[f"{size:.2f}"] = np.polyfit(h[use], data.labels[use, 0], 1).tolist()
    return lines


def predict(lines: dict[str, list[float]], data: Pictures) -> np.ndarray:
    """(level, flow, air) for every picture; air is not estimated (NaN)."""
    h = heights(data)
    level = np.zeros(len(data))
    for size, line in lines.items():
        mine = np.isclose(data.size, float(size))
        level[mine] = np.where(h[mine] > 0, np.polyval(line, h[mine]), 0.0)
    level = np.clip(level, 0, 100)
    earlier = np.maximum(np.arange(len(data)) - 2, data.first)
    flow = (level - level[earlier]) / 0.2
    return np.stack([level, flow, np.full(len(data), np.nan)], 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=Path("data/level"))
    parser.add_argument("--out", type=Path, default=Path("runs/rule"))
    args = parser.parse_args()
    lines = fit(Pictures(args.data / "train.npz"))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "lines.json").write_text(json.dumps(lines, indent=1))
    print(json.dumps(lines))


if __name__ == "__main__":
    main()
