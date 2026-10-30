"""The two ways to choose a grasp: a programmed rule on Step 0's list, and the model on the close-up.

Both answer the same question for one target tube: where the pinch point
goes on the table, at which of the 18 turns, and how high the rim is (which
sets the grip height). Or they say there is no clear turn, and the arm
should pick another tube.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from detect_classify.camera import Calibration

from . import rules

# The model reads the best pixel this close to the middle of the picture,
# where the camera aimed at the target: Step 0 is rarely more than 12 mm off
# (8 pixels), and a neighbour's centre is at least 36 mm (22 pixels) away.
SEARCH = 12
# Below this score the model says there is no clear turn. Chosen on the
# validation pictures: the labels are soft bumps whose peak is often under 1,
# so a clear turn's score can be as low as 0.5, while boxed-in tubes mostly
# stay under 0.05. 0.25 answers 96.1% of validation targets right, 0.5 94.8%.
THRESHOLD = 0.25
# The rim is the highest ring around the chosen pixel; inside a tube the
# camera sees the liquid or the bottom.
RIM_SEARCH = 10


@dataclass(frozen=True)
class Grasp:
    xy: np.ndarray  # pinch point on the table, m
    yaw: float  # radians
    top: float  # rim height, m
    score: float


# ------------------------------------------------------------------ programmed


@dataclass(frozen=True)
class Footprint:
    """The open fingers seen from above, as a rectangle around the pinch point.

    ``reach`` is how far it runs along the opening direction from the middle,
    ``half_width`` how far across. A neighbour whose wall comes inside it
    blocks that turn. Fitted on training scenes (evaluate.fit_footprint).
    """

    reach: float = 0.060
    half_width: float = 0.016


def programmed(target: dict, scene_step0: list[dict], footprint: Footprint) -> Grasp | None:
    """The turn that keeps the open fingers furthest from every neighbour Step 0 reported."""
    xy = np.array(target["xy"])
    low = rules.grip_height(target["top"])
    best, best_room = None, -math.inf
    for k in range(rules.BINS):
        yaw = rules.bin_yaw(k)
        along = np.array([math.cos(yaw), math.sin(yaw)])
        across = np.array([-along[1], along[0]])
        room = math.inf
        for other in scene_step0:
            if other["name"] == target["name"] or other["top"] < low - 0.02:
                continue  # itself, or too short to reach the fingers
            d = np.array(other["xy"]) - xy
            a, c = abs(d @ along), abs(d @ across)
            # How far the neighbour's wall is outside the rectangle (negative: inside).
            out_a = a - footprint.reach - rules.TUBE_RADIUS
            out_c = c - footprint.half_width - rules.TUBE_RADIUS
            room = min(room, max(out_a, out_c))
        if room > best_room:
            best, best_room = yaw, room
    if best_room <= 0:
        return None
    return Grasp(xy, best, target["top"], best_room)


# ------------------------------------------------------------------ learned


def model_scores(net: torch.nn.Module, heights: np.ndarray, device: str) -> np.ndarray:
    """Scores from 0 to 1, (N, 18, H, W), for height pictures (N, H, W)."""
    with torch.no_grad():
        x = torch.from_numpy(heights.astype(np.float32))[:, None].to(device)
        return torch.sigmoid(net(x)).cpu().numpy()


def learned(
    scores: np.ndarray, heights: np.ndarray, calib: Calibration, threshold: float | None = None
) -> tuple[Grasp | None, tuple]:
    """The best turn at the best pixel near the middle, turned into a place on the table.

    Also returns (u, v, bin) of the choice, for drawing.
    """
    size = heights.shape[0]
    v, u = np.mgrid[0:size, 0:size]
    centre = (size - 1) / 2
    near = (u - centre) ** 2 + (v - centre) ** 2 <= SEARCH**2
    masked = np.where(near[None], scores, -1.0)
    k, pv, pu = np.unravel_index(int(np.argmax(masked)), masked.shape)
    score = float(scores[k, pv, pu])
    ring = (u - pu) ** 2 + (v - pv) ** 2 <= RIM_SEARCH**2
    top = float(heights[ring].max())
    if score < (THRESHOLD if threshold is None else threshold):
        return None, (pu, pv, k)
    depth = calib.pos[2] - top
    point = calib.deproject(np.array([pu], float), np.array([pv], float), np.array([depth]))[0]
    return Grasp(point[:2], rules.bin_yaw(k), top, score), (pu, pv, k)
