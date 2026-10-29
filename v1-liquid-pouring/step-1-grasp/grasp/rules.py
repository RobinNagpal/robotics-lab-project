"""The programmed parts of a grasp: the turn angles, the grip height, the squeeze, and Step 0's errors.

The model only chooses where the fingers close and at what turn. Everything
here is fixed by the tube and the gripper, so it is worked out, not learned
(docs/solutions/step-1-grasp.md).

A turn (yaw) is the direction the two fingers open along, measured on the
table from the x axis towards the y axis. A two-finger gripper turned by
180 degrees is the same grasp, so turns run from 0 to 180 degrees.
"""

from __future__ import annotations

import math

import numpy as np
from lab_scene.settings import RACK

# 18 turns, 10 degrees apart: 0, 10, ..., 170.
BINS = 18
BIN_STEP = math.pi / BINS
# Tube walls are 25 mm across.
TUBE_RADIUS = 0.0125
# The fingers close 3 cm below the rim, which keeps the opening clear and the
# pads fully on the glass, but never lower than 3 cm above the rack's top
# plate, or the pads catch on it (simulation-environment/docs: Gripping a tube).
BELOW_RIM = 0.03
ABOVE_RACK = 0.03
# The way down starts this far above the rim.
APPROACH = 0.05
# The lift after closing.
LIFT = 0.10

# Step 0's position errors on its 300 test pictures (step-0 docs/results.md):
# median about 1 mm, 95 % under about 10 mm, and about 3 % of tubes missed.
# Most estimates are close; a few are well off.
STEP0_CLOSE = 0.0007  # standard deviation per axis, m, for most tubes
STEP0_FAR = (0.003, 0.012)  # distance range, m, for the few that are well off
STEP0_FAR_SHARE = 0.12
STEP0_TOP = 0.002  # standard deviation of the top height, m
STEP0_MISSED = 0.03


def bin_yaw(k: int | np.ndarray) -> float | np.ndarray:
    """The turn of bin ``k``, radians."""
    return k * BIN_STEP


def yaw_bin(yaw: float) -> int:
    """The bin nearest a turn, radians."""
    return int(round(yaw / BIN_STEP)) % BINS


def grip_height(top: float) -> float:
    """Where the pinch point between the fingers goes, for a tube whose rim is at ``top``."""
    return max(top - BELOW_RIM, RACK.height + ABOVE_RACK)


def rotate_bins(labels: np.ndarray, quarter_turns: int, axis: int = 0, step: float = BIN_STEP) -> np.ndarray:
    """Bins after the picture is turned anticlockwise by ``quarter_turns`` x 90 degrees.

    A turn of the picture is a turn of the scene: every grasp turns with it.
    90 degrees is 9 bins of 10 degrees (or 6 of 15, for side grasps).
    """
    return np.roll(labels, round(math.pi / 2 / step) * quarter_turns, axis=axis)


def mirror_bins(labels: np.ndarray, axis: int = 0, step: float = BIN_STEP) -> np.ndarray:
    """Bins after the picture is mirrored left to right: a direction t becomes 180 - t."""
    bins = labels.shape[axis]
    order = (round(math.pi / step) - np.arange(bins)) % bins
    return np.take(labels, order, axis=axis)


def step0_estimate(rng: np.random.Generator, xy: np.ndarray, top: float) -> tuple[np.ndarray, float]:
    """Where Step 0 would say a tube stands: the truth plus errors like Step 0's own."""
    if rng.random() < STEP0_FAR_SHARE:
        angle = rng.uniform(0, 2 * math.pi)
        error = rng.uniform(*STEP0_FAR) * np.array([math.cos(angle), math.sin(angle)])
    else:
        error = rng.normal(0, STEP0_CLOSE, 2)
    return xy + error, top + float(rng.normal(0, STEP0_TOP))
