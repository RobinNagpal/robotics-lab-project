"""Turn a detected object's mask and the depth picture into where it stands.

Both the programmed and the learned method end here: the detector only says
which pixels belong to an object; the position comes from depth and the
camera's calibration (docs/solutions/step-0-detect-and-classify.md, "How to
locate an object").
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .camera import Calibration

CONTAINERS = ("test_tube",)
# An object's top surface: its points within this much of its highest point.
# For a tube this is the rim, which is centred on its axis; seen from an
# angle the camera also sees the near side of the wall, which would pull the
# middle towards the camera. For a rack it is the top plate.
TOP_BAND = 0.01


@dataclass(frozen=True)
class Located:
    x: float
    y: float
    top: float  # highest point above the table, m


def locate(cls: str, mask: np.ndarray, depth: np.ndarray, calib: Calibration) -> Located | None:
    """Where the object in ``mask`` stands, in the robot's base frame.

    For a rack, take out first the pixels of the tubes standing in it
    (see own_pixels), or the tubes' rims become its top.
    """
    v, u = np.nonzero(mask)
    d = depth[v, u]
    keep = d > 0
    if keep.sum() < 3:
        return None
    points = calib.deproject(u[keep].astype(float), v[keep].astype(float), d[keep])
    top = float(np.percentile(points[:, 2], 99))
    own = points[points[:, 2] > top - TOP_BAND]
    if cls in CONTAINERS:
        return Located(float(own[:, 0].mean()), float(own[:, 1].mean()), top)
    # Racks are rectangles. Tubes seen at an angle hide
    # part of a rack's top on one side, which would pull an average off; the
    # smallest rectangle around what is seen still has the rack's outer edge.
    (x, y), _, _ = cv2.minAreaRect(own[:, :2].astype(np.float32))
    return Located(float(x), float(y), top)


def own_pixels(detected: list[tuple[str, np.ndarray]]) -> list[np.ndarray]:
    """Each detection's mask without the containers standing on it.

    A rack's outline covers the tubes in it. Tubes keep their whole mask.
    """
    held = np.zeros_like(detected[0][1]) if detected else None
    for cls, mask in detected:
        if cls in CONTAINERS:
            held |= mask
    return [mask if cls in CONTAINERS else mask & ~held for cls, mask in detected]
