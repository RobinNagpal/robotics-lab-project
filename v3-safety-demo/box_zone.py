"""From a hand box in the safety camera's picture to a zone: the rule after the detector.

The detector says where in the picture a hand is; it does not say how high above the table the hand is, and
from one camera looking down, height and distance mix: a hand high up looks further from the arm than it is.
So the rule takes the worst case. Every point of the box is a ray from the camera; along each ray, the hand
could be anywhere from the table up to ``HAND_TOP``. The hand's reach is taken as the closest any of those
points comes to the arm's base, and the zone follows from that.

This never puts a hand in a safer zone than it is in (if the box covers the hand and the hand is below
``HAND_TOP``); the price is some stops and slow-downs that weren't needed, near the zones' edges. Package 2
measures both.

With depth (the safety camera is an RGB-D camera, as 3D safety cameras are), the height is measured instead:
``depth_reach`` turns every pixel in the box into a point in space, from its depth, and takes the closest one
to the arm's base. The bare table seen around the hand is left out (it is not the hand). This is the rule the
safety feature uses; the worst-case rule above is the fallback when a box has no depth.
"""

from __future__ import annotations

import mujoco
import numpy as np

import cameras
import zones

HAND_TOP = 0.6  # m above the table: the highest a hand is taken to be when reaching over the bench


def box_reach(model: mujoco.MjModel, data: mujoco.MjData, box, camera: str = cameras.SAFETY.name) -> float:
    """The closest a hand in ``box`` (x0, y0, x1, y1 pixels) can be to the arm's base, seen from above, m.

    Exact, not sampled: the places the box could mean form a frustum (the box's rays, cut at the table and at
    ``HAND_TOP``). Seen from above, that is the outline around its 8 corners, and the answer is the outline's
    distance from the arm's base (0 if the base is inside it).
    """
    x0, y0, x1, y1 = box
    px = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)
    pos, rays = cameras.pixel_rays(model, data, camera, px, cameras.SAFETY.width, cameras.SAFETY.height)
    corners = [pos[:2] + (z - pos[2]) / rays[:, 2:3] * rays[:, :2] for z in (0.0, HAND_TOP)]
    hull = _hull(np.concatenate(corners) - zones.ARM_BASE)
    return _distance_from_origin(hull)


def _hull(points: np.ndarray) -> np.ndarray:
    """The convex outline of 2D points, anticlockwise (Andrew's monotone chain)."""
    pts = sorted(map(tuple, points))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _distance_from_origin(hull: np.ndarray) -> float:
    """Distance from (0, 0) to a convex outline (anticlockwise); 0 if (0, 0) is inside."""
    a, b = hull, np.roll(hull, -1, axis=0)
    edge = b - a
    if np.all(edge[:, 0] * (-a[:, 1]) - edge[:, 1] * (-a[:, 0]) >= 0):
        return 0.0
    t = np.clip(np.einsum("ij,ij->i", -a, edge) / np.einsum("ij,ij->i", edge, edge), 0.0, 1.0)
    closest = a + t[:, None] * edge
    return float(np.linalg.norm(closest, axis=1).min())


def zone_of_reach(reach: float) -> zones.Zone:
    if reach < zones.STOP_RADIUS:
        return zones.Zone.STOP
    if reach < zones.WARNING_RADIUS:
        return zones.Zone.WARNING
    return zones.Zone.CLEAR


def zone_of_boxes(model: mujoco.MjModel, data: mujoco.MjData, boxes) -> zones.Zone:
    """The zone for all the hands found in one picture: the most serious one; clear if none."""
    return max((zone_of_reach(box_reach(model, data, b)) for b in boxes), default=zones.Zone.CLEAR)


TABLE_SKIP = 0.01  # m: points this close to the table's top are the table itself, not the hand
# m taken off the measured reach: covers the depth picture's precision (millimetres, limited further by the
# Mac's OpenGL) and the pixels skipped by STRIDE. Without it, 2 of 3,000 test hands right on the warning
# zone's edge were placed 1 to 5 mm outside it.
DEPTH_MARGIN = 0.02
STRIDE = 2  # use every second pixel of the box (the hand is hundreds of pixels; this halves the work twice)


def depth_reach(
    model: mujoco.MjModel, data: mujoco.MjData, box, depth: np.ndarray, camera: str = cameras.SAFETY.name
) -> float:
    """The closest point in ``box`` to the arm's base, seen from above, from the depth picture.

    ``depth`` is in metres along the camera's axis. Falls back to the worst-case rule if every pixel in the
    box is bare table.
    """
    h, w = depth.shape
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    ys, xs = np.mgrid[y0:y1:STRIDE, x0:x1:STRIDE]
    if xs.size == 0:
        return box_reach(model, data, box, camera)
    px = np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], axis=1)
    pos, rays = cameras.pixel_rays(model, data, camera, px, w, h)
    points = pos + depth[ys.ravel(), xs.ravel()][:, None] * rays  # rays have length 1 along the camera's axis
    points = points[points[:, 2] > TABLE_SKIP]
    if len(points) == 0:
        return box_reach(model, data, box, camera)
    return float(np.linalg.norm(points[:, :2] - zones.ARM_BASE, axis=1).min()) - DEPTH_MARGIN


def zone_of_boxes_depth(model: mujoco.MjModel, data: mujoco.MjData, boxes, depth: np.ndarray) -> zones.Zone:
    """As ``zone_of_boxes``, with the depth picture."""
    return max((zone_of_reach(depth_reach(model, data, b, depth)) for b in boxes), default=zones.Zone.CLEAR)
