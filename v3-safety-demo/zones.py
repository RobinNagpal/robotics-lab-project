"""The safety zones, and the truth about where the person is: the labels the detector learns from.

Two fields around the arm's base, as a safety laser scanner or a safety camera would have:

- **Stop zone**: closer than ``STOP_RADIUS`` to the arm's base (seen from above). That is the UR5e's reach
  (850 mm), so anything the arm could touch is in it. A hand here holds the arm.
- **Warning zone**: from ``STOP_RADIUS`` to ``WARNING_RADIUS``. A hand here slows the arm to 25%.
- Outside both: clear.

Both reach from the table up to ``ZONE_TOP``. The bench's back edge, behind the arm, is taken to be against a
wall, so people come from the front (+x) and the two sides (+y and -y).

A person counts as in a zone when any point of their arm or hand is in it. ``person_points`` gives the points
checked: the fingertips, the hand's corners, and points along the forearm and upper arm.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import mujoco
import numpy as np

import hand

STOP_RADIUS = 0.85  # m from the arm's base axis, seen from above: the UR5e's reach
WARNING_RADIUS = 1.15  # m
ZONE_TOP = 1.0  # m above the table
ARM_BASE = np.array([0.0, 0.0])  # the arm's base, x y (v3 puts it at the origin)
ROBOT_PREFIX = "arm/"


class Zone(IntEnum):
    CLEAR = 0
    WARNING = 1
    STOP = 2


def zone_of_point(p: np.ndarray) -> Zone:
    """The zone a single point (x, y, z) is in."""
    if p[2] > ZONE_TOP:
        return Zone.CLEAR
    r = float(np.linalg.norm(p[:2] - ARM_BASE))
    if r < STOP_RADIUS:
        return Zone.STOP
    if r < WARNING_RADIUS:
        return Zone.WARNING
    return Zone.CLEAR


def person_points(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Points on the person's surface, world frame (n, 3): the ends and sides of every shape.

    Checked against the zones. Taken from the compiled shapes, so a larger hand reaches further.
    """
    pts = []
    for name in hand.SHAPES:
        gid = model.geom(hand.PREFIX + name).id
        centre, rot, size = data.geom_xpos[gid], data.geom_xmat[gid].reshape(3, 3), model.geom_size[gid]
        if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
            corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * size
            pts.extend(centre + corners @ rot.T)
        else:  # capsule: its two end caps, and four sides at each end
            r, half = size[0], size[1]
            for end in (-1, 1):
                axis_end = centre + rot[:, 2] * half * end
                pts.append(axis_end + rot[:, 2] * r * end)
                for side in (rot[:, 0], -rot[:, 0], rot[:, 1], -rot[:, 1]):
                    pts.append(axis_end + side * r)
    return np.array(pts)


@dataclass(frozen=True)
class Truth:
    """Where the person is in one frame, from the simulation."""

    zone: Zone  # the deepest zone any point of the person is in
    reach: float  # m: the person's closest point to the arm's base axis, seen from above (inf if parked)
    visible: bool  # False when the person is parked out of sight


def truth(model: mujoco.MjModel, data: mujoco.MjData) -> Truth:
    m = model.body(hand.BODY).mocapid[0]
    if np.allclose(data.mocap_pos[m], hand.PARKED):
        return Truth(Zone.CLEAR, float("inf"), False)
    pts = person_points(model, data)
    pts = pts[pts[:, 2] <= ZONE_TOP]
    if len(pts) == 0:
        return Truth(Zone.CLEAR, float("inf"), True)
    r = np.linalg.norm(pts[:, :2] - ARM_BASE, axis=1)
    closest = float(r.min())
    zone = Zone.STOP if closest < STOP_RADIUS else Zone.WARNING if closest < WARNING_RADIUS else Zone.CLEAR
    return Truth(zone, closest, True)


def robot_geoms(model: mujoco.MjModel) -> np.ndarray:
    """The robot's shapes that can collide (arm and gripper), for the distance check."""
    ids = []
    for g in range(model.ngeom):
        body = model.body(model.geom_bodyid[g]).name
        if body.startswith(ROBOT_PREFIX) and (model.geom_contype[g] or model.geom_conaffinity[g]):
            ids.append(g)
    return np.array(ids)


def person_geoms(model: mujoco.MjModel) -> np.ndarray:
    return np.array([model.geom(hand.PREFIX + n).id for n in hand.SHAPES])


def distance_to_robot(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot: np.ndarray | None = None,
    person: np.ndarray | None = None,
    distmax: float = 0.5,
) -> float:
    """The smallest gap between the person and the robot, m (negative: they overlap, a contact event).

    Returns ``distmax`` when they are further apart than that. Uses MuJoCo's own shape distance, which works
    on shapes that don't collide.
    """
    robot = robot_geoms(model) if robot is None else robot
    person = person_geoms(model) if person is None else person
    best = distmax
    for p in person:
        for g in robot:
            d = mujoco.mj_geomDistance(model, data, int(p), int(g), distmax, None)
            best = min(best, d)
    return best


ZONE_NAMES = {Zone.CLEAR: "clear", Zone.WARNING: "warning", Zone.STOP: "stop"}  # as the detector names them


def zone_of_points(points: np.ndarray | None, arm=None) -> tuple[str, float]:
    """The zone for a set of person points (n, 3), and how close they come to the arm's base (m, seen from
    above). The call the safety layer's perfect detector uses; ``arm`` is not needed (the zones are fixed)."""
    if points is None or len(points) == 0:
        return "clear", float("inf")
    pts = np.atleast_2d(points)
    pts = pts[pts[:, 2] <= ZONE_TOP]
    if len(pts) == 0:
        return "clear", float("inf")
    reach = float(np.linalg.norm(pts[:, :2] - ARM_BASE, axis=1).min())
    zone = Zone.STOP if reach < STOP_RADIUS else Zone.WARNING if reach < WARNING_RADIUS else Zone.CLEAR
    return ZONE_NAMES[zone], reach
