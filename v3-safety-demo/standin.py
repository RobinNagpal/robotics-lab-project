"""Stand-ins for Track A's parts, so the safety layer can be built and tested before they exist.

- **Zones** (``zone_of``): how far a hand is from the arm, and so which zone it is in. The arm is drawn as a
  chain of thick lines through its joints (each ``ARM_RADIUS`` thick), and a hand point's distance is to the
  nearest of them. Warning closer than ``WARNING_DISTANCE``, stop closer than ``STOP_DISTANCE``. These sizes
  are
  placeholders: Track A's ``zones.py`` settles the real ones, and its function replaces ``zone_of`` with the
  same call.
- **A scripted hand** (``ScriptedHand``): a single point placed beside the gripper at set times, at a set
  distance. It is not in the scene (nothing to see or touch); only the "perfect" detector, which reads the
  simulation, knows where it is. Track A's simulated hand replaces it.

The calls the safety layer uses:

    hand.points(model, data) -> (k, 3) array of hand points in metres, or None when no hand is in the cell
    arm_points(model, data)  -> (m, 3) array: the arm's chain, base to fingertips
    zone_of(hand_points, arm_points) -> (zone, distance in metres)
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from detector import CLEAR, STOP, WARNING

WARNING_DISTANCE = 0.35  # m from the arm's surface (placeholder until zones.py)
STOP_DISTANCE = 0.15  # m
ARM_RADIUS = 0.06  # m: the arm's links, roughly, as thick lines
ARM_CHAIN = (
    "arm/base",
    "arm/shoulder_link",
    "arm/upper_arm_link",
    "arm/forearm_link",
    "arm/wrist_1_link",
    "arm/wrist_2_link",
    "arm/wrist_3_link",
)
PINCH = "arm/gripper/pinch"


def arm_points(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """The arm's chain from base to the pinch point between the fingers, as it is now (m)."""
    ids = [model.body(n).id for n in ARM_CHAIN]
    return np.vstack([data.xpos[ids], data.site_xpos[model.site(PINCH).id]]).copy()


def distance_to_chain(points: np.ndarray, chain: np.ndarray) -> float:
    """The smallest distance from any of ``points`` to the line pieces joining ``chain``'s points."""
    p = np.atleast_2d(points)[:, None, :]  # (k, 1, 3)
    a, b = chain[:-1][None], chain[1:][None]  # (1, m-1, 3)
    ab = b - a
    t = np.clip(((p - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0.0, 1.0)
    nearest = a + t[..., None] * ab
    return float(np.linalg.norm(p - nearest, axis=-1).min())


def zone_of(hand: np.ndarray, arm: np.ndarray) -> tuple[str, float]:
    """The zone a hand at ``hand`` (points) is in, and its distance from the arm's surface (m)."""
    d = max(0.0, distance_to_chain(hand, arm) - ARM_RADIUS)
    if d < STOP_DISTANCE:
        return STOP, d
    if d < WARNING_DISTANCE:
        return WARNING, d
    return CLEAR, d


@dataclass(frozen=True)
class Visit:
    """The hand beside the gripper from ``start`` to ``end`` (simulation seconds), ``distance`` from the
    arm."""

    start: float
    end: float
    distance: float


class ScriptedHand:
    """A hand point placed beside the gripper during each visit, on the side away from the arm's base."""

    def __init__(self, visits: list[Visit]):
        self.visits = sorted(visits, key=lambda v: v.start)

    def visit_at(self, t: float) -> Visit | None:
        for v in self.visits:
            if v.start <= t < v.end:
                return v
        return None

    def points(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray | None:
        v = self.visit_at(data.time)
        if v is None:
            return None
        tool = data.site_xpos[model.site(PINCH).id]
        base = data.xpos[model.body("arm/base").id]
        out = np.array([tool[0] - base[0], tool[1] - base[1], 0.0])
        n = np.linalg.norm(out)
        out = out / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
        return (tool + out * (ARM_RADIUS + v.distance))[None].copy()


def repeating_visits(
    first: float, every: float, until: float, warn: float = 4.0, stop: float = 2.0
) -> list[Visit]:
    """Every ``every`` seconds from ``first``: ``warn`` s in the warning zone, then ``stop`` s in the stop
    zone.

    The distances sit well inside each zone (0.27 m and 0.05 m from the arm) so the zone is clear-cut.
    """
    out = []
    t = first
    while t < until:
        out.append(Visit(t, t + warn, 0.27))
        out.append(Visit(t + warn, t + warn + stop, 0.05))
        t += every
    return out
