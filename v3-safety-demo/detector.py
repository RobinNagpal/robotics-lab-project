"""The detector interface: one call takes a camera frame and says which zone a hand is in.

    detection = detector.detect(frame)
    detection.zone           # "clear", "warning" or "stop"
    detection.score          # how sure the model is of that zone, 0 to 1
    detection.model_version  # the version it was loaded as (checked against the registry)

A ``Frame`` is what a camera hands over: its number (counting up from 1), the simulation time it was taken at,
the picture (an image array, or None for the truth camera), a depth picture if the camera has one (Track A's
safety camera is RGB-D), and, for the "perfect" detector only, the simulation's truth at that moment (where the
hand and the arm are).

Two pieces live here:

- ``PerfectDetector``: reads the hand's true position from the frame's truth and works out the zone exactly. It
  stands in for the trained model until that exists, and stays as the baseline the model is compared with.
- ``TruthCamera``: the camera that goes with it. It takes no picture; it records where the hand and the arm are.

The trained model (Track A, phase 2) implements the same ``detect`` and is loaded with ``load``, which goes
through the model registry. A rendering camera must add sensor noise to its pictures, as a real camera has:
the watchdog takes several identical pictures in a row to mean the camera has frozen.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import mujoco
import numpy as np

CLEAR, WARNING, STOP = "clear", "warning", "stop"
ZONES = (CLEAR, WARNING, STOP)


@dataclass(frozen=True)
class Frame:
    number: int
    time: float  # simulation seconds when it was taken
    image: np.ndarray | None = None
    truth: dict[str, Any] | None = field(default=None, compare=False)
    camera: str = "overhead"
    depth: np.ndarray | None = field(default=None, compare=False)  # metres per pixel, for an RGB-D camera


@dataclass(frozen=True)
class Detection:
    zone: str
    score: float
    model_version: str


class Detector(Protocol):
    version: str

    def detect(self, frame: Frame) -> Detection: ...


class Camera(Protocol):
    def capture(self, model: mujoco.MjModel, data: mujoco.MjData, number: int) -> Frame | None: ...


def image_digest(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


class PerfectDetector:
    """The zone straight from the simulation's truth. Always right, always sure (score 1)."""

    version = "perfect-1"

    def __init__(self, zone_fn: Callable[[np.ndarray, np.ndarray], tuple[str, float]] | None = None):
        if zone_fn is None:
            from standin import zone_of as zone_fn
        self.zone_fn = zone_fn

    def detect(self, frame: Frame) -> Detection:
        if frame.truth is None:
            raise ValueError("the perfect detector needs a frame from the truth camera")
        hand = frame.truth.get("hand")
        if hand is None or len(hand) == 0:
            return Detection(CLEAR, 1.0, self.version)
        zone, _ = self.zone_fn(hand, frame.truth["arm"])
        return Detection(zone, 1.0, self.version)


class TruthCamera:
    """Records where the hand and the arm are, at each frame. Also keeps the smallest hand-arm distance seen."""

    def __init__(self, hand=None, arm_points=None, zone_fn=None):
        import standin

        self.hand = hand
        self.arm_points = arm_points or standin.arm_points
        self.zone_fn = zone_fn or standin.zone_of
        self.min_distance = math.inf
        self.last_distance: float | None = None

    def capture(self, model: mujoco.MjModel, data: mujoco.MjData, number: int) -> Frame:
        hand = self.hand.points(model, data) if self.hand is not None else None
        arm = self.arm_points(model, data)
        self.last_distance = None
        if hand is not None:
            self.last_distance = self.zone_fn(hand, arm)[1]
            self.min_distance = min(self.min_distance, self.last_distance)
        return Frame(number, float(data.time), None, {"hand": hand, "arm": arm})


def load(model_path, registry, log, build: Callable[[bytes, str], Detector]) -> Detector:
    """A trained detector from ``model_path``, only if the registry approves the file.

    ``build(blob, version)`` turns the approved bytes into a detector. Refusals raise
    ``registry.UnapprovedModel`` and are logged.
    """
    blob, entry = registry.read_approved(model_path, log)
    det = build(blob, entry["version"])
    if det.version != entry["version"]:
        raise ValueError(f"the detector calls itself {det.version}, the registry says {entry['version']}")
    return det
