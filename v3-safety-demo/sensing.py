"""What the safety feature senses during a real v3 job: a person reaching in, seen by the RGB-D safety camera,
judged by a detector.

- ``Person``: a technician at the bench who reaches in now and then while the robot works (Track A's hand,
  ``hand.py``), with a new look for every reach. It is moved before every physics step (``Guard.before_physics``),
  also while the arm is held. Every 10 ms while the hand is near, it measures the gap between the person and the
  robot and whether the robot is moving: a **contact event** is a gap below zero, and one **while the robot
  moves** is what the safety feature must prevent.
- ``SafetyCamera``: the ceiling camera's colour and depth pictures, with sensor noise as a real camera has (the
  watchdog takes identical pictures in a row for a frozen camera). It also hands the simulation's truth along,
  for the perfect detector and for the record; the trained detector never reads it.
- ``YoloDetector``: the trained model (YOLO11n) finds hands in the colour picture; ``box_zone``'s fixed rule
  turns each box and the depth inside it into a zone. Built only from registry-approved bytes and settings.
- ``perfect_detector()``: the baseline, the zone straight from the truth.

The zones are the fixed ones in ``zones.py``: stop closer than 0.85 m to the arm's base, warning to 1.15 m.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import mujoco
import numpy as np
from centrifuge_scene import scene

import box_zone
import cameras
import hand
import zones
from detector import CLEAR, Detection, Frame, PerfectDetector

MONITOR_EVERY = 10  # physics steps (10 ms) between gap checks while the hand is near
NEAR = 1.0  # m: the gap to the robot is measured while the person comes this close to the arm's base
MOVING = 0.01  # rad/s: the robot counts as moving when any arm joint turns faster than this
RGB_NOISE = 2.0  # standard deviation of the colour noise, in 0-255 steps
DEPTH_NOISE = 0.002  # m, standard deviation of the depth noise (a good time-of-flight camera at 2 m)


# --- the person ---------------------------------------------------------------------------------------------


@dataclass
class Visit:
    """One reach of the person, and what happened during it."""

    start: float
    goal: str
    look: dict
    deepest: str = CLEAR  # the most serious zone the person was in (truth)
    reach: float = float("inf")  # the closest the person came to the arm's base, m
    gap: float = float("inf")  # the smallest gap to the robot, m
    gap_moving: float = float("inf")  # the smallest gap to the robot while it was moving, m
    contact_moving: int = 0  # 10 ms samples touching the robot while it moved
    contact_still: int = 0  # 10 ms samples touching the robot while it stood still (held)
    entered_stop: float | None = None  # when the person first entered the stop zone
    stopped: float | None = None  # when the robot was first still after that
    end: float | None = None

    def summary(self) -> dict:
        out = {k: v for k, v in self.__dict__.items()}
        for k in ("reach", "gap", "gap_moving"):
            out[k] = None if not np.isfinite(out[k]) else round(out[k], 4)
        if self.entered_stop is not None and self.stopped is not None:
            out["stop_after"] = round(self.stopped - self.entered_stop, 3)
        return out


class Person:
    """A person who reaches into the cell at random moments of a job.

    Between reaches the person waits ``pause`` seconds (drawn each time); the first reach comes ``first``
    seconds in. Each reach has a new look and a new path (``hand.draw_reach``: from the bench's edge, out of
    the zones, to a target outside the zones, in the warning zone, or in the stop zone).
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        rng: np.random.Generator,
        first: tuple[float, float] = (3.0, 12.0),
        pause: tuple[float, float] = (6.0, 20.0),
        goals: tuple[str, ...] | None = None,
        reaches: bool = True,
    ):
        self.rng = rng
        self.pause = pause
        self.goals = goals
        # reaches=False: nobody comes (runs that check the feature leaves a job alone with nobody there).
        self.next_start = float(data.time) + float(rng.uniform(*first)) if reaches else float("inf")
        self.reach: hand.Reach | None = None
        self.visit: Visit | None = None
        self.visits: list[Visit] = []
        self.robot = zones.robot_geoms(model)
        self.person = zones.person_geoms(model)
        self.dofs = np.array([model.joint(f"{scene.ARM_NAME}/{n}").dofadr[0] for n in scene.ARM_JOINTS])
        self.steps = 0
        hand.place(model, data, None, np.array([1.0, 0, 0, 0]))

    def before_physics(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        t = float(data.time)
        if self.reach is None and t >= self.next_start:
            look = hand.Look.draw(self.rng)
            hand.set_look(model, look)
            goal = str(self.rng.choice(self.goals)) if self.goals else None
            goal, self.reach = hand.draw_reach(self.rng, model, data, t, t, goal=goal)
            self.visit = Visit(start=t, goal=goal, look=look.__dict__.copy())
        if self.reach is None:
            return
        wrist = self.reach.wrist(t)
        if wrist is None:  # the reach is over: out of sight until the next one
            hand.place(model, data, None, np.array([1.0, 0, 0, 0]))
            self.visit.end = t
            self.visits.append(self.visit)
            self.reach, self.visit = None, None
            self.next_start = t + float(self.rng.uniform(*self.pause))
            return
        hand.place(model, data, wrist, self.reach.quat())
        self.steps += 1
        if self.steps % MONITOR_EVERY == 0:
            self._monitor(model, data, t)

    def _monitor(self, model: mujoco.MjModel, data: mujoco.MjData, t: float) -> None:
        v = self.visit
        truth = zones.truth(model, data)
        if not truth.visible:
            return
        if truth.zone > {"clear": 0, "warning": 1, "stop": 2}[v.deepest]:
            v.deepest = zones.ZONE_NAMES[truth.zone]
        v.reach = min(v.reach, truth.reach)
        moving = float(np.abs(data.qvel[self.dofs]).max()) > MOVING
        if truth.zone == zones.Zone.STOP and v.entered_stop is None:
            v.entered_stop = t
        if v.entered_stop is not None and v.stopped is None and not moving:
            v.stopped = t
        if truth.reach > NEAR:
            return
        gap = zones.distance_to_robot(model, data, self.robot, self.person)
        v.gap = min(v.gap, gap)
        if moving:
            v.gap_moving = min(v.gap_moving, gap)
        if gap < 0:
            if moving:
                v.contact_moving += 1
            else:
                v.contact_still += 1

    def points(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray | None:
        """The person's surface points (for the truth camera), or None when out of sight."""
        if self.reach is None:
            return None
        return zones.person_points(model, data)

    def finish(self, t: float) -> None:
        """Close a reach still going when the job ends."""
        if self.visit is not None:
            self.visit.end = t
            self.visits.append(self.visit)
            self.visit = None


# --- the camera ---------------------------------------------------------------------------------------------


class SafetyCamera:
    """The RGB-D safety camera, as the guard's ``Camera``: a frame with colour, depth and the truth.

    ``render=False`` gives frames with the truth only (no pictures): for runs with the perfect detector, which
    needs no pictures, at a fraction of the cost.
    """

    def __init__(self, person: Person, rng: np.random.Generator, render: bool = True):
        self.person = person
        self.rng = rng
        self.render = render
        self._rgb = self._depth = None
        self.last_distance: float | None = None  # for the guard's trace: how close to the arm's base

    def capture(self, model: mujoco.MjModel, data: mujoco.MjData, number: int) -> Frame:
        points = self.person.points(model, data)
        zone, reach = zones.zone_of_points(points)
        self.last_distance = None if not np.isfinite(reach) else round(reach, 4)
        truth = {"hand": points, "arm": None, "zone": zone, "reach": reach}
        image = depth = None
        if self.render:
            if self._rgb is None:
                w, h = cameras.SAFETY.width, cameras.SAFETY.height
                self._rgb = mujoco.Renderer(model, height=h, width=w)
                self._depth = mujoco.Renderer(model, height=h, width=w)
                self._depth.enable_depth_rendering()
            self._rgb.update_scene(data, camera=cameras.SAFETY.name)
            clean = self._rgb.render().astype(np.float32)
            noise = self.rng.normal(0.0, RGB_NOISE, clean.shape)
            image = np.clip(np.round(clean + noise), 0, 255).astype(np.uint8)
            self._depth.update_scene(data, camera=cameras.SAFETY.name)
            depth = self._depth.render()
            depth = (depth + self.rng.normal(0.0, DEPTH_NOISE, depth.shape)).astype(np.float32)
        return Frame(number, float(data.time), image, truth, camera=cameras.SAFETY.name, depth=depth)

    def close(self) -> None:
        for r in (self._rgb, self._depth):
            if r is not None:
                r.close()


# --- the detectors --------------------------------------------------------------------------------------------


def perfect_detector() -> PerfectDetector:
    """The baseline: the zone straight from the simulation's truth, with the fixed zones."""
    return PerfectDetector(zones.zone_of_points)


class YoloDetector:
    """The trained hand detector (YOLO11n) and the fixed rule (box + depth -> zone).

    Made only by ``detector.load`` from registry-approved bytes and settings (``threshold``: the confidence a
    box needs; ``imgsz``: the picture size the model runs at). The score is the confidence of the box that set
    the zone, or 1 when no hand was found.
    """

    def __init__(self, blob: bytes, version: str, settings: dict, device: str = "mps"):
        from ultralytics import YOLO

        self.version = version
        self.device = device
        self.threshold = float(settings["threshold"])
        self.imgsz = int(settings.get("imgsz", 640))
        # Ultralytics loads from a file: the approved bytes, written to a private temporary file and removed.
        fd, path = tempfile.mkstemp(suffix=".pt")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(blob)
            self._model = YOLO(path)
        finally:
            os.remove(path)
        self.cam_model, self.cam_data = cameras.camera_only()
        # The first prediction sets the model up (several hundred ms): done now, so the first real frame meets
        # the guard's time budget.
        blank = np.zeros((cameras.SAFETY.height, cameras.SAFETY.width, 3), dtype=np.uint8)
        self._predict(blank)

    def _predict(self, rgb: np.ndarray) -> tuple[list, list]:
        # Ultralytics takes pictures in OpenCV's channel order (blue, green, red), as its training read them.
        res = self._model.predict(
            np.ascontiguousarray(rgb[..., ::-1]),
            imgsz=self.imgsz,
            conf=self.threshold,
            device=self.device,
            verbose=False,
        )[0]
        return res.boxes.xyxy.tolist(), res.boxes.conf.tolist()

    def detect(self, frame: Frame) -> Detection:
        if frame.image is None:
            raise ValueError("the trained detector needs a picture")
        boxes, confs = self._predict(frame.image)
        if not boxes:
            return Detection(CLEAR, 1.0, self.version)
        found = []
        for box, conf in zip(boxes, confs, strict=True):
            if frame.depth is not None:
                reach = box_zone.depth_reach(self.cam_model, self.cam_data, box, frame.depth)
            else:
                reach = box_zone.box_reach(self.cam_model, self.cam_data, box)
            found.append((box_zone.zone_of_reach(reach), float(conf)))
        zone = max(z for z, _ in found)
        score = max(c for z, c in found if z == zone)
        return Detection(
            zones.ZONE_NAMES[zone],
            score,
            self.version,
            boxes=tuple(tuple(float(v) for v in b) for b in boxes),
        )


def yolo_builder(device: str = "mps"):
    """The ``build`` for ``detector.load``: approved bytes, version and settings -> a ``YoloDetector``."""

    def build(blob: bytes, version: str, settings: dict) -> YoloDetector:
        return YoloDetector(blob, version, settings, device=device)

    return build
