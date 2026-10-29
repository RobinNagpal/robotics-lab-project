"""What is on the table: Step 0's detector on the overhead camera, then sizes, liquid and roles.

Step 0 gives every rack and tube with its place and its top. Three things
are added here:

- **size** (Step 0.1): a tube's top minus the rack floor it stands on is
  its height, rounded to the nearest of 100, 150 and 200 mm;
- **liquid**: the wrist camera looks straight down into each tube, from
  25 cm above its rim (the same close-up the grasp uses). At the centre it
  sees the tube's bottom if the tube is empty, or the liquid's surface,
  always more than half-way up a source;
- **roles**: the rack whose tubes hold liquid is the source rack, the rack
  with empty tubes the empty rack, the rack with no tubes the destination.

Liquid needs a look from straight above each tube: from the overhead camera,
or from one camera over a whole rack, the tubes are seen at an angle, and a
narrow tube then shows its inside wall instead of its bottom.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from detect_classify.camera import Calibration
from detect_classify.classes import CLASSES
from detect_classify.locate import locate, own_pixels
from lab_scene.racks import RACK_FLOOR
from lab_scene.settings import OVERHEAD, RACK

from . import look

ROOT = Path(__file__).resolve().parents[2]
STEP0_WEIGHTS = ROOT / "step-0-detect-and-classify" / "runs" / "yolo11s_640_800" / "weights" / "best.pt"
# Step 0 was trained on pictures this size.
OVERHEAD_SIZE = (1280, 960)
SIZES = (0.100, 0.150, 0.200)
# Where a tube stands on the rack's floor, above the table.
FLOOR = RACK.height / 2 + RACK_FLOOR
# A tube counts as in a rack when it is this close to the rack's middle.
IN_RACK = 0.12
# A source is at least 60 % full; an empty tube shows its bottom. Halfway
# between: a tube whose inside shows something higher than this share of its
# height holds liquid.
LIQUID_RISE = 0.3


@dataclass
class Tube:
    xy: np.ndarray
    top: float  # rim height, m
    size: float  # 0.100, 0.150 or 0.200 m
    liquid: bool
    rack: str = ""  # source_rack, empty_rack or destination_rack


@dataclass
class Table:
    racks: dict[str, np.ndarray] = field(default_factory=dict)  # role -> (x, y)
    tubes: list[Tube] = field(default_factory=list)

    def in_rack(self, role: str) -> list[Tube]:
        return [t for t in self.tubes if t.rack == role]

    @property
    def tallest_rim(self) -> float:
        return max((t.top for t in self.tubes), default=RACK.height)


def overhead(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray, Calibration]:
    """The overhead camera's colour and depth pictures, at Step 0's size."""
    w, h = OVERHEAD_SIZE
    model.vis.global_.offwidth, model.vis.global_.offheight = w, h
    with mujoco.Renderer(model, h, w) as r:
        r.update_scene(data, camera=OVERHEAD.name)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        r.update_scene(data, camera=OVERHEAD.name)
        depth = r.render().copy()
    return rgb, depth, Calibration.from_model(model, data, OVERHEAD.name, OVERHEAD_SIZE)


class Perception:
    def __init__(self, weights: Path = STEP0_WEIGHTS):
        from ultralytics import YOLO

        self.detector = YOLO(str(weights))

    def table(self, model: mujoco.MjModel, data: mujoco.MjData) -> Table:
        rgb, depth, calib = overhead(model, data)
        result = self.detector.predict(rgb[..., ::-1].copy(), conf=0.25, retina_masks=True, verbose=False)[0]
        if result.masks is None:
            return Table()
        masks = result.masks.data.cpu().numpy() > 0.5
        classes = [CLASSES[c] for c in result.boxes.cls.cpu().numpy().astype(int)]
        found = list(zip(classes, masks, strict=True))
        table = Table()
        racks = []
        for (cls, _), own in zip(found, own_pixels(found), strict=True):
            where = locate(cls, own, depth, calib)
            if where is None:
                continue
            xy = np.array([where.x, where.y])
            if cls == "rack":
                racks.append(xy)
            else:
                size = min(SIZES, key=lambda s: abs(where.top - FLOOR - s))
                table.tubes.append(Tube(xy, where.top, size, liquid=False))
        camera = look.Camera(model, data)
        _mark_liquid(table, camera)
        camera.close()
        _roles(table, racks)
        return table


def _mark_liquid(table: Table, camera: look.Camera) -> None:
    """Look straight down into each tube and read the height at its centre."""
    for t in table.tubes:
        heights, calib = camera.above_rim(t.xy, t.top)
        u, v = calib.project(np.array([[t.xy[0], t.xy[1], t.top]]))[0]
        pu, pv = int(round(u)), int(round(v))
        inside = heights[pv - 1 : pv + 2, pu - 1 : pu + 2]
        t.liquid = bool(np.median(inside) - FLOOR > LIQUID_RISE * t.size)


def _roles(table: Table, racks: list[np.ndarray]) -> None:
    tubes_of = {i: [] for i in range(len(racks))}
    for t in table.tubes:
        if racks:
            i = int(np.argmin([np.linalg.norm(t.xy - r) for r in racks]))
            if np.linalg.norm(t.xy - racks[i]) < IN_RACK:
                tubes_of[i].append(t)
    roles = {}
    for i, tubes in tubes_of.items():
        if not tubes:
            roles[i] = "destination_rack"
        elif np.mean([t.liquid for t in tubes]) > 0.5:
            roles[i] = "source_rack"
        else:
            roles[i] = "empty_rack"
    for i, role in roles.items():
        table.racks[role] = racks[i]
        for t in tubes_of[i]:
            t.rack = role
