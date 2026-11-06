"""Step 0 on a table with capped tubes: v1's detector, and a cap check in place of v1's liquid check.

v1's Step 0 (the YOLO detector on the overhead camera) finds capped tubes as well as open ones: on scenes
150 and 153 it found all six, within 1.3 mm, with the right sizes (the cap's top is 4.6 mm above the rim,
well inside the rounding to 100, 150 or 200 mm). But v1 then tells a source from an empty tube by looking
straight down into it with the wrist camera: an empty tube shows its bottom, a source its liquid, both
far below the top. A capped tube shows its cap, at the top, which v1 reads as "full". So here the same
look decides first whether the tube is **capped**: at its centre it shows a surface within 5 mm of its
top. Only an open tube's liquid can be seen.

The cap holder (uncap/holder.py) is a fixed part whose place is known; the detector sees it as a small
rack, so anything it calls a rack there is left out.

Roles: the rack with no tubes is the destination, a rack of mostly capped tubes the **capped rack**, one
of mostly full open tubes the source rack.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from carry import look
from carry.perceive import FLOOR, IN_RACK, LIQUID_RISE, SIZES, Perception, Table, Tube, overhead
from detect_classify.classes import CLASSES
from detect_classify.locate import locate, own_pixels

CAPPED_WITHIN = 0.005  # m: a surface this close to the top, at the tube's centre, is a cap
FIXTURE = 0.06  # m: a "rack" found this close to a known fixture is that fixture


@dataclass
class SeenTube(Tube):
    capped: bool = False


class CapPerception(Perception):
    def __init__(self, fixtures: tuple[tuple[float, float], ...] = (), **kwargs):
        super().__init__(**kwargs)
        self.fixtures = [np.array(f) for f in fixtures]

    def table(self, model: mujoco.MjModel, data: mujoco.MjData) -> Table:
        """As v1's (carry.perceive.Perception.table), with the cap check."""
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
                if all(np.linalg.norm(xy - f) > FIXTURE for f in self.fixtures):
                    racks.append(xy)
            else:
                size = min(SIZES, key=lambda s: abs(where.top - FLOOR - s))
                table.tubes.append(SeenTube(xy, where.top, size, liquid=False))
        camera = look.Camera(model, data)
        try:
            for t in table.tubes:
                centre = _centre_height(camera, t)
                t.capped = bool(t.top - centre < CAPPED_WITHIN)
                t.liquid = bool(not t.capped and centre - FLOOR > LIQUID_RISE * t.size)
        finally:
            camera.close()
        _roles(table, racks)
        return table


def _centre_height(camera: look.Camera, t: Tube) -> float:
    """The height seen at the tube's centre, from straight above it (the grasp close-up)."""
    heights, calib = camera.above_rim(t.xy, t.top)
    u, v = calib.project(np.array([[t.xy[0], t.xy[1], t.top]]))[0]
    pu, pv = int(round(u)), int(round(v))
    return float(np.median(heights[pv - 1 : pv + 2, pu - 1 : pu + 2]))


def _roles(table: Table, racks: list[np.ndarray]) -> None:
    tubes_of: dict[int, list[SeenTube]] = {i: [] for i in range(len(racks))}
    for t in table.tubes:
        if racks:
            i = int(np.argmin([np.linalg.norm(t.xy - r) for r in racks]))
            if np.linalg.norm(t.xy - racks[i]) < IN_RACK:
                tubes_of[i].append(t)
    for i, tubes in tubes_of.items():
        if not tubes:
            role = "destination_rack"
        elif np.mean([t.capped for t in tubes]) > 0.5:
            role = "capped_rack"
        elif np.mean([t.liquid for t in tubes]) > 0.5:
            role = "source_rack"
        else:
            role = "empty_rack"
        table.racks[role] = racks[i]
        for t in tubes:
            t.rack = role
