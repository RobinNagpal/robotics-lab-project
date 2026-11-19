"""What the arm knows before it looks, and the cameras' geometry.

Known (set up once, like a real robot work cell): where the two racks stand and so where their 16 cells are,
and where each camera is and how wide its lens is. Unknown (the job): which cells hold a tube, and what each
tube is. That is what Step 0 finds.

A camera turns a point in the world into a pixel; ``ray`` goes the other way: a pixel becomes the line of
points it could have come from. The depth picture says how far along that line the nearest surface is.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass

import mujoco
import numpy as np
from centrifuge_scene import scene
from centrifuge_scene.racks import cell_centres, floor_height
from centrifuge_scene.settings import IMAGE_SIZE, LAYOUT, TUBE
from PIL import Image

RACKS = ("sample_rack", "supply_rack")
# A tube in a cell stands from the rack's floor up to this (the top of a cap).
TUBE_TOP = floor_height() + TUBE.capped_height


@dataclass(frozen=True)
class Cell:
    rack: str
    index: int
    xy: tuple[float, float]

    @property
    def key(self) -> str:
        return f"{self.rack}/{self.index}"


CELLS = [Cell(rack, i, xy) for rack in RACKS for i, xy in enumerate(cell_centres(getattr(LAYOUT, rack)))]


@dataclass(frozen=True)
class Camera:
    """A camera's place and lens: everything needed to go between pixels and the world."""

    name: str
    pos: np.ndarray  # metres
    rot: np.ndarray  # columns: right, up, backwards (it looks along -z)
    fovy: float  # degrees, top to bottom of the picture
    width: int
    height: int

    @property
    def focal(self) -> float:
        """Pixels per unit of sideways offset at unit distance."""
        return (self.height / 2) / math.tan(math.radians(self.fovy) / 2)

    def project(self, p: np.ndarray) -> tuple[float, float, float]:
        """The pixel (u across, v down) a world point lands on, and its depth (m along the view)."""
        c = self.rot.T @ (np.asarray(p) - self.pos)
        depth = -c[2]
        return self.width / 2 + self.focal * c[0] / depth, self.height / 2 - self.focal * c[1] / depth, depth

    def ray(self, u: float, v: float) -> np.ndarray:
        """The direction (unit length, world frame) of the line through pixel (u, v)."""
        d = self.rot @ np.array(
            [(u - self.width / 2) / self.focal, -(v - self.height / 2) / self.focal, -1.0]
        )
        return d / np.linalg.norm(d)

    def unproject(self, u: float, v: float, depth: float) -> np.ndarray:
        """The world point at pixel (u, v) and ``depth`` (as the depth picture gives it: along the view)."""
        c = np.array(
            [(u - self.width / 2) / self.focal * depth, -(v - self.height / 2) / self.focal * depth, -depth]
        )
        return self.pos + self.rot @ c


@dataclass
class View:
    camera: Camera
    png: bytes  # the colour picture, as sent to the model
    depth: np.ndarray  # metres along the view, one per pixel


def cameras(model: mujoco.MjModel, data: mujoco.MjData, names, size=IMAGE_SIZE) -> dict[str, Camera]:
    out = {}
    for n in names:
        c = model.camera(n).id
        out[n] = Camera(
            n, data.cam_xpos[c].copy(), data.cam_xmat[c].reshape(3, 3).copy(), float(model.cam_fovy[c]), *size
        )
    return out


def take_pictures(seed: int, names, settle: float = 0.3, arrange=None) -> tuple[dict[str, View], object]:
    """Colour and depth pictures from ``names`` for one seed's scene, and its truth (for grading only).

    ``arrange(model, data, truth)``, if given, moves things before the scene settles (for example, knocks
    tubes over); the scene is then built awake, so moved tubes are simulated.
    """
    model, truth = scene.build(seed, awake=arrange is not None)
    data = mujoco.MjData(model)
    scene.reset(model, data)
    if arrange is not None:
        arrange(model, data, truth)
    scene.settle(model, data, settle)
    cams = cameras(model, data, names)
    views = {}
    with mujoco.Renderer(model, height=IMAGE_SIZE[1], width=IMAGE_SIZE[0]) as r:
        for n in names:
            r.update_scene(data, camera=n)
            buf = io.BytesIO()
            Image.fromarray(r.render()).save(buf, format="PNG")
            r.enable_depth_rendering()
            r.update_scene(data, camera=n)
            depth = r.render().copy()
            r.disable_depth_rendering()
            views[n] = View(cams[n], buf.getvalue(), depth)
    return views, (model, data, truth)
