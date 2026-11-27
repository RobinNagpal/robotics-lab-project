"""The camera the safety feature watches through.

v3's ``overhead`` camera (1.25 m above the bench, 55° view) was made to see the racks and the centrifuge. A
hand comes in from the bench's edges, 10 to 45 cm above the table, where that view is narrowest. The safety
camera hangs higher (1.9 m above the bench, about 2.8 m above the floor: a lab ceiling) with a wider lens,
centred over the zones, so it sees the whole warning zone even where a hand is high up at the bench's sides.

It is an RGB-D camera: a colour picture and a depth picture (MuJoCo renders both). The depth is what tells how
high a hand is (docs/detector.md). ``docs/camera-check.md`` compares the two.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco


@dataclass(frozen=True)
class SafetyCamera:
    name: str
    pos: tuple[float, float, float]
    # Straight down; the picture's top is +x (away from the arm), its right is -y, as v3's overhead camera.
    xyaxes: tuple[float, ...]
    fovy: float
    width: int
    height: int


SAFETY = SafetyCamera(
    name="safety",
    pos=(0.50, 0.0, 1.90),
    xyaxes=(0.0, -1.0, 0.0, 1.0, 0.0, 0.0),
    fovy=75.0,
    width=800,
    height=600,
)


def add_safety_camera(spec: mujoco.MjSpec) -> None:
    """On a body of its own: cameras on the world body itself are numbered first, which would shift v3's."""
    c = SAFETY
    mount = spec.worldbody.add_body(name="safety_camera_mount")
    mount.add_camera(name=c.name, pos=list(c.pos), xyaxes=list(c.xyaxes), fovy=c.fovy)


def to_pixels(model: mujoco.MjModel, data: mujoco.MjData, camera: str, points, width: int, height: int):
    """Where world points (n, 3) land in a camera's picture: (n, 2) pixel x, y, and whether each is in
    front.
    """
    import numpy as np

    cid = model.camera(camera).id
    pos, rot = data.cam_xpos[cid], data.cam_xmat[cid].reshape(3, 3)
    local = (np.asarray(points) - pos) @ rot  # camera frame: x right, y up, looking along -z
    f = 0.5 * height / np.tan(np.radians(model.cam_fovy[cid]) / 2)
    depth = -local[:, 2]
    px = width / 2 + f * local[:, 0] / np.maximum(depth, 1e-6)
    py = height / 2 - f * local[:, 1] / np.maximum(depth, 1e-6)
    return np.stack([px, py], axis=1), depth > 0


def pixel_rays(model: mujoco.MjModel, data: mujoco.MjData, camera: str, px, width: int, height: int):
    """The camera's position, and the world direction (n, 3) of the ray through each pixel (n, 2)."""
    import numpy as np

    cid = model.camera(camera).id
    pos, rot = data.cam_xpos[cid], data.cam_xmat[cid].reshape(3, 3)
    f = 0.5 * height / np.tan(np.radians(model.cam_fovy[cid]) / 2)
    px = np.asarray(px, dtype=float)
    local = np.stack([(px[:, 0] - width / 2) / f, -(px[:, 1] - height / 2) / f, -np.ones(len(px))], axis=1)
    return pos.copy(), local @ rot.T


def camera_only():
    """A model holding just the safety camera (and its data, posed): for working out where a picture's pixels
    point, without the whole scene.
    """
    spec = mujoco.MjSpec()
    add_safety_camera(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data
