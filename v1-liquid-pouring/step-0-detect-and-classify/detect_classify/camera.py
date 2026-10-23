"""The overhead camera's lens and mount, and turning pixels plus depth into points.

A real lab calibrates its cameras once, so the arm may know these. MuJoCo's
camera looks along its own -z axis, with x to the right of the picture and y
up it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class Calibration:
    width: int
    height: int
    fovy: float  # degrees
    pos: np.ndarray  # camera position in the world, (3,)
    rot: np.ndarray  # camera axes as columns, in the world, (3, 3)

    @property
    def focal(self) -> float:
        """Focal length in pixels; the same across and down, as MuJoCo's pixels are square."""
        return (self.height / 2) / math.tan(math.radians(self.fovy) / 2)

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "fovy": self.fovy,
            "pos": self.pos.tolist(),
            "rot": self.rot.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Calibration:
        return cls(d["width"], d["height"], d["fovy"], np.array(d["pos"]), np.array(d["rot"]))

    @classmethod
    def from_model(cls, model: mujoco.MjModel, data: mujoco.MjData, name: str, size: tuple[int, int]):
        cam = model.camera(name).id
        return cls(
            size[0],
            size[1],
            float(model.cam_fovy[cam]),
            data.cam_xpos[cam].copy(),
            data.cam_xmat[cam].reshape(3, 3).copy(),
        )

    def deproject(self, u: np.ndarray, v: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """World points (N, 3) for pixel columns u, rows v and their depths along the view axis."""
        f = self.focal
        x = (u + 0.5 - self.width / 2) * depth / f
        y = -(v + 0.5 - self.height / 2) * depth / f
        local = np.stack([x, y, -depth], axis=1)
        return self.pos + local @ self.rot.T

    def project(self, points: np.ndarray) -> np.ndarray:
        """Pixel (u, v) for world points (N, 3). The inverse of deproject."""
        local = (points - self.pos) @ self.rot
        depth = -local[:, 2]
        f = self.focal
        u = local[:, 0] * f / depth + self.width / 2 - 0.5
        v = -local[:, 1] * f / depth + self.height / 2 - 0.5
        return np.stack([u, v], axis=1)
