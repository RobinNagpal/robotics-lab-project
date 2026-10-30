"""The wrist camera looking straight down, as in Step 1: a depth picture turned into heights.

As in Step 1 (step-1-grasp/docs/top-down/approach.md), the camera is placed on the
gripper's axis, looking straight down, with the table's x to the right of the
picture and its y up it. The picture is rendered with the arms where they
are (at home while looking), so the arm does not block its own view.
"""

from __future__ import annotations

import mujoco
import numpy as np
from detect_classify.camera import Calibration
from grasp import scenes

FOVY = scenes.LOOK_FOVY


class Camera:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, size: int = scenes.LOOK_SIZE):
        self.model, self.data, self.size = model, data, size
        model.vis.global_.fovy = FOVY
        # One eye, at the camera's own place (no stereo offset).
        model.vis.global_.ipd = 0.0
        self.renderer = mujoco.Renderer(model, size, size)
        self.renderer.enable_depth_rendering()
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        # Looking straight down, with the picture's up along the table's +y.
        self.cam.elevation, self.cam.azimuth = -90.0, 90.0

    def heights(self, xy: np.ndarray, height: float) -> tuple[np.ndarray, Calibration]:
        """Heights above the table (m) seen from ``height`` above the point ``xy`` on the table."""
        self.cam.lookat[:] = [xy[0], xy[1], 0.0]
        self.cam.distance = height
        self.renderer.update_scene(self.data, camera=self.cam)
        depth = self.renderer.render().copy()
        gl = self.renderer.scene.camera[0]
        forward, up = np.array(gl.forward), np.array(gl.up)
        z = -forward
        x = np.cross(up, z)
        pos = np.array(self.cam.lookat) + height * np.array([0, 0, 1.0])
        calib = Calibration(self.size, self.size, FOVY, pos, np.stack([x, up, z], axis=1))
        return np.clip(pos[2] - depth, 0.0, None).astype(np.float32), calib

    def above_rim(self, xy: np.ndarray, top: float) -> tuple[np.ndarray, Calibration]:
        """The grasp picture: from ``scenes.LOOK_HEIGHT`` above a tube's rim, as in Step 1."""
        heights, calib = self.heights(xy, top + scenes.LOOK_HEIGHT)
        return heights, calib

    def close(self) -> None:
        self.renderer.close()
