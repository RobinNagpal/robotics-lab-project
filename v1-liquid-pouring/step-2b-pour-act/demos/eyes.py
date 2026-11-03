"""What the learned pour sees: three pictures from two cameras, and Arm B's joints.

The same code makes the pictures when the programmed pour is recorded and when the learned pour runs,
so the policy is shown exactly what it was trained on.

    strip   the pour camera's picture cut to the receiver (Step 2's 96 x 384 strip), shrunk by half to
            48 x 192: 1.25 mm per pixel, the level network's own input. It reaches 30 mm above the rim,
            so it also shows the lip and the balls falling from it.
    wide    the pour camera's whole picture shrunk five times, to 128 x 96 (about 3 mm per pixel): both
            tubes, both grippers, how far the source is tipped.
    wrist   Arm B's wrist camera, 96 x 72: the source in the fingers, seen from another side.

Where the strip is cut from is found once, from the receiver's rim at the start (as in Step 2).
"""

from __future__ import annotations

import mujoco
import numpy as np
from lab_scene import scene as lab
from pour.pour import Camera
from pour.tilt import ARM

WIDE_SHRINK = 5  # 640 x 480 -> 128 x 96
STRIP_SHRINK = 2  # 96 x 384 -> 48 x 192
WRIST_SIZE = (96, 72)  # width, height
WRIST_SHRINK = 2  # rendered at twice the size, then averaged, so thin edges do not flicker


def shrink(image: np.ndarray, k: int) -> np.ndarray:
    """Average k x k blocks of pixels (the picture's sides must be multiples of k)."""
    h, w, c = image.shape
    x = image.reshape(h // k, k, w // k, k, c).astype(np.float32).mean(axis=(1, 3))
    return np.round(x).astype(np.uint8)


class Eyes:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, receiver: str, height: float):
        self.pour = Camera(model, data, receiver, height)
        w, h = WRIST_SIZE
        self.wrist = mujoco.Renderer(model, h * WRIST_SHRINK, w * WRIST_SHRINK)
        self.wrist_name = f"{ARM}/wrist"
        self.joints = lab.arm_joint_ids(model, ARM)
        self.actuators = lab.arm_actuator_ids(model, ARM)

    def look(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        full = self.pour.full(data)
        self.wrist.update_scene(data, camera=self.wrist_name)
        return {
            "full": full,
            "strip": shrink(self.pour.crop(full), STRIP_SHRINK),
            "wide": shrink(full, WIDE_SHRINK),
            "wrist": shrink(self.wrist.render(), WRIST_SHRINK),
            "qpos": data.qpos[self.joints].copy(),
            "ctrl": data.ctrl[self.actuators].copy(),
        }

    def close(self) -> None:
        self.pour.renderer.close()
        self.wrist.close()
