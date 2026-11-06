"""The cap holder: where Arm B puts a cap it has taken off.

A narrow post on a stand, beside Arm B, out of the middle of the table. A post rather than a tray: Arm B
holds a cap by its side, and its finger pads (37.5 mm tall) reach about 8 mm below the cap's bottom. Set
down on a tray, the pads would hit the tray first; on a post narrower than the open fingers, they pass
either side of it. The cap rests on the post's top, open side down, where it can be gripped again the
same way to screw it back on later.

It is a fixed part of the lab, like the racks: its place is known, not looked for.
"""

from __future__ import annotations

import mujoco
import numpy as np
from lab_scene.settings import RACK, TABLE

NAME = "cap_holder"
XY = (0.25, 0.12)  # m: Arm B's side, clear of the source rack (x 0.35 to 0.55) and the middle
STAND = (0.025, 0.025, RACK.height / 2)  # half sizes, m: as tall as a rack
POST = (0.008, 0.008, 0.015)  # half sizes, m: 16 mm square (its corners 11 mm out, inside a 28.8 mm cap)
TOP = 2 * STAND[2] + 2 * POST[2]  # m: the post's top above the table, where a cap's bottom rests


def add(spec: mujoco.MjSpec) -> None:
    """Put the holder on the table (an ``edit`` for capping.scene.build)."""
    body = spec.worldbody.add_body(name=NAME, pos=[XY[0], XY[1], 0.0])
    for name, half, z in (("stand", STAND, STAND[2]), ("post", POST, 2 * STAND[2] + POST[2])):
        body.add_geom(
            name=f"{NAME}_{name}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=list(half),
            pos=[0, 0, z],
            friction=[TABLE.friction, 0.005, 0.0001],
            rgba=[0.55, 0.55, 0.6, 1.0],
        )


def top() -> np.ndarray:
    """Where the middle of a cap's bottom goes, set on the post."""
    return np.array([XY[0], XY[1], TOP])
