"""The v1 lab scene with a screw cap on each of the three receivers.

The scene is v1's, built by lab_build.py (v1's builder, copied so the caps can be added before the scene
is compiled; v1 itself is not changed). A capped scene is stepped with ``caps.step`` and reset with
``reset`` here (see caps.py for why).

    from capping import caps, scene
    model, truth = scene.build(150)
    data = mujoco.MjData(model)
    scene.reset(model, data)
    caps.step(model, data, 1000)
"""

from __future__ import annotations

import mujoco
import numpy as np
from lab_scene import scene as lab
from lab_scene.settings import RECEIVERS

from . import caps, lab_build

CAPPED = tuple(c.name for c in RECEIVERS)  # the tubes with a cap on: every receiver

# The capped rack's cells, spread out. v1 stands its three receivers in cells of (0, 3, 6, 9) or
# (1, 4, 5, 8) (cells 0 to 4 are the first row, 5 to 9 the second, 36 mm apart), which always puts two
# of them diagonally next to each other, 51 mm apart, centre to centre. That suits open tubes; a cap
# makes a tube's top 28.8 mm wide instead of 25, and Arm A's closing fingers came within the margin of a
# neighbour's cap (4 of 12 uncapping runs could not pick a tube). Here the three stand at the two ends
# of one row and the middle of the other: 144 mm and 80 mm apart.
SPREAD_CELLS = ((0, 4, 7), (5, 9, 2))


def spread_cells(rng: np.random.Generator) -> list[int]:
    """The receivers' cells, spread out (SPREAD_CELLS). v1's own draw is made first and mapped onto
    them, so the random numbers used after it (the sources' liquid) are v1's, and the seed still picks
    which tube stands where."""
    v1 = [int(c) for c in lab._spaced_cells(rng)][: len(CAPPED)]
    k = 0 if v1[0] in lab.SPACED_CELLS[0] else 1
    places = [lab.SPACED_CELLS[k].index(c) for c in v1]
    rank = np.argsort(np.argsort(places))
    return [SPREAD_CELLS[k][r] for r in rank]


def build(seed: int, edit=None, spread: bool = False, placed=None) -> tuple[mujoco.MjModel, lab.Truth]:
    """v1's scene for ``seed`` with a cap on every receiver. ``edit``, if given, is called with the spec
    after the caps are on, for trials that add more (cap_trial.py holds the tube with it). ``spread``
    stands the capped tubes further apart (SPREAD_CELLS) instead of in v1's cells. ``placed`` maps a
    capped tube's name to (dx, dy, turn): moved that far (m) from the middle of its cell and turned about
    its axis (rad), as a tube put in a rack by hand stands (for the stress test)."""

    def add_caps(spec: mujoco.MjSpec) -> None:
        for container in RECEIVERS:
            tube = spec.body(container.name)
            if placed and container.name in placed:
                dx, dy, turn = placed[container.name]
                tube.pos = [tube.pos[0] + dx, tube.pos[1] + dy, tube.pos[2]]
                tube.quat = [np.cos(turn / 2), 0.0, 0.0, np.sin(turn / 2)]
            caps.add_cap(spec, tube, container, list(tube.pos))
            # A capped tube and its cap never sleep: asleep, the thread froze while the cap was being
            # turned. They are empty and light, so staying awake costs little.
            for name in (container.name, caps.cap_name(container.name)):
                spec.body(name).sleep = mujoco.mjtSleepPolicy.mjSLEEP_NEVER
        # Nor does the rack they stand in, which cannot sleep while they touch it.
        spec.body("empty_rack").sleep = mujoco.mjtSleepPolicy.mjSLEEP_NEVER
        if edit is not None:
            edit(spec)

    return lab_build.build(seed, edit=add_caps, receiver_cells=spread_cells if spread else None)


def reset(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """v1's reset (both arms home, grippers open), and every cap back on its thread."""
    lab.reset(model, data)
    caps.reset(model, data)
