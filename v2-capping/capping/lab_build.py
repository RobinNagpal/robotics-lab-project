"""v1's lab scene builder, copied here so that v2 can change the scene before it is compiled.

v1 (``lab_scene.scene.build``) is finished and is not changed: it compiles its scene inside ``build`` and
gives no way in before that. This is the same function, line for line, with two additions:

- ``edit``: called with the finished MjSpec before it is compiled (v2 puts the caps on with it);
- ``receiver_cells``: which cells of the empty rack the receivers stand in. ``None`` keeps v1's choice
  (the same random draw, so the scene is v1's exactly; tests/test_lab_build.py checks this).

Everything else (the arms, racks, sources, liquid, settings) is v1's own code, imported read only.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import mujoco
import numpy as np
from lab_scene import liquid
from lab_scene.glassware import add_container
from lab_scene.racks import add_rack, slot_positions
from lab_scene.scene import RACKS, Truth, _spaced_cells, arm_with_gripper
from lab_scene.settings import (
    ARMS,
    FRONT,
    IMAGE_SIZE,
    LAYOUT,
    LIQUID,
    OVERHEAD,
    POUR_CAMERA,
    RECEIVERS,
    SOURCES,
    TABLE,
    TIMESTEP,
)


def build(
    seed: int,
    edit: Callable[[mujoco.MjSpec], None] | None = None,
    receiver_cells: Callable[[np.random.Generator], Sequence[int]] | None = None,
) -> tuple[mujoco.MjModel, Truth]:
    """v1's scene for ``seed`` (see the module's docstring for ``edit`` and ``receiver_cells``).

    ``receiver_cells`` is given the seed's random generator at the point where v1 draws the receivers'
    cells, and returns the cells (one per receiver). The draw for the sources and the liquid is v1's."""
    rng = np.random.default_rng(seed)
    truth = Truth(seed=seed)

    spec = mujoco.MjSpec()
    spec.option.timestep = TIMESTEP
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.option.enableflags |= mujoco.mjtEnableBit.mjENBL_SLEEP
    spec.option.solver = mujoco.mjtSolver.mjSOL_CG
    spec.option.iterations = 100
    spec.option.ls_iterations = 10
    for base in ARMS:
        turn = [np.cos(base.yaw / 2), 0, 0, np.sin(base.yaw / 2)]
        frame = spec.worldbody.add_frame(pos=list(base.pos), quat=turn)
        spec.attach(arm_with_gripper(), prefix=f"{base.name}/", frame=frame)

    spec.visual.global_.offwidth = IMAGE_SIZE[0]
    spec.visual.global_.offheight = IMAGE_SIZE[1]
    world = spec.worldbody
    world.add_light(pos=[0.4, 0, 2.0], dir=[0, 0, -1], diffuse=[0.5, 0.5, 0.5], castshadow=False)
    world.add_light(pos=[1.2, 0.6, 1.5], dir=[-0.6, -0.3, -1], diffuse=[0.35, 0.35, 0.35])
    world.add_geom(
        name="table",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[1.2, 1.2, 0.05],
        friction=[TABLE.friction, 0.005, 0.0001],
        rgba=list(TABLE.rgba),
    )
    for cam in (OVERHEAD, FRONT, POUR_CAMERA):
        world.add_camera(name=cam.name, pos=list(cam.pos), xyaxes=list(cam.xyaxes), fovy=cam.fovy)
    for rack in RACKS:
        add_rack(spec, rack, getattr(LAYOUT, rack))

    placed = []
    source_cells = slot_positions(LAYOUT.source_rack)
    for container, cell in zip(SOURCES, _spaced_cells(rng), strict=False):
        pos = source_cells[cell]
        add_container(spec, world, container, list(pos))
        placed.append((container, np.array(pos)))
        truth.source_slots[container.name] = int(cell)
    empty_cells = slot_positions(LAYOUT.empty_rack)
    chosen = _spaced_cells(rng) if receiver_cells is None else receiver_cells(rng)
    for container, cell in zip(RECEIVERS, chosen, strict=False):
        add_container(spec, world, container, list(empty_cells[cell]))
        truth.receiver_slots[container.name] = int(cell)

    start = 0
    for container, pos in placed:
        level = float(rng.uniform(*LIQUID.level))
        centres = liquid.settled_liquid(container, level, rng) + pos
        liquid.add_balls(spec, centres, start)
        start += len(centres)
        truth.levels[container.name] = level
        truth.balls[container.name] = len(centres)

    free = {*RACKS, *(c.name for c in SOURCES + RECEIVERS)}
    for body in spec.bodies:
        if body.name in free or body.name.startswith(liquid.BALL_PREFIX):
            body.sleep = mujoco.mjtSleepPolicy.mjSLEEP_INIT

    for c in SOURCES + RECEIVERS:
        truth.containers[c.name] = c
    if edit is not None:
        edit(spec)
    return spec.compile(), truth
