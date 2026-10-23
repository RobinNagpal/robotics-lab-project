"""Build the whole lab scene for one seed.

The seed picks each source's liquid level and which cells of the empty rack
the receiving tubes stand in. Everything else is fixed by settings.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import liquid
from .glassware import add_container
from .racks import add_rack, add_stand, add_tray, slot_positions
from .settings import (
    ARM_XML,
    FRONT,
    GRIPPER_XML,
    HOME,
    IMAGE_SIZE,
    LAYOUT,
    LIQUID,
    OVERHEAD,
    RECEIVERS,
    SOURCES,
    SOURCES_IN_RACK,
    TABLE,
    TIMESTEP,
    Container,
)

ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
GRIPPER_PREFIX = "gripper/"
PINCH_SITE = GRIPPER_PREFIX + "pinch"


@dataclass
class Truth:
    """What the scene was built with. For tests and the examiner; the arm must not read it."""

    seed: int
    containers: dict[str, Container] = field(default_factory=dict)
    levels: dict[str, float] = field(default_factory=dict)  # sources only, as asked for
    balls: dict[str, int] = field(default_factory=dict)
    receiver_slots: dict[str, int] = field(default_factory=dict)


def build(seed: int) -> tuple[mujoco.MjModel, Truth]:
    rng = np.random.default_rng(seed)
    truth = Truth(seed=seed)

    spec = mujoco.MjSpec.from_file(str(ARM_XML))
    spec.option.timestep = TIMESTEP
    # The gripper's own settings; attaching it does not carry them over. An
    # elliptic friction cone with a high impratio keeps a held tube from
    # creeping out of the fingers.
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    # Balls that have come to rest stop being simulated until something
    # touches them. Without this every ball in every source costs time on
    # every step, though only one source is ever being poured.
    spec.option.enableflags |= mujoco.mjtEnableBit.mjENBL_SLEEP
    # With two thousand free balls the model has over ten thousand degrees
    # of freedom. The default Newton solver factors a matrix that size every
    # step and runs at a few steps a second; conjugate gradient is many times
    # faster. It needs 100 iterations: at 20 a light tube holding a tall
    # stack of stiff balls is pushed harder than it is held and flies apart.
    spec.option.solver = mujoco.mjtSolver.mjSOL_CG
    spec.option.iterations = 100
    spec.option.ls_iterations = 10
    gripper = mujoco.MjSpec.from_file(str(GRIPPER_XML))
    spec.attach(gripper, prefix=GRIPPER_PREFIX, site=spec.site("attachment_site"))
    # A real UR controller cancels gravity itself; the model's position
    # actuators do not, and the arm would sag below every target.
    for body in spec.bodies:
        if body.name != "world":
            body.gravcomp = 1.0

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
    for cam in (OVERHEAD, FRONT):
        world.add_camera(name=cam.name, pos=list(cam.pos), xyaxes=list(cam.xyaxes), fovy=cam.fovy)
    # Clamped to the side of the wrist, looking along the fingers (the
    # wrist's +y axis) and tilted in towards them, as a wrist camera would be.
    spec.body("wrist_3_link").add_camera(
        name="wrist", pos=[0, 0.10, -0.07], xyaxes=[1, 0, 0, 0, -0.3, 1], fovy=75.0
    )

    add_rack(spec, "source_rack", LAYOUT.source_rack)
    add_rack(spec, "empty_rack", LAYOUT.empty_rack)
    add_rack(spec, "filled_rack", LAYOUT.filled_rack)
    tray_floor = add_tray(spec, "source_tray", LAYOUT.tray, LAYOUT.tray_size)
    add_stand(spec, "pouring_stand", LAYOUT.stand)

    # Sources: tubes in alternate cells of the source rack's front row, the
    # beaker and flask side by side on the tray.
    source_cells = slot_positions(LAYOUT.source_rack)
    tray_spots = [(LAYOUT.tray[0] - 0.045, LAYOUT.tray[1]), (LAYOUT.tray[0] + 0.045, LAYOUT.tray[1])]
    placed = []
    for container in SOURCES:
        if container.name in SOURCES_IN_RACK:
            pos = list(source_cells[2 * SOURCES_IN_RACK.index(container.name)])
        else:
            x, y = tray_spots.pop(0)
            pos = [x, y, tray_floor]
        add_container(spec, world, container, pos)
        placed.append((container, np.array(pos)))

    # Receivers: in cells of the empty rack chosen by the seed, one apart so
    # the fingers fit between them.
    free_cells = rng.permutation([0, 2, 4, 5, 7, 9])
    empty_cells = slot_positions(LAYOUT.empty_rack)
    for container, cell in zip(RECEIVERS, free_cells, strict=False):
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

    # The liquid is already settled, so everything except the arm starts
    # asleep and costs nothing until the arm touches it. A touch wakes the
    # touched body and everything resting on or in it.
    free = {"source_rack", "empty_rack", "filled_rack", "source_tray", *(c.name for c in SOURCES + RECEIVERS)}
    for body in spec.bodies:
        if body.name in free or body.name.startswith(liquid.BALL_PREFIX):
            body.sleep = mujoco.mjtSleepPolicy.mjSLEEP_INIT

    for c in SOURCES + RECEIVERS:
        truth.containers[c.name] = c
    return spec.compile(), truth


def arm_joint_ids(model: mujoco.MjModel) -> np.ndarray:
    return np.array([model.joint(name).qposadr[0] for name in ARM_JOINTS])


def reset(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Put the arm at home with the gripper open."""
    mujoco.mj_resetData(model, data)
    data.qpos[arm_joint_ids(model)] = HOME
    data.ctrl[:6] = HOME
    data.ctrl[6] = 0.0
    mujoco.mj_forward(model, data)


def settle(model: mujoco.MjModel, data: mujoco.MjData, seconds: float = 1.0) -> None:
    """Let the balls fall into place and the glassware come to rest."""
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
