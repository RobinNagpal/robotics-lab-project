"""Build the whole lab scene for one seed.

The seed picks each source's liquid level and which cells of the empty rack
the receiving tubes stand in. Everything else is fixed by settings.py.

There are two arms, each with its own gripper and wrist camera. Their names
start with the arm's name: "arm1/shoulder_pan_joint", "arm2/gripper/pinch",
"arm1/wrist".
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
    ARMS,
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
ARM_ACTUATORS = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
GRIPPER_PREFIX = "gripper/"
GRIPPER_ACTUATOR = GRIPPER_PREFIX + "fingers_actuator"
# The point halfway between the fingertips, within one arm.
PINCH_SITE = GRIPPER_PREFIX + "pinch"
ARM_NAMES = tuple(a.name for a in ARMS)


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

    spec = mujoco.MjSpec()
    spec.option.timestep = TIMESTEP
    # The arm model's own integrator; it is not carried over when attaching.
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # The gripper's own settings, which are not carried over either. An
    # elliptic friction cone with a high impratio keeps a held tube from
    # creeping out of the fingers.
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    # Balls that have come to rest stop being simulated until something
    # touches them. Without this every ball in every source costs time on
    # every step, though only one source is ever being poured.
    spec.option.enableflags |= mujoco.mjtEnableBit.mjENBL_SLEEP
    # With hundreds of free balls the model has thousands of degrees of
    # freedom. The default Newton solver factors a matrix that size every
    # step and runs at a few steps a second; conjugate gradient is many times
    # faster. It needs 100 iterations: at 20 a light tube holding a tall
    # stack of stiff balls is pushed harder than it is held and flies apart.
    spec.option.solver = mujoco.mjtSolver.mjSOL_CG
    spec.option.iterations = 100
    spec.option.ls_iterations = 10
    for base in ARMS:
        turn = [np.cos(base.yaw / 2), 0, 0, np.sin(base.yaw / 2)]
        frame = spec.worldbody.add_frame(pos=list(base.pos), quat=turn)
        spec.attach(_arm_with_gripper(), prefix=f"{base.name}/", frame=frame)

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


def _arm_with_gripper() -> mujoco.MjSpec:
    """One UR5e with its Robotiq gripper and wrist camera, ready to attach to the table."""
    arm = mujoco.MjSpec.from_file(str(ARM_XML))
    # Match the gripper's own settings so attaching it does not warn.
    arm.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    arm.option.impratio = 10
    gripper = mujoco.MjSpec.from_file(str(GRIPPER_XML))
    arm.attach(gripper, prefix=GRIPPER_PREFIX, site=arm.site("attachment_site"))
    # A real UR controller cancels gravity itself; the model's position
    # actuators do not, and the arm would sag below every target.
    for body in arm.bodies:
        if body.name != "world":
            body.gravcomp = 1.0
    # Clamped to the side of the wrist, looking along the fingers (the
    # wrist's +y axis) and tilted in towards them, as a wrist camera would be.
    arm.body("wrist_3_link").add_camera(
        name="wrist", pos=[0, 0.10, -0.07], xyaxes=[1, 0, 0, 0, -0.3, 1], fovy=75.0
    )
    return arm


def arm_joint_ids(model: mujoco.MjModel, arm: str = "arm1") -> np.ndarray:
    """Where the arm's six joint angles are in ``data.qpos``."""
    return np.array([model.joint(f"{arm}/{name}").qposadr[0] for name in ARM_JOINTS])


def arm_actuator_ids(model: mujoco.MjModel, arm: str = "arm1") -> np.ndarray:
    """Where the arm's six joint targets are in ``data.ctrl``."""
    return np.array([model.actuator(f"{arm}/{name}").id for name in ARM_ACTUATORS])


def gripper_actuator_id(model: mujoco.MjModel, arm: str = "arm1") -> int:
    """Where the arm's gripper command is in ``data.ctrl``: 0 is open, 255 closed."""
    return model.actuator(f"{arm}/{GRIPPER_ACTUATOR}").id


def pinch_site(arm: str = "arm1") -> str:
    return f"{arm}/{PINCH_SITE}"


def reset(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Put both arms at home with their grippers open."""
    mujoco.mj_resetData(model, data)
    for arm in ARM_NAMES:
        data.qpos[arm_joint_ids(model, arm)] = HOME
        data.ctrl[arm_actuator_ids(model, arm)] = HOME
        data.ctrl[gripper_actuator_id(model, arm)] = 0.0
    mujoco.mj_forward(model, data)


def settle(model: mujoco.MjModel, data: mujoco.MjData, seconds: float = 1.0) -> None:
    """Let the balls fall into place and the glassware come to rest."""
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
