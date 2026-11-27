"""Build the whole centrifuge scene for one seed.

The seed picks the job: how many sample tubes there are, how full each is, and so whether they pair up by
weight, need a balance tube, or can go three in a triangle. It also picks which rack cells the tubes stand
in. Everything else is fixed by settings.py.

There is one arm. Its names start with "arm/": "arm/shoulder_pan_joint", "arm/gripper/pinch", "arm/wrist".
The centrifuge's start with "centrifuge/": "centrifuge/lid", "centrifuge/slot3".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import centrifuge, liquid
from .racks import add_rack, cell_centres, floor_height
from .scale import add_scale
from .settings import (
    ARM_NAME,
    ARM_XML,
    BLANK_CELLS,
    CAP_PLASTIC,
    CAP_STAND,
    FRONT,
    GRIPPER_XML,
    HOME,
    IMAGE_SIZE,
    LAYOUT,
    LIQUID,
    OVERHEAD,
    PAD_FRICTION,
    PAD_TORSION,
    RACKS_HIGH,
    RACKS_LOW,
    ROTOR_CAMERA,
    SAMPLE_COLOURS,
    SAMPLES,
    SCALE,
    STEEL,
    TABLE,
    TIMESTEP,
    TUBE,
    TUBE_MASS_SPREAD,
    WATER_CELLS,
    WATER_COLOUR,
    WATER_LEVEL,
    WATER_TUBES,
)
from .tubes import add_tube

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
PINCH_SITE = GRIPPER_PREFIX + "pinch"
PAD_GEOMS = ("right_pad1", "right_pad2", "left_pad1", "left_pad2")
# The gap between the finger pads (m) for a gripper command, measured in v1 with nothing between the fingers.
GRIPPER_GAP = ((0, 0.0852), (90, 0.0570), (180, 0.0251), (240, 0.0029), (255, 0.0004))
GRIP_SQUEEZE = 0.004
RACKS = ("sample_rack", "supply_rack")
SCENARIOS = ("pairs", "odd_one", "triangle", "mismatch")


@dataclass
class Sample:
    balls: int
    body_mass: float  # kg, the tube with its cap, without the liquid
    colour: tuple[float, float, float, float]

    @property
    def mass(self) -> float:
        """The whole tube as it would weigh on the balance, kg."""
        return self.body_mass + self.balls * LIQUID.ball_mass


@dataclass
class Truth:
    """What the scene was built with. For tests and grading; the arm must not read it."""

    seed: int
    scenario: str
    samples: dict[str, Sample] = field(default_factory=dict)
    blanks: dict[str, float] = field(default_factory=dict)  # name: body mass, kg
    water_balls: int = 0  # in each water tube
    cells: dict[str, tuple[str, int]] = field(default_factory=dict)  # tube: (rack, cell)
    capped: dict[str, bool] = field(default_factory=dict)

    @property
    def tubes(self) -> list[str]:
        return list(self.cells)


def build(seed: int, scenario: str | None = None, awake: bool = False) -> tuple[mujoco.MjModel, Truth]:
    """The scene for ``seed``. The job is SCENARIOS[seed % 4] unless ``scenario`` names one.

    Normally the racks, tubes and balls start asleep, already at rest, and wake when touched. With ``awake``
    they start awake and fall asleep once still: for tests that move tubes by hand (``move_tube``), since
    MuJoCo has no call to wake a body that is moved while asleep.
    """
    rng = np.random.default_rng(seed)
    truth = Truth(seed=seed, scenario=scenario or SCENARIOS[seed % len(SCENARIOS)])

    spec = mujoco.MjSpec()
    spec.option.timestep = TIMESTEP
    # The arm model's own integrator, and the gripper's friction settings (neither carries over on attach).
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    # Balls and tubes at rest stop being simulated until something touches them.
    spec.option.enableflags |= mujoco.mjtEnableBit.mjENBL_SLEEP
    # Conjugate gradient with 100 iterations, as in v1: fast with hundreds of balls, and stiff enough that a
    # light tube holding a tall stack of balls stays together.
    spec.option.solver = mujoco.mjtSolver.mjSOL_CG
    spec.option.iterations = 100
    spec.option.ls_iterations = 10
    spec.visual.global_.offwidth = IMAGE_SIZE[0]
    spec.visual.global_.offheight = IMAGE_SIZE[1]

    world = spec.worldbody
    spec.visual.headlight.ambient = [0.25, 0.25, 0.25]
    spec.visual.headlight.diffuse = [0.35, 0.35, 0.35]
    world.add_light(pos=[0.4, 0, 2.0], dir=[0, 0, -1], diffuse=[0.45, 0.45, 0.45], castshadow=False)
    world.add_light(pos=[1.2, 0.6, 1.5], dir=[-0.6, -0.3, -1], diffuse=[0.30, 0.30, 0.30])
    world.add_geom(
        name="table",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[1.2, 1.2, 0.05],
        friction=[TABLE.friction, 0.005, 0.0001],
        rgba=list(TABLE.rgba),
    )
    for cam in (OVERHEAD, FRONT, ROTOR_CAMERA, RACKS_HIGH, RACKS_LOW):
        world.add_camera(name=cam.name, pos=list(cam.pos), xyaxes=list(cam.xyaxes), fovy=cam.fovy)

    spec.attach(arm_with_gripper(), prefix=f"{ARM_NAME}/", frame=world.add_frame())
    yaw = LAYOUT.centrifuge_yaw
    frame = world.add_frame(
        pos=[LAYOUT.centrifuge[0], LAYOUT.centrifuge[1], 0], quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    )
    spec.attach(centrifuge.load(), prefix=centrifuge.PREFIX, frame=frame)
    add_scale(spec, LAYOUT.scale)
    add_cap_stand(spec)
    for rack in RACKS:
        add_rack(spec, rack, getattr(LAYOUT, rack))

    _choose_samples(truth, rng)
    names = list(truth.samples)
    sample_cells = rng.choice(8, size=len(names), replace=False)
    supply = [*WATER_TUBES, "blank0", "blank1"]
    # The water tubes are held from the side (to pour, the opening must stay free), and a side grip needs
    # room: they stand in fixed corner cells, and the blanks in the two cells not beside them, in either order.
    supply_cells = [*WATER_CELLS, *(int(c) for c in rng.permutation(BLANK_CELLS))]
    for blank in supply[len(WATER_TUBES) :]:
        truth.blanks[blank] = TUBE.body_mass + float(rng.uniform(-TUBE_MASS_SPREAD, TUBE_MASS_SPREAD))
    truth.water_balls = int(round(liquid.capacity() * WATER_LEVEL))

    start = 0
    placements = [("sample_rack", n, int(c)) for n, c in zip(names, sample_cells, strict=True)]
    placements += [("supply_rack", n, int(c)) for n, c in zip(supply, supply_cells, strict=True)]
    for rack, name, cell in placements:
        x, y = cell_centres(getattr(LAYOUT, rack))[cell]
        pos = np.array([x, y, floor_height()])
        sample = truth.samples.get(name)
        capped = sample is not None
        if sample:
            body_mass = sample.body_mass - TUBE.cap_mass
        elif name in WATER_TUBES:
            body_mass = TUBE.body_mass
        else:
            body_mass = truth.blanks[name]
        # The empty tubes become balance tubes: Step 3 caps them once filled.
        add_tube(
            spec,
            world,
            name,
            pos.tolist(),
            capped=capped,
            body_mass=body_mass,
            cap_later=name in truth.blanks,
        )
        truth.cells[name] = (rack, cell)
        truth.capped[name] = capped
        balls = sample.balls if sample else truth.water_balls if name in WATER_TUBES else 0
        if balls:
            centres = liquid.settled_liquid(TUBE, balls, rng) + pos
            liquid.add_balls(spec, centres, start, sample.colour if sample else WATER_COLOUR)
            start += len(centres)

    # Everything but the arm starts asleep and costs nothing until the arm touches it.
    free = {*RACKS, *truth.cells}
    for body in spec.bodies:
        if body.name in free or body.name.startswith(liquid.BALL_PREFIX):
            body.sleep = (
                mujoco.mjtSleepPolicy.mjSLEEP_AUTO_ALLOWED if awake else mujoco.mjtSleepPolicy.mjSLEEP_INIT
            )
    return spec.compile(), truth


SPARE_CAP = "spare_cap{}"


def spare_cap_top(i: int) -> np.ndarray:
    """Where spare cap ``i``'s top is, standing on its peg (the bench layout: the arm may use it)."""
    dx, dy = CAP_STAND.offsets[i]
    z = 2 * SCALE.base[2] + CAP_STAND.foot_height + CAP_STAND.peg_height + CAP_STAND.cap_top_above_peg
    return np.array([LAYOUT.scale[0] + dx, LAYOUT.scale[1] + dy, z])


def add_cap_stand(spec: mujoco.MjSpec) -> None:
    """The pegs on the balance's base and a spare cap on each.

    A spare cap is only drawn: it is moved with the hand that holds it (a mocap body), and nothing touches it.
    When the arm sets it on a tube, ``tubes.put_cap_on`` switches on that tube's own cap and this one is put
    away out of sight. So the cap's thread is not simulated, only where the cap goes.
    """
    from .tubes import add_meshes

    add_meshes(spec)
    stand = spec.worldbody.add_body(name="cap_stand")
    base_top = 2 * SCALE.base[2]
    s = CAP_STAND
    for i in range(len(s.offsets)):
        top = spare_cap_top(i)
        x, y = top[:2]
        stand.add_geom(
            name=f"cap_stand_foot{i}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[s.foot_radius, s.foot_height / 2, 0],
            pos=[x, y, base_top + s.foot_height / 2],
            rgba=list(STEEL.rgba),
        )
        stand.add_geom(
            name=f"cap_stand_peg{i}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[s.peg_radius, s.peg_height / 2, 0],
            pos=[x, y, base_top + s.foot_height + s.peg_height / 2],
            rgba=list(STEEL.rgba),
        )
        cap = spec.worldbody.add_body(name=SPARE_CAP.format(i), pos=top.tolist(), mocap=True)
        cap.add_geom(
            name=f"{SPARE_CAP.format(i)}_visual",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="tube50_cap",
            quat=[0, 1, 0, 0],
            rgba=list(CAP_PLASTIC.rgba),
            contype=0,
            conaffinity=0,
            group=2,
        )


def _choose_samples(truth: Truth, rng: np.random.Generator) -> None:
    """Sample tubes for the scenario, in a random order. Groups that should not pair are far apart in weight.

    - pairs: 2 or 4 samples, in pairs of nearly equal weight. No balance tube needed.
    - odd_one: 3 samples: a pair, and one with no partner. It needs a balance tube.

    No job puts more than 4 tubes in the rotor: the gripper's fingers need an empty hole beside the hole a
    tube goes into (see grasp-plan/docs/results.md), so a full rotor of 6 could not be loaded.
    - triangle: 3 samples of nearly equal weight. They balance 120 degrees apart, no balance tube needed.
    - mismatch: 2 samples far apart in weight. Each needs its own balance tube.
    """
    s = SAMPLES

    def group_counts(groups: int) -> list[int]:
        while True:
            counts = sorted(rng.integers(s.min_balls, s.max_balls + 1, size=groups).tolist())
            if all(b - a >= s.odd_gap for a, b in zip(counts, counts[1:], strict=False)):
                return counts

    match truth.scenario:
        case "pairs":
            sizes = [2] * int(rng.choice([1, 2]))
        case "odd_one":
            sizes = [2, 1]
        case "triangle":
            sizes = [3]
        case "mismatch":
            sizes = [1, 1]
        case _:
            raise ValueError(f"unknown scenario {truth.scenario!r}")
    balls = []
    for size, base in zip(sizes, group_counts(len(sizes)), strict=True):
        balls.append(base)
        balls += [
            int(np.clip(base + rng.integers(-s.pair_spread, s.pair_spread + 1), s.min_balls, s.max_balls))
            for _ in range(size - 1)
        ]
    order = rng.permutation(len(balls))
    for i, k in enumerate(order):
        body = TUBE.body_mass + TUBE.cap_mass + float(rng.uniform(-TUBE_MASS_SPREAD, TUBE_MASS_SPREAD))
        truth.samples[f"sample{i}"] = Sample(balls=balls[k], body_mass=body, colour=SAMPLE_COLOURS[i])


def add_wrist_sensors(arm: mujoco.MjSpec) -> None:
    """A force and a torque sensor between wrist and gripper: the UR5e's own wrist sensor (as in v2)."""
    mount = arm.body(GRIPPER_PREFIX + "base_mount")
    mount.add_site(name="wrist_ft", pos=[0, 0, 0], size=[0.005, 0, 0], group=4)
    for kind, name in (
        (mujoco.mjtSensor.mjSENS_FORCE, "wrist_force"),
        (mujoco.mjtSensor.mjSENS_TORQUE, "wrist_torque"),
    ):
        arm.add_sensor(name=name, type=kind, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="wrist_ft")


def arm_with_gripper() -> mujoco.MjSpec:
    """The UR5e with its Robotiq 2F-85, wrist camera and wrist force/torque sensor (v1's arm, v2's sensor)."""
    arm = mujoco.MjSpec.from_file(str(ARM_XML))
    # The arm model's own spotlight follows its wrist and washes out the overhead picture.
    arm.delete(arm.light("spotlight"))
    arm.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    arm.option.impratio = 10
    gripper = mujoco.MjSpec.from_file(str(GRIPPER_XML))
    for pad in PAD_GEOMS:
        gripper.geom(pad).friction[0] = PAD_FRICTION
        gripper.geom(pad).friction[1] = PAD_TORSION
        gripper.geom(pad).condim = 4
    arm.attach(gripper, prefix=GRIPPER_PREFIX, site=arm.site("attachment_site"))
    # A real UR controller cancels gravity itself; the model's position actuators do not.
    for body in arm.bodies:
        if body.name != "world":
            body.gravcomp = 1.0
    arm.body("wrist_3_link").add_camera(
        name="wrist", pos=[0, 0.10, -0.07], xyaxes=[1, 0, 0, 0, -0.3, 1], fovy=75.0
    )
    add_wrist_sensors(arm)
    return arm


def arm_joint_ids(model: mujoco.MjModel, arm: str = ARM_NAME) -> np.ndarray:
    """Where the arm's six joint angles are in ``data.qpos``."""
    return np.array([model.joint(f"{arm}/{name}").qposadr[0] for name in ARM_JOINTS])


def arm_actuator_ids(model: mujoco.MjModel, arm: str = ARM_NAME) -> np.ndarray:
    """Where the arm's six joint targets are in ``data.ctrl``."""
    return np.array([model.actuator(f"{arm}/{name}").id for name in ARM_ACTUATORS])


def gripper_actuator_id(model: mujoco.MjModel, arm: str = ARM_NAME) -> int:
    """Where the gripper command is in ``data.ctrl``: 0 is open, 255 closed."""
    return model.actuator(f"{arm}/{GRIPPER_ACTUATOR}").id


def grip_command(width: float, squeeze: float = GRIP_SQUEEZE) -> float:
    """The gripper command that closes the fingers to ``squeeze`` less than ``width``, m (see v1)."""
    commands, gaps = zip(*GRIPPER_GAP[::-1], strict=True)
    return float(np.interp(width - squeeze, gaps, commands))


def pinch_site(arm: str = ARM_NAME) -> str:
    return f"{arm}/{PINCH_SITE}"


def slot_site(k: int) -> str:
    return centrifuge.PREFIX + centrifuge.SLOT_SITE.format(k)


def entry_site(k: int) -> str:
    return centrifuge.PREFIX + centrifuge.ENTRY_SITE.format(k)


def reset(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """The arm at home with its gripper open, the lid open, the latch off."""
    mujoco.mj_resetData(model, data)
    data.qpos[arm_joint_ids(model)] = HOME
    data.ctrl[arm_actuator_ids(model)] = HOME
    data.ctrl[gripper_actuator_id(model)] = 0.0
    centrifuge.release_latch(model, data)
    mujoco.mj_forward(model, data)


def step(model: mujoco.MjModel, data: mujoco.MjData, n: int = 1) -> None:
    """Step the simulation, letting the centrifuge's latch catch the lid when it is pushed shut."""
    for _ in range(n):
        mujoco.mj_step(model, data)
        centrifuge.update_latch(model, data)


def move_tube(
    model: mujoco.MjModel, data: mujoco.MjData, name: str, pos: np.ndarray, rot: np.ndarray
) -> None:
    """Put a tube, and the balls in it, at ``pos`` (its tip) turned by ``rot``, at rest. For tests only.

    The tube must be awake (see ``build``'s ``awake``): a body moved while asleep is not simulated.
    """
    body = model.body(name).id
    old_pos, old_rot = data.xpos[body].copy(), data.xmat[body].reshape(3, 3).copy()
    ids = liquid.ball_body_ids(model)
    local = (data.xpos[ids] - old_pos) @ old_rot if len(ids) else np.zeros((0, 3))
    inside = [
        (b, p)
        for b, p in zip(ids, local, strict=True)
        if liquid.inner_radius(TUBE, p[2]) > 0 and np.hypot(p[0], p[1]) < TUBE.radius
    ]
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(rot, dtype=float).flatten())
    for b, p in [(body, np.zeros(3)), *inside]:
        joint = model.body_jntadr[b]
        adr, dof = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
        data.qpos[adr : adr + 3] = pos + rot @ p
        if b == body:
            data.qpos[adr + 3 : adr + 7] = quat
        data.qvel[dof : dof + 6] = 0
    mujoco.mj_forward(model, data)


def settle(model: mujoco.MjModel, data: mujoco.MjData, seconds: float = 1.0) -> None:
    """Let everything come to rest."""
    step(model, data, int(seconds / model.opt.timestep))
