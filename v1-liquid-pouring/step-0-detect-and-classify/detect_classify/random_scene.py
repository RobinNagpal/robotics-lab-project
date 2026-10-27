"""A lab scene where the seed picks everything the detector must not memorise.

The simulation's own scene (lab_scene.scene) keeps the racks in fixed places,
so a detector trained on it would learn where things usually are, not what
they look like. Here the seed also picks how many tubes there are and of which
heights, which cells they stand in, where each rack stands and how it is
turned, the colours, the lights, the camera's exact mount and the arms' poses.

The building blocks (arms, racks, test tubes, liquid) are the simulation's
own. The scene is only rendered, never stepped: the liquid is settled before
it is placed, so one mj_forward gives the picture.
"""

from __future__ import annotations

import colorsys
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import mujoco
import numpy as np
from lab_scene import liquid
from lab_scene.glassware import add_container
from lab_scene.racks import RACK_FLOOR, add_rack
from lab_scene.scene import ARM_JOINTS, RACKS, arm_with_gripper
from lab_scene.settings import ARMS, HOME, LIQUID, OVERHEAD, RACK, RECEIVERS, SOURCES, TABLE, Container

from .classes import CONTAINER_CLASS

IMAGE_SIZE = (1280, 960)

BY_NAME = {c.name: c for c in SOURCES + RECEIVERS}
# Every tube is a plain glass test tube, 100, 150 or 200 mm tall. Sources and
# receivers are the same tubes; only the liquid tells them apart.
TUBES = tuple(c.name for c in RECEIVERS)
# How many tubes stand in each rack: (fewest, most). The destination rack
# sometimes already holds tubes filled earlier.
TUBES_PER_RACK = {"source_rack": (1, 4), "empty_rack": (1, 5), "destination_rack": (0, 2)}
# The role of a tube in each rack.
TUBE_ROLE = {"source_rack": "source", "empty_rack": "receiver", "destination_rack": "filled"}
# A finished tube holds about half its height.
FILLED_LEVEL = (0.40, 0.60)

# Half the diagonal of a rack's footprint, plus a little room, for keeping racks apart.
RACK_REACH = math.hypot(0.1025, 0.048) + 0.01
# Where racks may stand: within both arms' reach of their bases, and inside
# what the overhead camera sees of the table (x -0.16..0.96, y -0.75..0.75).
REACH = (0.30, 0.80)
VIEW_X = (-0.08, 0.90)
VIEW_Y = 0.66


@dataclass
class Instance:
    """One object in the scene, for labels and grading. The arms must not read it."""

    name: str  # MuJoCo body name
    cls: str
    role: str  # source_rack, empty_rack, destination_rack, source, receiver, filled
    top: float  # height of its highest point above the table, m
    level: float = 0.0  # liquid level as a fraction of the inside height
    balls: tuple[int, int] = (0, 0)  # first ball number and how many, for the liquid inside it
    kind: str = ""  # the container's name in lab_scene.settings, e.g. receiver_150; containers only
    rack: str = ""  # the rack a tube stands in; tubes only
    cell: int = -1  # the rack cell it stands in, 0 to 9; tubes only


@dataclass
class Scene:
    seed: int
    instances: list[Instance] = field(default_factory=list)


def _yaw_quat(yaw: float) -> list[float]:
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def _rotate(xy: tuple[float, float], yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * xy[0] - s * xy[1], s * xy[0] + c * xy[1]])


class _Placer:
    """Picks spots on the table, reachable by both arms, that keep every footprint apart."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.taken: list[tuple[np.ndarray, float]] = []

    def spot(self, reach: float) -> np.ndarray:
        for _ in range(2000):
            p = np.array(
                [self.rng.uniform(VIEW_X[0] + reach, VIEW_X[1] - reach), self.rng.uniform(-1, 1) * VIEW_Y]
            )
            if abs(p[1]) > VIEW_Y - reach:
                continue
            if not all(REACH[0] < np.linalg.norm(p - base.pos[:2]) < REACH[1] for base in ARMS):
                continue
            if all(np.linalg.norm(p - q) > reach + rq for q, rq in self.taken):
                self.taken.append((p, reach))
                return p
        raise RuntimeError("no free spot on the table")


def _random_rgb(rng: np.random.Generator, sat: tuple[float, float], val: tuple[float, float]) -> list[float]:
    return list(colorsys.hsv_to_rgb(rng.uniform(), rng.uniform(*sat), rng.uniform(*val)))


# The wrist camera as lab_scene.scene mounts it: clamped to the side of the
# wrist, looking along the fingers and tilted in towards them.
WRIST_CAMERA = {"name": "wrist", "pos": [0, 0.10, -0.07], "xyaxes": [1, 0, 0, 0, -0.3, 1], "fovy": 75.0}


def task_level(rng: np.random.Generator, role: str) -> float:
    """The task's own liquid: sources between 20 % and 90 %, filled tubes about half, receivers empty."""
    if role == "source":
        return float(rng.uniform(*LIQUID.level))
    if role == "filled":
        return float(rng.uniform(*FILLED_LEVEL))
    return 0.0


def build(
    seed: int,
    fill: Callable[[np.random.Generator, str], float] = task_level,
    hand: Callable[[mujoco.MjSpec], None] | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData, Scene]:
    """Build, colour and pose the scene for one seed. Returns it ready to render.

    ``fill`` gives each container's liquid level from its role (source,
    receiver or filled); 0 leaves it empty. ``hand``, if given, replaces the
    two arms: the scene starts without them and ``hand`` adds what should
    stand in for them, such as a gripper on its own.
    """
    rng = np.random.default_rng(seed)
    scene = Scene(seed=seed)
    place = _Placer(rng)

    spec = mujoco.MjSpec()
    # The arm's and the gripper's own settings, as in lab_scene.scene; without
    # them attaching warns.
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    if hand is None:
        # Both arms, where the simulation bolts them down, with their wrist cameras.
        for base in ARMS:
            turn = [math.cos(base.yaw / 2), 0, 0, math.sin(base.yaw / 2)]
            frame = spec.worldbody.add_frame(pos=list(base.pos), quat=turn)
            spec.attach(arm_with_gripper(), prefix=f"{base.name}/", frame=frame)
    else:
        hand(spec)
    spec.visual.global_.offwidth = IMAGE_SIZE[0]
    spec.visual.global_.offheight = IMAGE_SIZE[1]
    world = spec.worldbody
    world.add_geom(
        name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1.2, 1.2, 0.05], rgba=list(TABLE.rgba)
    )
    world.add_camera(
        name=OVERHEAD.name, pos=list(OVERHEAD.pos), xyaxes=list(OVERHEAD.xyaxes), fovy=OVERHEAD.fovy
    )
    # One light roughly overhead, one from a random side.
    world.add_light(
        pos=[0.4 + rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), 2.0],
        dir=[0, 0, -1],
        diffuse=[rng.uniform(0.25, 0.55)] * 3,
        castshadow=bool(rng.random() < 0.5),
    )
    side = rng.uniform(0, 2 * math.pi)
    world.add_light(
        pos=[0.4 + 1.2 * math.cos(side), 1.2 * math.sin(side), rng.uniform(1.0, 2.0)],
        dir=[-math.cos(side), -math.sin(side), -1.0],
        diffuse=[rng.uniform(0.1, 0.5)] * 3,
        castshadow=bool(rng.random() < 0.5),
    )

    # instance name, container, bottom centre, role, rack, cell
    containers: list[tuple[str, Container, np.ndarray, str, str, int]] = []

    # Three racks, each turned any way. Which rack plays which role is only
    # known from what stands in it.
    for rack in RACKS:
        xy, yaw = place.spot(RACK_REACH), rng.uniform(0, 2 * math.pi)
        add_rack(spec, rack, (float(xy[0]), float(xy[1]))).quat = _yaw_quat(yaw)
        scene.instances.append(Instance(rack, "rack", rack, RACK.height))
        fewest, most = TUBES_PER_RACK[rack]
        count = rng.integers(fewest, most + 1)
        for cell in rng.choice(RACK.rows * RACK.columns, size=count, replace=False):
            row, col = divmod(int(cell), RACK.columns)
            local = (RACK.pitch * (col - (RACK.columns - 1) / 2), RACK.pitch * (row - (RACK.rows - 1) / 2))
            bottom = np.array([*(xy + _rotate(local, yaw)), RACK.height / 2 + RACK_FLOOR])
            role = TUBE_ROLE[rack]
            name = f"{role}_{len(containers)}"
            containers.append((name, BY_NAME[str(rng.choice(TUBES))], bottom, role, rack, int(cell)))

    start = 0
    for name, c, bottom, role, rack, cell in containers:
        add_container(spec, world, c, bottom.tolist(), name=name)
        level, balls = fill(rng, role), (start, 0)
        if level > 0:
            centres = liquid.settled_liquid(c, level, rng) + bottom
            liquid.add_balls(spec, centres, start)
            balls = (start, len(centres))
            start += len(centres)
        top = float(bottom[2] + c.height)
        scene.instances.append(
            Instance(name, CONTAINER_CLASS[c.name], role, top, level, balls, c.name, rack, cell)
        )

    model = spec.compile()
    _recolour(model, rng)
    _jitter_camera(model, rng)
    data = mujoco.MjData(model)
    # Each arm near home, as it would be between moves.
    if hand is None:
        for base in ARMS:
            for joint, q in zip(ARM_JOINTS, HOME, strict=True):
                data.qpos[model.joint(f"{base.name}/{joint}").qposadr[0]] = q + rng.uniform(-0.15, 0.15)
    mujoco.mj_forward(model, data)
    return model, data, scene


def _recolour(model: mujoco.MjModel, rng: np.random.Generator) -> None:
    """New colours for the table, plastic, glass and liquid, so the detector learns shapes."""
    plastic = {model.body(n).id for n in RACKS}
    rack_rgb = _random_rgb(rng, (0.0, 0.9), (0.3, 0.95))
    glass_a = rng.uniform(0.2, 0.45)
    liquid_rgb = _random_rgb(rng, (0.3, 1.0), (0.3, 1.0))
    for g in range(model.ngeom):
        body = model.body(model.geom_bodyid[g]).name
        if model.geom(g).name == "table":
            model.geom_rgba[g, :3] = _random_rgb(rng, (0.0, 0.3), (0.45, 0.95))
        elif model.geom_bodyid[g] in plastic:
            model.geom_rgba[g, :3] = rack_rgb
        elif body.startswith(liquid.BALL_PREFIX):
            model.geom_rgba[g, :3] = liquid_rgb
        elif body.startswith(tuple(f"{r}_" for r in TUBE_ROLE.values())) and model.geom_group[g] == 2:
            model.geom_rgba[g, 3] = glass_a
    model.vis.headlight.ambient[:] = rng.uniform(0.05, 0.25)
    model.vis.headlight.diffuse[:] = rng.uniform(0.2, 0.5)


def _jitter_camera(model: mujoco.MjModel, rng: np.random.Generator) -> None:
    """Move the overhead camera a little, as a real mount is never exactly where drawn."""
    cam = model.camera(OVERHEAD.name).id
    model.cam_pos[cam] += rng.uniform(-0.02, 0.02, 3)
    tilt = np.zeros(4)
    mujoco.mju_axisAngle2Quat(tilt, rng.normal(size=3), math.radians(rng.uniform(0, 2)))
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, model.cam_quat[cam], tilt)
    model.cam_quat[cam] = out
