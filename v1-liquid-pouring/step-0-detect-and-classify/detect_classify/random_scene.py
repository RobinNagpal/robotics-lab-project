"""A lab scene where the seed picks everything the detector must not memorise.

The simulation's own scene (lab_scene.scene) keeps the racks, tray and stand
in fixed places, so a detector trained on it would learn where things usually
are, not what they look like. Here the seed also picks how many containers
there are and of which kinds, where everything stands and how it is turned,
the colours, the lights, the camera's exact mount and the arm's pose.

The building blocks (racks, glassware, liquid) are the simulation's own. The
scene is only rendered, never stepped: the liquid is settled before it is
placed, so one mj_forward gives the picture.
"""

from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass, field

import mujoco
import numpy as np
from lab_scene import liquid
from lab_scene.glassware import add_container
from lab_scene.racks import RACK_FLOOR, add_rack, add_stand, add_tray
from lab_scene.scene import ARM_JOINTS, GRIPPER_PREFIX
from lab_scene.settings import (
    ARM_XML,
    GRIPPER_XML,
    HOME,
    LIQUID,
    OVERHEAD,
    RACK,
    RECEIVERS,
    SOURCES,
    TABLE,
    Container,
)

from .classes import CONTAINER_CLASS

IMAGE_SIZE = (1280, 960)

BY_NAME = {c.name: c for c in SOURCES + RECEIVERS}
# Tubes that can stand in a rack. Any of them can hold liquid; the receivers
# in the empty rack are always empty.
RACK_TUBES = ("receiver_100", "receiver_150", "receiver_200", "source_tube", "source_centrifuge_tube")
RECEIVER_TUBES = ("receiver_100", "receiver_150", "receiver_200")
GLASSWARE = ("source_beaker", "source_flask")

TRAY_SIZE = (0.20, 0.11)
# Half the diagonal of each footprint, plus a little room, for keeping things apart.
RACK_REACH = math.hypot(0.1025, 0.048) + 0.01
TRAY_REACH = math.hypot(TRAY_SIZE[0] / 2, TRAY_SIZE[1] / 2) + 0.01
STAND_REACH = math.hypot(0.05, 0.05) + 0.01
# Where things may stand: within the arm's reach, clear of its base, and inside
# what the overhead camera sees of the table (x -0.16..0.96, y -0.75..0.75).
REACH = (0.30, 0.80)
VIEW_X = (-0.08, 0.90)
VIEW_Y = 0.66


@dataclass
class Instance:
    """One object in the scene, for labels and grading. The arm must not read it."""

    name: str  # MuJoCo body name
    cls: str
    role: str  # source_rack, empty_rack, filled_rack, source, receiver, tray, stand
    top: float  # height of its highest point above the table, m
    level: float = 0.0  # liquid level as a fraction of the inside height; sources only
    balls: tuple[int, int] = (0, 0)  # first ball number and how many, for the liquid inside it


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
    """Picks spots on the table that keep every footprint apart."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.taken: list[tuple[np.ndarray, float]] = []

    def spot(self, reach: float) -> np.ndarray:
        for _ in range(2000):
            r = self.rng.uniform(*REACH)
            a = self.rng.uniform(-math.radians(80), math.radians(80))
            p = np.array([r * math.cos(a), r * math.sin(a)])
            if not (VIEW_X[0] + reach < p[0] < VIEW_X[1] - reach and abs(p[1]) < VIEW_Y - reach):
                continue
            if all(np.linalg.norm(p - q) > reach + rq for q, rq in self.taken):
                self.taken.append((p, reach))
                return p
        raise RuntimeError("no free spot on the table")


def _random_rgb(rng: np.random.Generator, sat: tuple[float, float], val: tuple[float, float]) -> list[float]:
    return list(colorsys.hsv_to_rgb(rng.uniform(), rng.uniform(*sat), rng.uniform(*val)))


def build(seed: int) -> tuple[mujoco.MjModel, mujoco.MjData, Scene]:
    """Build, colour and pose the scene for one seed. Returns it ready to render."""
    rng = np.random.default_rng(seed)
    scene = Scene(seed=seed)
    place = _Placer(rng)

    spec = mujoco.MjSpec.from_file(str(ARM_XML))
    # The gripper's own settings, as in lab_scene.scene; without them attaching it warns.
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    gripper = mujoco.MjSpec.from_file(str(GRIPPER_XML))
    spec.attach(gripper, prefix=GRIPPER_PREFIX, site=spec.site("attachment_site"))
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

    containers: list[tuple[str, Container, np.ndarray, str]] = []  # instance, container, bottom, role

    # Three racks, each turned any way. Which rack plays which role is only
    # known from what stands in it.
    for role in ("source_rack", "empty_rack", "filled_rack"):
        xy, yaw = place.spot(RACK_REACH), rng.uniform(0, 2 * math.pi)
        body = add_rack(spec, role, (float(xy[0]), float(xy[1])))
        body.quat = _yaw_quat(yaw)
        scene.instances.append(Instance(role, "rack", role, RACK.height))
        if role == "filled_rack":
            continue
        count = rng.integers(1, 5) if role == "source_rack" else rng.integers(1, 6)
        kinds = RACK_TUBES if role == "source_rack" else RECEIVER_TUBES
        for cell in rng.choice(RACK.rows * RACK.columns, size=count, replace=False):
            row, col = divmod(int(cell), RACK.columns)
            local = (RACK.pitch * (col - (RACK.columns - 1) / 2), RACK.pitch * (row - (RACK.rows - 1) / 2))
            bottom = np.array([*(xy + _rotate(local, yaw)), RACK.height / 2 + RACK_FLOOR])
            kind = str(rng.choice(kinds))
            tube_role = "source" if role == "source_rack" else "receiver"
            containers.append((f"{tube_role}_{len(containers)}", BY_NAME[kind], bottom, tube_role))

    # Beakers and flasks: on the tray if there is one, otherwise on the table.
    glass = [str(g) for g in rng.choice(GLASSWARE, size=rng.integers(0, 3))]
    if rng.random() < 0.8:
        xy, yaw = place.spot(TRAY_REACH), rng.uniform(0, 2 * math.pi)
        floor = add_tray(spec, "tray", (float(xy[0]), float(xy[1])), TRAY_SIZE)
        spec.body("tray").quat = _yaw_quat(yaw)
        scene.instances.append(Instance("tray", "tray", "tray", 0.012))
        for kind, dx in zip(glass[:2], (-0.045, 0.045), strict=False):
            bottom = np.array([*(xy + _rotate((dx, 0.0), yaw)), floor])
            containers.append((f"source_{len(containers)}", BY_NAME[kind], bottom, "source"))
        glass = glass[2:]
    for kind in glass:
        c = BY_NAME[kind]
        xy = place.spot(c.radius + 0.01)
        containers.append((f"source_{len(containers)}", c, np.array([*xy, 0.0]), "source"))

    xy = place.spot(STAND_REACH)
    add_stand(spec, "pouring_stand", (float(xy[0]), float(xy[1])))
    spec.body("pouring_stand").quat = _yaw_quat(rng.uniform(0, 2 * math.pi))
    scene.instances.append(Instance("pouring_stand", "pouring_stand", "stand", 0.066))

    start = 0
    for name, c, bottom, role in containers:
        add_container(spec, world, c, bottom.tolist(), name=name)
        level, balls = 0.0, (start, 0)
        if role == "source":
            level = float(rng.uniform(*LIQUID.level))
            centres = liquid.settled_liquid(c, level, rng) + bottom
            liquid.add_balls(spec, centres, start)
            balls = (start, len(centres))
            start += len(centres)
        top = float(bottom[2] + c.height)
        scene.instances.append(Instance(name, CONTAINER_CLASS[c.name], role, top, level, balls))

    model = spec.compile()
    _recolour(model, rng)
    _jitter_camera(model, rng)
    data = mujoco.MjData(model)
    # The arm near home, as it would be between moves.
    for joint, q in zip(ARM_JOINTS, HOME, strict=True):
        data.qpos[model.joint(joint).qposadr[0]] = q + rng.uniform(-0.15, 0.15)
    mujoco.mj_forward(model, data)
    return model, data, scene


def _recolour(model: mujoco.MjModel, rng: np.random.Generator) -> None:
    """New colours for the table, plastic, glass and liquid, so the detector learns shapes."""
    plastic = {model.body(n).id for n in ("source_rack", "empty_rack", "filled_rack")}
    if "tray" in [model.body(i).name for i in range(model.nbody)]:
        plastic.add(model.body("tray").id)
    rack_rgb = _random_rgb(rng, (0.0, 0.9), (0.3, 0.95))
    glass_a = rng.uniform(0.2, 0.45)
    liquid_rgb = _random_rgb(rng, (0.3, 1.0), (0.3, 1.0))
    for g in range(model.ngeom):
        body = model.body(model.geom_bodyid[g]).name
        if model.geom(g).name == "table":
            model.geom_rgba[g, :3] = _random_rgb(rng, (0.0, 0.3), (0.45, 0.95))
        elif model.geom_bodyid[g] in plastic:
            model.geom_rgba[g, :3] = rack_rgb
        elif body == "pouring_stand":
            model.geom_rgba[g, :3] = [rng.uniform(0.15, 0.6)] * 3
        elif body.startswith(liquid.BALL_PREFIX):
            model.geom_rgba[g, :3] = liquid_rgb
        elif body.startswith(("source_", "receiver_")) and model.geom_group[g] == 2:
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
