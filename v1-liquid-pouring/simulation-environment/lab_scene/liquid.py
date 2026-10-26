"""The liquid: small, frictionless, free balls.

Balls start in a close-packed stack inside a container and settle when the
simulation runs. Counting balls is how the examiner measures levels and
spills; the arm itself must never read ball positions.
"""

from __future__ import annotations

import json
import math

import mujoco
import numpy as np

from .glassware import add_container, inner_radius, inner_volume
from .settings import CONVERTED, LIQUID, TIMESTEP, Container

BALL_PREFIX = "ball"
BALL_VOLUME = 4 / 3 * math.pi * LIQUID.ball_radius**3


def ball_count(container: Container, level: float) -> int:
    """How many balls settle to ``level`` (0 to 1) of the container's inside height."""
    top = container.wall + level * (container.height - container.wall)
    return int(round(inner_volume(container, top) * settled_packing(container) / BALL_VOLUME))


def stack(container: Container, count: int, rng: np.random.Generator) -> np.ndarray:
    """``count`` ball centres in the container's frame, lowest first, in a close-packed stack.

    The stack is looser than the balls are once settled, so for a high level
    it reaches above the rim. Above the rim it keeps the rim's width, and the
    balls drop in when the simulation starts.
    """
    points: list[tuple[float, float, float]] = []
    for _, layer in _layers(container):
        points.extend(layer)
        if len(points) >= count:
            break
    chosen = np.array(points[:count])
    return chosen + rng.uniform(-0.01, 0.01, chosen.shape) * LIQUID.ball_radius


def _layers(container: Container):
    """Yield (height, ball centres) for each layer of the stack, from the floor up, without end."""
    rb = LIQUID.ball_radius
    d = 2 * rb * 1.02
    rows = int(container.radius / (d * math.sqrt(3) / 2)) + 2
    z = container.wall + rb
    n = 0
    while True:
        # Shift alternate layers by half a cell so they sit in the hollows below.
        shift = (d / 2, d * math.sqrt(3) / 6) if n % 2 else (0.0, 0.0)
        room = _room(container, z)
        layer = []
        for j in range(-rows, rows + 1):
            for i in range(-rows - 1, rows + 2):
                x = i * d + (j % 2) * d / 2 + shift[0]
                y = j * d * math.sqrt(3) / 2 + shift[1]
                if math.hypot(x, y) <= room:
                    layer.append((x, y, z))
        yield z, layer
        z += d * math.sqrt(2 / 3)
        n += 1


def _room(container: Container, z: float) -> float:
    """How far from the axis a ball centred at height z can sit without touching the wall.

    Checked across the ball's whole height, so a ball in a cone does not
    poke through the sloping wall. Above the rim the rim's radius is used.
    """
    rb = LIQUID.ball_radius
    room = math.inf
    for dz in np.linspace(-rb, rb, 9):
        wall = inner_radius(container, min(z + dz, container.height))
        if wall == 0.0:
            return -1.0
        room = min(room, wall - math.sqrt(max(rb**2 - dz**2, 0.0)))
    return room


def settled_packing(container: Container) -> float:
    """The fraction of the inside that settled balls fill, measured once and cached.

    Narrow tubes pack looser than wide beakers, because fewer balls fit
    across, so this is measured in the container itself: fill it to the
    rim, let the balls settle, and divide their volume by the volume below
    the surface they settle to.
    """
    cache = CONVERTED / f"{container.name}_packing.json"
    if cache.exists():
        return json.loads(cache.read_text())["packing"]
    full = 0
    for z, layer in _layers(container):
        if z + LIQUID.ball_radius > container.height:
            break
        full += len(layer)
    model, data = _settle_alone(container, stack(container, full, np.random.default_rng(0)))
    surface = level(model, data, container, container.name)
    top = container.wall + surface * (container.height - container.wall)
    packing = full * BALL_VOLUME / inner_volume(container, top)
    CONVERTED.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"packing": packing, "balls": full, "settled_level": surface}))
    return packing


def settled_liquid(container: Container, level_: float, rng: np.random.Generator) -> np.ndarray:
    """Ball centres, in the container's frame, of liquid already settled to ``level_``.

    The balls are poured in and settled in the container alone, held
    upright, before the scene is built. In the scene a tube leans a few
    degrees in its rack cell, and balls dropped in there from above the rim
    would miss it; settled ones start where they will stay.
    """
    model, data = _settle_alone(container, stack(container, ball_count(container, level_), rng))
    return balls_in(model, data, container, container.name)


def _settle_alone(container: Container, centres: np.ndarray) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Simulate the balls in the container, fixed upright at the origin, until they stop."""
    spec = mujoco.MjSpec()
    spec.option.timestep = TIMESTEP
    spec.option.solver = mujoco.mjtSolver.mjSOL_CG
    spec.option.iterations = 100
    spec.option.enableflags |= mujoco.mjtEnableBit.mjENBL_SLEEP
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.1])
    add_container(spec, spec.worldbody, container, [0, 0, 0], free=False)
    add_balls(spec, centres, 0)
    model = spec.compile()
    data = mujoco.MjData(model)
    while data.time < 2.0:
        mujoco.mj_step(model, data)
        if data.nbody_awake <= 1:
            break
    return model, data


def add_balls(spec: mujoco.MjSpec, centres: np.ndarray, start: int) -> None:
    """Add one free body per ball at the given world positions."""
    for i, c in enumerate(centres, start=start):
        body = spec.worldbody.add_body(name=f"{BALL_PREFIX}{i}", pos=c.tolist())
        body.add_freejoint()
        # priority 1 makes every contact with a ball use the ball's condim:
        # 1 means no friction at all, like a liquid, and is the cheapest
        # contact MuJoCo solves.
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[LIQUID.ball_radius, 0, 0],
            density=LIQUID.ball_density,
            condim=1,
            priority=1,
            # The stiffest contact the time step allows. MuJoCo's default is
            # ten times softer, and under the weight of a tall column the
            # balls sink a millimetre or more into each other.
            solref=[2 * TIMESTEP, 1.0],
            rgba=list(LIQUID.rgba),
        )


def ball_body_ids(model: mujoco.MjModel) -> np.ndarray:
    return np.array(
        [i for i in range(model.nbody) if model.body(i).name.startswith(BALL_PREFIX)], dtype=np.int64
    )


def balls_in(model: mujoco.MjModel, data: mujoco.MjData, container: Container, body: str) -> np.ndarray:
    """Positions, in the container's frame, of the balls inside it now. For the examiner only."""
    ids = ball_body_ids(model)
    b = model.body(body).id
    local = (data.xpos[ids] - data.xpos[b]) @ data.xmat[b].reshape(3, 3)
    inside = [p for p in local if (r := inner_radius(container, p[2])) > 0 and math.hypot(p[0], p[1]) < r]
    return np.array(inside).reshape(-1, 3)


def level(model: mujoco.MjModel, data: mujoco.MjData, container: Container, body: str) -> float:
    """The liquid's surface height as a fraction of the inside height, 0 when empty.

    The surface is the top of the highest balls, ignoring the few that sit
    loose above the rest.
    """
    balls = balls_in(model, data, container, body)
    if len(balls) == 0:
        return 0.0
    surface = float(np.percentile(balls[:, 2], 97)) + LIQUID.ball_radius
    return (surface - container.wall) / (container.height - container.wall)


def spilled(model: mujoco.MjModel, data: mujoco.MjData, containers: dict[str, Container]) -> int:
    """How many balls are in no container at all."""
    total = len(ball_body_ids(model))
    held = sum(len(balls_in(model, data, c, name)) for name, c in containers.items())
    return total - held
