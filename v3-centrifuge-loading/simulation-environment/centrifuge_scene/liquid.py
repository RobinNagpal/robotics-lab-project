"""The liquid: small, frictionless, free balls (v1's liquid, for a 50 ml tube).

Balls are settled inside a tube on its own before the scene is built, then copied into the scene where
they will stay. Counting balls is how tests and grading measure what a tube holds and what was spilled; the
arm itself must never read ball positions. It weighs tubes on the balance instead.
"""

from __future__ import annotations

import json
import math

import mujoco
import numpy as np

from .settings import CONVERTED, LIQUID, TIMESTEP, TUBE, Tube
from .tubes import add_tube, inner_radius

__all__ = ["add_balls", "balls_in", "capacity", "inner_radius", "settled_liquid", "spilled"]

BALL_PREFIX = "ball"


def stack(tube: Tube, count: int, rng: np.random.Generator) -> np.ndarray:
    """``count`` ball centres in the tube's frame, lowest first, in a close-packed stack.

    The stack is looser than the settled balls, so it can reach above the rim; above the rim it keeps the
    rim's width, and the balls drop in when the simulation starts.
    """
    points: list[tuple[float, float, float]] = []
    for _, layer in _layers(tube):
        points.extend(layer)
        if len(points) >= count:
            break
    chosen = np.array(points[:count])
    return chosen + rng.uniform(-0.01, 0.01, chosen.shape) * LIQUID.ball_radius


def _layers(tube: Tube):
    """Yield (height, ball centres) for each layer of the stack, from the floor up, without end."""
    rb = LIQUID.ball_radius
    d = 2 * rb * 1.02
    rows = int(tube.radius / (d * math.sqrt(3) / 2)) + 2
    z = tube.wall + rb
    n = 0
    while True:
        # Alternate layers shift by half a cell so they sit in the hollows below.
        shift = (d / 2, d * math.sqrt(3) / 6) if n % 2 else (0.0, 0.0)
        room = _room(tube, z)
        layer = []
        for j in range(-rows, rows + 1):
            for i in range(-rows - 1, rows + 2):
                x = i * d + (j % 2) * d / 2 + shift[0]
                y = j * d * math.sqrt(3) / 2 + shift[1]
                if room >= 0 and math.hypot(x, y) <= room:
                    layer.append((x, y, z))
        yield z, layer
        z += d * math.sqrt(2 / 3)
        n += 1


def _room(tube: Tube, z: float) -> float:
    """How far from the axis a ball centred at height z can sit without touching the wall (-1: nowhere).

    Checked across the ball's whole height, so a ball in the cone does not poke through the sloping wall.
    Above the rim the rim's radius is used.
    """
    rb = LIQUID.ball_radius
    room = math.inf
    for dz in np.linspace(-rb, rb, 9):
        wall = inner_radius(tube, min(z + dz, tube.height))
        if wall == 0.0:
            return -1.0
        room = min(room, wall - math.sqrt(max(rb**2 - dz**2, 0.0)))
    return room


def capacity(tube: Tube = TUBE) -> int:
    """How many balls fit in the tube, settled, up to its rim. Measured once and cached."""
    cache = CONVERTED / "tube50_capacity.json"
    if cache.exists():
        return json.loads(cache.read_text())["balls"]
    full = 0
    for z, layer in _layers(tube):
        if z + LIQUID.ball_radius > tube.height:
            break
        full += len(layer)
    # Overfill, settle, and count what stays below the rim.
    model, data = _settle_alone(tube, stack(tube, int(full * 1.3), np.random.default_rng(0)))
    inside = balls_in(model, data, tube, "tube")
    below = int(np.sum(inside[:, 2] + LIQUID.ball_radius <= tube.height)) if len(inside) else 0
    CONVERTED.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"balls": below}))
    return below


def settled_liquid(tube: Tube, count: int, rng: np.random.Generator) -> np.ndarray:
    """Ball centres, in the tube's frame, of ``count`` balls already settled at the bottom of the tube."""
    model, data = _settle_alone(tube, stack(tube, count, rng))
    return balls_in(model, data, tube, "tube")


def _settle_alone(tube: Tube, centres: np.ndarray) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Simulate the balls in the tube, fixed upright at the origin, until they stop."""
    spec = mujoco.MjSpec()
    spec.option.timestep = TIMESTEP
    spec.option.solver = mujoco.mjtSolver.mjSOL_CG
    spec.option.iterations = 100
    spec.option.enableflags |= mujoco.mjtEnableBit.mjENBL_SLEEP
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.1])
    add_tube(spec, spec.worldbody, "tube", [0, 0, 0], capped=False, free=False, tube=tube)
    add_balls(spec, centres, 0, (0, 0, 1, 1))
    model = spec.compile()
    data = mujoco.MjData(model)
    while data.time < 2.5:
        mujoco.mj_step(model, data)
        if data.nbody_awake <= 1:
            break
    return model, data


def add_balls(spec: mujoco.MjSpec, centres: np.ndarray, start: int, rgba: tuple[float, ...]) -> None:
    """Add one free body per ball at the given world positions."""
    for i, c in enumerate(centres, start=start):
        body = spec.worldbody.add_body(name=f"{BALL_PREFIX}{i}", pos=c.tolist())
        body.add_freejoint()
        # priority 1 makes every contact with a ball use the ball's condim: 1 means no friction at all, like
        # a liquid, and is the cheapest contact MuJoCo solves.
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[LIQUID.ball_radius, 0, 0],
            density=LIQUID.ball_density,
            condim=1,
            priority=1,
            # The stiffest contact the time step allows, so a tall column of balls does not sink into itself.
            solref=[2 * TIMESTEP, 1.0],
            rgba=list(rgba),
        )


def ball_body_ids(model: mujoco.MjModel) -> np.ndarray:
    return np.array(
        [i for i in range(model.nbody) if model.body(i).name.startswith(BALL_PREFIX)], dtype=np.int64
    )


def balls_in(model: mujoco.MjModel, data: mujoco.MjData, tube: Tube, body: str) -> np.ndarray:
    """Positions, in the tube's frame, of the balls inside it now. For tests and grading only."""
    ids = ball_body_ids(model)
    if len(ids) == 0:
        return np.zeros((0, 3))
    b = model.body(body).id
    local = (data.xpos[ids] - data.xpos[b]) @ data.xmat[b].reshape(3, 3)
    inside = [p for p in local if (r := inner_radius(tube, p[2])) > 0 and math.hypot(p[0], p[1]) < r]
    return np.array(inside).reshape(-1, 3)


def spilled(model: mujoco.MjModel, data: mujoco.MjData, tubes: list[str], tube: Tube = TUBE) -> int:
    """How many balls are in no tube at all."""
    return len(ball_body_ids(model)) - sum(len(balls_in(model, data, tube, name)) for name in tubes)
