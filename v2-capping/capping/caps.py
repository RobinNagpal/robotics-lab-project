"""Screw caps for the test tubes, which come off only by turning.

The cap is a standard **24-410** screw cap, the one 25 x 150 mm screw-cap culture tubes use (Kimble,
Pyrex, Fisherbrand): a 24 mm thread of 8 turns per inch (3.175 mm per turn) that passes round 1.5
times, so the cap comes off after 1.5 turns anticlockwise (seen from above). It is a black phenolic
cap, 28 mm across with grip ribs on its side and 14 mm tall. No free model of one fits; the shape here
is made from those numbers.

**How the thread works in the simulation.** AutoBio models real threads with its own MuJoCo plugin, which
is only built for Linux, and MuJoCo's own bolt and nut are 1 m long with an 83 mm pitch. So the thread is
made from plain joints, as MuJoCo's documentation suggests for a screw. A cap has two bodies:

- **On the thread**, the cap is a body inside the top of the tube that can only turn about the tube's
  axis (a hinge) and move along it (a slide), tied together: every turn anticlockwise lifts it one pitch.
  It stops at 0 (screwed tight) and at 1.5 turns (the end of the thread), and resists turning as a real
  cap does: hard to start (``BREAKAWAY_TORQUE``, the seal sticks), then easy (``RUNNING_TORQUE``).
  Pulling it straight up does nothing: it can only rise by turning.
- **Off the thread**, the cap is an ordinary free body (``<tube>_cap``). Until then it is not there for
  anything to touch or see, and is only kept in the same place as the cap on the thread. At the end of
  the thread they swap: the free cap appears exactly where the threaded one was, moving as it moved, and
  the threaded one goes. Fingers holding it go on holding it.

(A first version had only the free cap, welded to the thread while on it. A 4 g free body squeezed by
fingers between a stiff weld and stiff contacts made the simulation unstable again and again. On the
thread the cap is now part of the tube's own joints, as a real thread guides it.)

The swap, and the friction changing as the seal breaks, are part of the physics, so a capped scene is
stepped with ``caps.step(model, data)`` instead of ``mujoco.mj_step``. (MuJoCo's own per-step hook would
do it for every step, but the Python bindings fail when it is set while any model is built.) The swap
changes the model (which geoms collide and show), so ``caps.reset(model, data)`` puts every cap back on.
An arm opening a cap is told nothing by any of this, and has to feel when the cap is off, as a real one
does: by the torque it takes to turn, and by whether it lifts freely.

The tube itself is not changed: its thread is not drawn (the cap covers that part of a real tube).

``capping.scene.build(seed)`` builds the v1 scene with one on each receiver.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from lab_scene.glassware import wall_boxes
from lab_scene.settings import Container, Material

# The cap's mesh is made on first use and kept here (not in v1's assets).
CONVERTED = Path(__file__).resolve().parents[1] / "assets" / "converted"

# The 24-410 finish: thread pitch 8 per inch, 1.5 turns.
PITCH = 0.0254 / 8  # m per turn
TURNS = 1.5
OPEN_ANGLE = 2 * math.pi * TURNS  # rad of anticlockwise turning, from tight to off the thread

# The cap (a black phenolic 24-410 cap; the liner is not shown).
OUTER_R = 0.0140  # m, the side between the ribs
RIB = 0.0004  # m, how far the grip ribs stand out
RIBS = 48
INNER_R = 0.0130  # m: 0.5 mm round the 25 mm tube, where a real cap's thread would be
HEIGHT = 0.014  # m, all of it
TOP = 0.0025  # m, the closed top's thickness
SKIRT = HEIGHT - TOP  # m, how far the side reaches down over the tube
GAP = 0.0005  # m between the cap's inside top and the top of the tube's collision walls, screwed tight
MASS = 0.004  # kg
PHENOLIC = Material(friction=0.8, rgba=(0.07, 0.07, 0.07, 1.0))

# The thread's friction. 24 mm caps are put on at 10 to 18 in-lb (1.1 to 2.0 N m) and take 40 to 60 % of
# that to start undoing (0.45 to 1.25 N m); a lab tube tightened by hand is at the low end. Once the seal
# breaks the cap turns much more easily; no published figure was found for that, so a sixth is assumed.
BREAKAWAY_TORQUE = 0.6  # N m, to start turning a tight cap
RUNNING_TORQUE = 0.1  # N m, to keep it turning
SEAL_ANGLE = math.radians(20)  # how far it turns before the seal has let go
END_MARGIN = 0.02 * 2 * math.pi  # rad: within this of the end of the thread, the cap is off
# Armature on the thread's joints (inertia only, no weight). MuJoCo sets how far a constraint gives way
# from the inertia of the joints it ties, ignoring the gear between them: with only the cap's own tiny
# inertia about its axis (6e-7 kg m^2) the screw gave way 4.7 mm under a 2 N pull. It makes the cap a
# little slow to spin up, which a gripper turning it at about a turn a second does not feel.
TURN_ARMATURE = 1e-4  # kg m^2
RISE_ARMATURE = MASS  # kg
# MuJoCo's constraints give way in proportion to the force over the mass they hold, and a 4 g cap pushed
# by fingers takes forces many times its weight. So the thread's constraints (the screw and its stops)
# are made nearly hard: the shortest time constant the 1 ms step allows, and an impedance close to 1.
SOLREF = (0.002, 1.0)
SOLIMP = (0.99, 0.999, 0.001, 0.5, 2.0)
HIDDEN = 1e-6  # kg: what the threaded cap weighs once the free one has taken over


def cap_name(tube: str) -> str:
    """The free cap, once it is off the thread."""
    return f"{tube}_cap"


def follower_name(tube: str) -> str:
    """The cap on the thread."""
    return f"{tube}_thread"


def _inertia() -> tuple[list[float], list[float]]:
    """Centre of mass and principal inertia of the cap, a thin-walled cup: its side's is m r^2 about the
    axis, its top's half that. In the cap's frame (origin: the middle of the inside of its top)."""
    side, top = MASS * SKIRT / HEIGHT, MASS * TOP / HEIGHT
    r2 = ((OUTER_R + INNER_R) / 2) ** 2
    axial = side * r2 + top * OUTER_R**2 / 2
    across = side * (r2 / 2 + SKIRT**2 / 12) + top * OUTER_R**2 / 4
    return [0, 0, TOP - HEIGHT / 2], [across, across, axial]


def _cap_geoms(body: mujoco.MjsBody, mesh: str, prefix: str, shown: bool) -> None:
    """The cap's look (the made mesh) and its touch: one solid cylinder, its outside. It need not be
    hollow, as it never touches its own tube. (A ring of boxes, as the tubes are built, gave the flat
    finger pads box corners to catch on, and the grip on it was erratic.)"""
    body.add_geom(
        name=f"{prefix}_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=mesh,
        rgba=list(PHENOLIC.rgba[:3]) + [1.0 if shown else 0.0],
        contype=0,
        conaffinity=0,
        group=2,
        mass=0,
    )
    body.add_geom(
        name=f"{prefix}_shell",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[OUTER_R, HEIGHT / 2, 0],
        pos=[0, 0, TOP - HEIGHT / 2],
        friction=[PHENOLIC.friction, 0.005, 0.0001],
        contype=1 if shown else 0,
        conaffinity=1 if shown else 0,
        mass=0,
        group=3,
        rgba=[0.5, 0.5, 0.5, 0.3 if shown else 0.0],
    )
    ipos, inertia = _inertia()
    body.mass = MASS
    body.ipos = ipos
    body.inertia = inertia
    body.explicitinertial = True


def visual_mesh_path() -> str:
    """The cap's shape, made once and kept in assets/converted/: a hollow cylinder closed at the top,
    with rounded grip ribs on the outside. Its origin is the middle of the inside of the top, so the
    side reaches down from z = 0 to z = -SKIRT and the top up to z = TOP."""
    path = CONVERTED / "cap_24_410.obj"
    if path.exists():
        return str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = RIBS * 8
    phi = np.linspace(0, 2 * math.pi, n, endpoint=False)
    r_out = OUTER_R + RIB * (0.5 + 0.5 * np.cos(RIBS * phi))
    c, s = np.cos(phi), np.sin(phi)

    def ring(r, z):
        r = np.broadcast_to(r, phi.shape)
        return np.stack([r * c, r * s, np.full(n, z)], axis=1)

    rings = [
        ring(r_out, -SKIRT),  # 0: outside, bottom
        ring(r_out, TOP),  # 1: outside, top
        ring(INNER_R, -SKIRT),  # 2: inside, bottom
        ring(INNER_R, 0.0),  # 3: inside, top (the ceiling's edge)
    ]
    verts = np.concatenate(rings + [np.array([[0, 0, TOP], [0, 0, 0.0]])])
    top_c, ceil_c = 4 * n, 4 * n + 1
    faces = []

    def band(a, b, outward=True):
        for i in range(n):
            j = (i + 1) % n
            p, q, r_, t = a * n + i, a * n + j, b * n + j, b * n + i
            faces.extend([[p, q, r_], [p, r_, t]] if outward else [[p, r_, q], [p, t, r_]])

    band(0, 1)  # the ribbed outside
    band(2, 3, outward=False)  # the inside of the side, facing in
    band(2, 0)  # the bottom edge of the side, between inside and outside
    for i in range(n):
        j = (i + 1) % n
        faces.append([top_c, n + i, n + j])  # the top, facing up
        faces.append([ceil_c, 3 * n + j, 3 * n + i])  # the inside of the top, facing down
    mesh = trimesh.Trimesh(verts, np.array(faces), process=True)  # faces wound to point outwards
    mesh.export(path)
    return str(path)


def add_cap(spec: mujoco.MjSpec, tube: mujoco.MjsBody, container: Container, tube_pos: list[float]) -> None:
    """Screw a cap onto ``tube`` (``container``, standing upright with its bottom at ``tube_pos``)."""
    name = tube.name
    # The tube's collision walls overlap a little and reach about 1 mm above the drawn rim.
    wall_top = max(p[2] + h[2] for h, p, _ in wall_boxes(container))
    seat = wall_top + GAP
    mesh = spec.add_mesh(name=f"{cap_name(name)}_mesh", file=visual_mesh_path()).name

    # The cap on the thread: part of the tube, turning and rising only as the thread lets it. (A child
    # never collides with its parent body, so it does not touch its tube: the thread carries that.)
    threaded = tube.add_body(name=follower_name(name), pos=[0, 0, seat])
    _cap_geoms(threaded, mesh, follower_name(name), shown=True)
    turn = threaded.add_joint(
        name=f"{follower_name(name)}_turn",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[0, 0, 1],
        # The scene's ranges are in degrees (MjSpec's default).
        range=[0, math.degrees(OPEN_ANGLE) if spec.compiler.degree else OPEN_ANGLE],
        limited=mujoco.mjtLimited.mjLIMITED_TRUE,
        frictionloss=BREAKAWAY_TORQUE,
        armature=TURN_ARMATURE,
        solref_limit=list(SOLREF),
        solimp_limit=list(SOLIMP),
        # Stiff friction too: with MuJoCo's default it crept, and pulling the cap straight up turned it
        # about 18 degrees through the thread. A real thread this fine cannot be turned by pulling.
        solref_friction=list(SOLREF),
        solimp_friction=list(SOLIMP),
    )
    rise = threaded.add_joint(
        name=f"{follower_name(name)}_rise",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[0, 0, 1],
        range=[0, PITCH * TURNS],
        limited=mujoco.mjtLimited.mjLIMITED_TRUE,
        armature=RISE_ARMATURE,
        solref_limit=list(SOLREF),
        solimp_limit=list(SOLIMP),
    )
    # rise = PITCH / (2 pi) x turn: one pitch up per turn anticlockwise.
    spec.add_equality(
        name=f"{follower_name(name)}_screw",
        type=mujoco.mjtEq.mjEQ_JOINT,
        objtype=mujoco.mjtObj.mjOBJ_JOINT,
        name1=rise.name,
        name2=turn.name,
        data=[0, PITCH / (2 * math.pi), 0, 0, 0] + [0] * (mujoco.mjNEQDATA - 5),
        solref=list(SOLREF),
        solimp=list(SOLIMP),
    )

    # The free cap, for once it is off: hidden, touching nothing, and kept with the threaded one.
    free = spec.worldbody.add_body(name=cap_name(name), pos=[tube_pos[0], tube_pos[1], tube_pos[2] + seat])
    free.add_freejoint(name=f"{cap_name(name)}_joint")
    _cap_geoms(free, mesh, cap_name(name), shown=False)
    spec.add_exclude(bodyname1=free.name, bodyname2=name)  # nor does it touch its tube, once off
    spec.add_equality(
        name=f"{cap_name(name)}_kept",
        type=mujoco.mjtEq.mjEQ_WELD,
        objtype=mujoco.mjtObj.mjOBJ_BODY,
        name1=free.name,
        name2=threaded.name,
        data=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1] + [0] * (mujoco.mjNEQDATA - 11),
    )


def tubes_with_caps(model: mujoco.MjModel) -> list[str]:
    names = [model.body(i).name for i in range(model.nbody)]
    return [n[: -len("_cap")] for n in names if n.endswith("_cap")]


def turned(model: mujoco.MjModel, data: mujoco.MjData, tube: str) -> float:
    """How far the cap on ``tube`` has been turned off, in turns (0 tight, 1.5 at the end of the thread)."""
    adr = model.joint(f"{follower_name(tube)}_turn").qposadr[0]
    return float(data.qpos[adr]) / (2 * math.pi)


def on_thread(model: mujoco.MjModel, data: mujoco.MjData, tube: str) -> bool:
    return bool(data.eq_active[model.equality(f"{cap_name(tube)}_kept").id])


def cap_body(model: mujoco.MjModel, data: mujoco.MjData, tube: str) -> int:
    """The body that is the cap now: the threaded one while on, the free one after."""
    return model.body(follower_name(tube) if on_thread(model, data, tube) else cap_name(tube)).id


def _show(model: mujoco.MjModel, prefix: str, shown: bool) -> None:
    visual, shell = model.geom(f"{prefix}_visual").id, model.geom(f"{prefix}_shell").id
    model.geom_rgba[visual, 3] = 1.0 if shown else 0.0
    model.geom_rgba[shell, 3] = 0.3 if shown else 0.0
    model.geom_contype[shell] = model.geom_conaffinity[shell] = 1 if shown else 0
    # MuJoCo first skips whole bodies none of whose geoms collide, by flags worked out when the model
    # was built: those must change too, or the free cap would pass through the fingers.
    body = model.geom_bodyid[shell]
    model.body_contype[body] = model.body_conaffinity[body] = 1 if shown else 0


def _swap(model: mujoco.MjModel, data: mujoco.MjData, tube: str, off: bool) -> None:
    """Off the thread (``off``): the free cap takes over from the threaded one; or back on."""
    _show(model, cap_name(tube), off)
    _show(model, follower_name(tube), not off)
    body = model.body(follower_name(tube)).id
    ipos, inertia = _inertia()
    model.body_mass[body] = HIDDEN if off else MASS
    model.body_inertia[body] = np.array(inertia) * (HIDDEN / MASS if off else 1.0)
    data.eq_active[model.equality(f"{cap_name(tube)}_kept").id] = 0 if off else 1


def release(model: mujoco.MjModel, data: mujoco.MjData) -> list[str]:
    """Take off every cap that has reached the end of its thread (``step`` does this every step).
    Returns the tubes whose caps came off just now."""
    freed = []
    for tube, _, turn_adr in _threads(model):
        if on_thread(model, data, tube) and data.qpos[turn_adr] >= OPEN_ANGLE - END_MARGIN:
            _swap(model, data, tube, off=True)
            freed.append(tube)
    return freed


def reset(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Every cap back on its thread (after ``mujoco.mj_resetData``, which does not undo a swap)."""
    for tube, _, _ in _threads(model):
        _swap(model, data, tube, off=False)
    mujoco.mj_forward(model, data)


def step(model: mujoco.MjModel, data: mujoco.MjData, n: int = 1) -> None:
    """``n`` simulation steps of a scene with caps: each cap's friction is set by how far it has turned
    (hard to start, then easy), the step is taken, and a cap at the end of its thread comes off."""
    threads = _threads(model)
    for _ in range(n):
        for _, turn_dof, turn_adr in threads:
            sealed = data.qpos[turn_adr] < SEAL_ANGLE
            model.dof_frictionloss[turn_dof] = BREAKAWAY_TORQUE if sealed else RUNNING_TORQUE
        mujoco.mj_step(model, data)
        if threads:
            release(model, data)


_THREADS: dict[tuple, list[tuple[str, int, int]]] = {}


def _threads(model: mujoco.MjModel) -> list[tuple[str, int, int]]:
    """(the tube, its thread's turn dof, the turn's qpos address) for every cap, found once per model."""
    key = (model._address, model.nbody, model.nq, model.neq)
    if key not in _THREADS:
        found = []
        for tube in tubes_with_caps(model):
            turn = model.joint(f"{follower_name(tube)}_turn")
            found.append((tube, int(turn.dofadr[0]), int(turn.qposadr[0])))
        _THREADS[key] = found
    return _THREADS[key]
