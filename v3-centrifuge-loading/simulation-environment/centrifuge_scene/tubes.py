"""The 50 ml tubes: AutoBio's mesh for looks, a ring of thin boxes for collision.

MuJoCo collides with a mesh as if it were its convex hull, so a tube mesh would be a solid rod and the balls
would sit on top of it. Instead each piece of the outline becomes a ring of thin boxes (as in v1, and as
AutoBio builds its own tubes), and the mesh is only drawn. A capped tube also gets a solid cylinder over its
rim, so nothing falls out when it is tilted into the rotor.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from .settings import CAP_OVERLAP, CAP_PLASTIC, TUBE, TUBE_MESHES, TUBE_PLASTIC, Tube

# Box halves are this much wider than a perfect ring needs, so neighbouring boxes overlap and leave no gap a
# ball could squeeze through.
RING_OVERLAP = 1.15


def ring_boxes(
    outline: tuple[tuple[float, float], ...], wall: float, pieces: int, inward: bool = True
) -> list[tuple[list[float], list[float], list[float]]]:
    """(half sizes, position, quaternion) of a ring of boxes along ``outline``, in the outline's frame.

    The wall lies inside the outline when ``inward`` (a tube: the outline is its outside), and outside it
    otherwise (a hole: the outline is its inside).
    """
    boxes = []
    for (z1, r1), (z2, r2) in zip(outline, outline[1:], strict=False):
        dz, dr = z2 - z1, r2 - r1
        length = math.hypot(dz, dr)
        along = np.array([dr, dz]) / length  # (radial, up)
        outward = np.array([dz, -dr]) / length
        middle = np.array([(r1 + r2) / 2, (z1 + z2) / 2]) + outward * wall / 2 * (-1 if inward else 1)
        half_width = (max(r1, r2) + (0 if inward else wall)) * math.tan(math.pi / pieces) * RING_OVERLAP
        # Lengthened by half a wall so the joints between pieces stay closed.
        half = [half_width, wall / 2, length / 2 + wall / 2]
        for i in range(pieces):
            theta = 2 * math.pi * i / pieces
            radial = np.array([math.cos(theta), math.sin(theta), 0.0])
            up = np.array([0.0, 0.0, 1.0])
            x_axis = np.array([math.sin(theta), -math.cos(theta), 0.0])
            y_axis = outward[0] * radial + outward[1] * up
            z_axis = along[0] * radial + along[1] * up
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, np.column_stack([x_axis, y_axis, z_axis]).flatten())
            pos = middle[0] * radial + middle[1] * up
            boxes.append((half, pos.tolist(), quat.tolist()))
    return boxes


def inner_radius(tube: Tube, z: float) -> float:
    """The inside radius at height z, or 0 below the floor and above the rim."""
    if z < tube.wall or z > tube.height:
        return 0.0
    heights, radii = zip(*tube.outline, strict=True)
    return max(0.0, float(np.interp(z, heights, radii)) - tube.wall)


def inner_volume(tube: Tube, top: float, steps: int = 400) -> float:
    """Volume inside the tube from its floor up to height ``top``, m^3."""
    zs = np.linspace(tube.wall, min(top, tube.height), steps)
    return float(np.trapezoid([math.pi * inner_radius(tube, z) ** 2 for z in zs], zs))


def add_meshes(spec: mujoco.MjSpec) -> None:
    """The tube body and cap meshes (AutoBio's, in millimetres), once per scene."""
    if spec.mesh("tube50_body") is None:
        for part, file in (
            ("body", "centrifuge_tube_50ml_body.STL"),
            ("cap", "centrifuge_tube_50ml_cap.STL"),
        ):
            spec.add_mesh(name=f"tube50_{part}", file=str(TUBE_MESHES / file), scale=[0.001] * 3)


def add_tube(
    spec: mujoco.MjSpec,
    parent: mujoco.MjsBody,
    name: str,
    pos: list[float],
    capped: bool,
    body_mass: float = TUBE.body_mass,
    quat: list[float] | None = None,
    free: bool = True,
    tube: Tube = TUBE,
    cap_later: bool = False,
) -> mujoco.MjsBody:
    """Add a tube with its tip at ``pos``, free or fixed where it stands, its mass spread over its walls.

    ``cap_later`` (an open tube that Step 3 caps once filled): the cap is built but switched off (not drawn,
    nothing touches it, no mass) until ``put_cap_on`` switches it on.
    """
    add_meshes(spec)
    body = parent.add_body(name=name, pos=pos, quat=quat or [1, 0, 0, 0])
    if free:
        body.add_freejoint(name=f"{name}_joint")
    body.add_geom(
        name=f"{name}_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="tube50_body",
        rgba=list(TUBE_PLASTIC.rgba),
        contype=0,
        conaffinity=0,
        group=2,
        mass=0,
    )
    boxes = ring_boxes(tube.outline, tube.wall, tube.wall_pieces)
    bottom_r = tube.outline[0][1]
    volume = sum(8 * h[0] * h[1] * h[2] for h, _, _ in boxes) + math.pi * bottom_r**2 * tube.wall
    common = dict(
        friction=[TUBE_PLASTIC.friction, 0.005, 0.0001],
        density=body_mass / volume,
        group=3,
        rgba=[0.5, 0.5, 0.5, 0.3],
    )
    body.add_geom(
        name=f"{name}_bottom",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[bottom_r, tube.wall / 2, 0],
        pos=[0, 0, tube.wall / 2],
        **common,
    )
    for i, (half, p, q) in enumerate(boxes):
        body.add_geom(
            name=f"{name}_wall{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=half, pos=p, quat=q, **common
        )
    if capped or cap_later:
        # AutoBio's cap mesh points down from its top; turned over, its top is the tube's capped height.
        top = tube.capped_height
        body.add_geom(
            name=f"{name}_cap_visual",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="tube50_cap",
            pos=[0, 0, top],
            quat=[0, 1, 0, 0],
            rgba=list(CAP_PLASTIC.rgba),
            contype=0,
            conaffinity=0,
            group=2 if capped else HIDDEN_GROUP,
            mass=0,
        )
        body.add_geom(
            name=f"{name}_cap",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[tube.cap_radius, tube.cap_height / 2, 0],
            pos=[0, 0, top - tube.cap_height / 2],
            mass=tube.cap_mass if capped else 0,
            friction=[CAP_PLASTIC.friction, 0.005, 0.0001],
            group=3,
            rgba=[0.5, 0.5, 0.5, 0.3],
            # Switched off: a collision bit nothing else has, so it touches nothing (with both 0, MuJoCo
            # leaves it out of the body's collision tree, and switching it on later would do nothing).
            contype=1 if capped else OFF_BIT,
            conaffinity=1 if capped else 0,
        )
    return body


# Geoms in this group are never drawn (cameras draw groups 0 to 2).
HIDDEN_GROUP = 5
OFF_BIT = 1 << 4


def cap_on(model: mujoco.MjModel, name: str) -> bool:
    """Whether tube ``name`` has its cap on (built capped, or capped later)."""
    try:
        return bool(model.geom_conaffinity[model.geom(f"{name}_cap").id])
    except KeyError:
        return False


def put_cap_on(model: mujoco.MjModel, data: mujoco.MjData, name: str, tube: Tube = TUBE) -> None:
    """Switch on the cap of a tube built with ``cap_later``: drawn, touched, and its mass added.

    This stands in for the cap's thread catching: the cap the arm carries is let go here, and from now on the
    tube is a capped tube like a sample, 2 g heavier.
    """
    if cap_on(model, name):
        return
    model.geom_group[model.geom(f"{name}_cap_visual").id] = 2
    cap = model.geom(f"{name}_cap").id
    model.geom_contype[cap] = 1
    model.geom_conaffinity[cap] = 1
    b = model.body(name).id
    m_old, m_cap = float(model.body_mass[b]), tube.cap_mass
    p_old = model.body_ipos[b].copy()
    c = np.array([0.0, 0.0, tube.capped_height - tube.cap_height / 2])  # the cap's centre, tube's frame

    # Only the mass and the three inertia values change (the tube is round about its axis, so the inertia's
    # axes stay the tube's own). The balance point is left where it was, 8 mm below where the cap would put
    # it: changed while the simulation runs (body_ipos, or turning body_iquat), MuJoCo flung the tube across
    # the bench at the next grip. The weight, and so the rotor's balance, get the cap's full 2 g.
    d = c - p_old
    r, h = tube.cap_radius, tube.cap_height
    cap_own = np.array([m_cap * (3 * r**2 + h**2) / 12] * 2 + [m_cap * r**2 / 2])
    model.body_inertia[b] += cap_own + m_cap * (float(d @ d) - d**2)
    model.body_mass[b] = m_old + m_cap
    model.body_subtreemass[b] += m_cap
    model.body_subtreemass[0] += m_cap


def top_of(tube: Tube, capped: bool) -> float:
    """Height of the tube's highest point above its tip."""
    return tube.capped_height if capped else tube.height


__all__ = ["CAP_OVERLAP", "add_tube", "inner_radius", "inner_volume", "ring_boxes", "top_of"]
