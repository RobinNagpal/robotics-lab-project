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
) -> mujoco.MjsBody:
    """Add a tube with its tip at ``pos``, free or fixed where it stands, its mass spread over its walls."""
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
    if capped:
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
            group=2,
            mass=0,
        )
        body.add_geom(
            name=f"{name}_cap",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[tube.cap_radius, tube.cap_height / 2, 0],
            pos=[0, 0, top - tube.cap_height / 2],
            mass=tube.cap_mass,
            friction=[CAP_PLASTIC.friction, 0.005, 0.0001],
            group=3,
            rgba=[0.5, 0.5, 0.5, 0.3],
        )
    return body


def top_of(tube: Tube, capped: bool) -> float:
    """Height of the tube's highest point above its tip."""
    return tube.capped_height if capped else tube.height


__all__ = ["CAP_OVERLAP", "add_tube", "inner_radius", "inner_volume", "ring_boxes", "top_of"]
