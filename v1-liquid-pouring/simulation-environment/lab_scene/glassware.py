"""Glassware: the downloaded mesh for looks, a ring of thin boxes for collision.

MuJoCo collides with a mesh as if it were its convex hull, so a tube mesh
would be a solid rod and the balls would sit on top of it. Instead each wall
piece of the outline becomes a ring of thin boxes, the way AutoBio builds its
own tubes, and the mesh is only drawn.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import trimesh

from .settings import CONVERTED, Container

# Box halves are this much wider than a perfect ring needs, so neighbouring
# boxes overlap and leave no gap a ball could squeeze through.
RING_OVERLAP = 1.15


def visual_mesh_path(container: Container) -> str:
    """Turn the mesh z-up, stand it on z = 0 and stretch it to the container's size.

    The result is cached in assets/converted/. Both faces of every triangle
    are kept, because the pack's glassware is a single sheet and would
    vanish from behind.
    """
    out = CONVERTED / f"{container.name}.obj"
    if out.exists():
        return str(out)
    mesh = trimesh.load(container.mesh, force="mesh")
    v = np.asarray(mesh.vertices, dtype=np.float64)
    if container.mesh_up == "y":
        v = v[:, [0, 2, 1]] * np.array([1.0, -1.0, 1.0])
    # Scale about the axis, which every mesh in use already sits on. The width
    # is matched in the lower 30 % of the height, the body: the widest point
    # overall can be a beaker's pouring spout, which would make the body too
    # narrow, and the balls would seem to stick out through the glass.
    v[:, 2] -= v[:, 2].min()
    radius = np.hypot(v[:, 0], v[:, 1])
    body = v[:, 2] < 0.3 * v[:, 2].max()
    band = 0.3 * container.height
    heights = [z for z, _ in container.outline if z < band] + [band]
    body_radius = max(float(np.interp(z, *zip(*container.outline, strict=True))) for z in heights)
    across = body_radius / np.percentile(radius[body], 90)
    up = container.height / v[:, 2].max()
    v *= np.array([across, across, up])
    faces = np.asarray(mesh.faces)
    both = trimesh.Trimesh(np.vstack([v, v]), np.vstack([faces, faces[:, ::-1] + len(v)]), process=False)
    CONVERTED.mkdir(parents=True, exist_ok=True)
    both.export(out)
    return str(out)


def wall_boxes(container: Container) -> list[tuple[list[float], list[float], list[float]]]:
    """(half sizes, position, quaternion) of every wall box, in the container's frame."""
    boxes = []
    t = container.wall
    k = container.wall_pieces
    points = container.outline
    for (z1, r1), (z2, r2) in zip(points, points[1:], strict=False):
        dz, dr = z2 - z1, r2 - r1
        length = math.hypot(dz, dr)
        along = np.array([dr, dz]) / length  # (radial, up)
        outward = np.array([dz, -dr]) / length
        middle = np.array([(r1 + r2) / 2, (z1 + z2) / 2]) - outward * t / 2
        half_width = max(r1, r2) * math.tan(math.pi / k) * RING_OVERLAP
        # Lengthen by half a wall so the joints between pieces stay closed.
        half = [half_width, t / 2, length / 2 + t / 2]
        for i in range(k):
            theta = 2 * math.pi * i / k
            radial = np.array([math.cos(theta), math.sin(theta), 0.0])
            up = np.array([0.0, 0.0, 1.0])
            # Chosen so x, y, z form a right-handed frame.
            x_axis = np.array([math.sin(theta), -math.cos(theta), 0.0])
            y_axis = outward[0] * radial + outward[1] * up
            z_axis = along[0] * radial + along[1] * up
            rot = np.column_stack([x_axis, y_axis, z_axis])
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, rot.flatten())
            pos = middle[0] * radial + middle[1] * up
            boxes.append((half, pos.tolist(), quat.tolist()))
    return boxes


def inner_radius(container: Container, z: float) -> float:
    """The inside radius at height z, or 0 below the floor and above the rim."""
    floor = container.wall
    if z < floor or z > container.height:
        return 0.0
    heights = [p[0] for p in container.outline]
    radii = [p[1] for p in container.outline]
    return max(0.0, float(np.interp(z, heights, radii)) - container.wall)


def inner_volume(container: Container, top: float, steps: int = 400) -> float:
    """Volume inside the container from the floor up to height ``top``, m^3."""
    zs = np.linspace(container.wall, min(top, container.height), steps)
    areas = [math.pi * inner_radius(container, z) ** 2 for z in zs]
    return float(np.trapezoid(areas, zs))


def add_container(
    spec: mujoco.MjSpec,
    parent: mujoco.MjsBody,
    container: Container,
    pos: list[float],
    name: str | None = None,
    free: bool = True,
) -> mujoco.MjsBody:
    """Add the container standing at ``pos`` (its bottom centre): free, or fixed where it stands."""
    name = name or container.name
    mesh = spec.add_mesh(name=f"{name}_mesh", file=visual_mesh_path(container))
    mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
    body = parent.add_body(name=name, pos=pos)
    if free:
        body.add_freejoint(name=f"{name}_joint")
    body.add_geom(
        name=f"{name}_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=mesh.name,
        rgba=list(container.material.rgba),
        contype=0,
        conaffinity=0,
        group=2,
        mass=0,
    )
    boxes = wall_boxes(container)
    bottom_r = container.outline[0][1]
    # Spread the mass over the walls by volume, so the inertia is a hollow
    # vessel's and not a solid one's.
    volumes = [8 * h[0] * h[1] * h[2] for h, _, _ in boxes]
    bottom_volume = math.pi * bottom_r**2 * container.wall
    density = container.mass / (sum(volumes) + bottom_volume)
    common = dict(
        friction=[container.material.friction, 0.005, 0.0001],
        density=density,
        group=3,
        rgba=[0.5, 0.5, 0.5, 0.3],
    )
    body.add_geom(
        name=f"{name}_bottom",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[bottom_r, container.wall / 2, 0],
        pos=[0, 0, container.wall / 2],
        **common,
    )
    for i, (half, p, q) in enumerate(boxes):
        body.add_geom(
            name=f"{name}_wall{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=half, pos=p, quat=q, **common
        )
    return body
