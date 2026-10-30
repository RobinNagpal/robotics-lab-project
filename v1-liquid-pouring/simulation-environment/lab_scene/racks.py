"""The three tube racks: source, empty and destination.

The racks are AutoBio's 10-slot tube rack. Its meshes are drawn;
its collision is the set of boxes AutoBio generated for it (a floor, four
walls and the dividers between the 30 mm cells), copied here.
"""

from __future__ import annotations

import mujoco

from .settings import AUTOBIO, RACK, RACK_PLASTIC

RACK_MESHES = ("lower_plane", "pillars", "upper_plane")

# AutoBio's centrifuge_rack(0.205, 0.096, 0.060, ...), in the rack's frame,
# whose origin is the middle of the rack: (half sizes, position).
RACK_BOXES = [
    ([0.1025, 0.048, 0.003], [0, 0, -0.027]),
    ([0.1025, 0.00175, 0.03], [0, 0.04625, 0]),
    ([0.1025, 0.00175, 0.03], [0, -0.04625, 0]),
    ([0.002, 0.048, 0.03], [0.1005, 0, 0]),
    ([0.002, 0.048, 0.03], [-0.1005, 0, 0]),
]
RACK_BOXES += [
    ([0.003, 0.0095, 0.03], [-0.09 + 0.036 * i, -0.018 + 0.036 * j, 0]) for j in range(2) for i in range(6)
]
RACK_BOXES += [
    ([0.0095, 0.003, 0.03], [-0.072 + 0.036 * i, -0.036 + 0.036 * j, 0]) for j in range(3) for i in range(5)
]
# Top of the rack's floor, above the rack's middle. A tube stands on it.
RACK_FLOOR = -0.024


def slot_positions(rack_xy: tuple[float, float]) -> list[tuple[float, float, float]]:
    """Where a tube's bottom centre rests in each of the 10 cells, in the world."""
    x0, y0 = rack_xy
    floor = RACK.height / 2 + RACK_FLOOR
    return [
        (
            x0 + RACK.pitch * (col - (RACK.columns - 1) / 2),
            y0 + RACK.pitch * (row - (RACK.rows - 1) / 2),
            floor,
        )
        for row in range(RACK.rows)
        for col in range(RACK.columns)
    ]


def add_rack(spec: mujoco.MjSpec, name: str, xy: tuple[float, float]) -> mujoco.MjsBody:
    """A free rack standing on the table, so a careless arm can push it."""
    body = spec.worldbody.add_body(name=name, pos=[xy[0], xy[1], RACK.height / 2])
    body.add_freejoint(name=f"{name}_joint")
    for part in RACK_MESHES:
        mesh_name = f"rack_{part}"
        if spec.mesh(mesh_name) is None:
            spec.add_mesh(
                name=mesh_name,
                file=str(AUTOBIO / "rack" / "centrifuge_10slot_vis" / f"{part}_visual.obj"),
                scale=[0.001, 0.001, 0.001],
            )
        body.add_geom(
            name=f"{name}_{part}",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh_name,
            rgba=list(RACK_PLASTIC.rgba),
            contype=0,
            conaffinity=0,
            group=2,
            mass=0,
        )
    volume = sum(8 * h[0] * h[1] * h[2] for h, _ in RACK_BOXES)
    for i, (half, pos) in enumerate(RACK_BOXES):
        body.add_geom(
            name=f"{name}_box{i}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=half,
            pos=pos,
            density=RACK.mass / volume,
            friction=[RACK_PLASTIC.friction, 0.005, 0.0001],
            group=3,
        )
    return body
