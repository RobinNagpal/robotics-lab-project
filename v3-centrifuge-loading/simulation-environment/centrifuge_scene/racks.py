"""Racks for 50 ml tubes: a floor plate and a top plate with square holes, on four legs.

v1's AutoBio rack has 30 mm cells 36 mm apart: too tight for 50 ml tubes with 35 mm caps. This rack is
built from boxes, and the boxes are both what is drawn and what collides. A conical tube stands with its
tip in a small square seat on the floor plate (as real racks have, so the tip cannot slide); the hole in
the top plate keeps it upright.
"""

from __future__ import annotations

import mujoco

from .settings import RACK, RACK_PLASTIC


def cell_centres(xy: tuple[float, float]) -> list[tuple[float, float]]:
    """The centre of each cell, in the world: row 0 (cells 0-3) nearer the arm, then row 1 (cells 4-7)."""
    x0, y0 = xy
    return [
        (x0 + RACK.pitch * (row - (RACK.rows - 1) / 2), y0 + RACK.pitch * (col - (RACK.columns - 1) / 2))
        for row in range(RACK.rows)
        for col in range(RACK.columns)
    ]


def rack_neighbours(cell: int) -> set[int]:
    """The cells touching ``cell``, diagonals included."""
    row, col = divmod(cell, RACK.columns)
    return {
        r * RACK.columns + c
        for r in range(row - 1, row + 2)
        for c in range(col - 1, col + 2)
        if 0 <= r < RACK.rows and 0 <= c < RACK.columns and (r, c) != (row, col)
    }


def floor_height() -> float:
    """Height of the top of the floor plate: where a tube's tip rests."""
    return RACK.plate


# The seat's inside half width and its height: the 9.6 mm tip goes in, the cone rests on the walls' tops.
SEAT_HALF = 0.0062
SEAT_HEIGHT = 0.004
SEAT_WALL = 0.002


def tip_seat(x: float, y: float, floor: float) -> list[tuple[list[float], list[float]]]:
    """(half sizes, position) of the four walls of a tip seat at (x, y), on a floor at height ``floor``."""
    a, w, h = SEAT_HALF + SEAT_WALL / 2, SEAT_WALL / 2, SEAT_HEIGHT / 2
    z = floor + h
    return [
        ([w, a + w, h], [x + a, y, z]),
        ([w, a + w, h], [x - a, y, z]),
        ([a + w, w, h], [x, y + a, z]),
        ([a + w, w, h], [x, y - a, z]),
    ]


def _boxes() -> list[tuple[list[float], list[float]]]:
    """(half sizes, position) of every box, in the rack's frame (origin on the table, under its middle)."""
    p, h, t = RACK.pitch, RACK.height, RACK.plate
    lx, ly = RACK.rows * p / 2, RACK.columns * p / 2
    bar = (p - RACK.hole) / 2
    boxes = [([lx, ly, t / 2], [0, 0, t / 2])]  # floor plate
    top = h - t / 2
    # Top plate: bars along y at the edges of every row of holes, bars along x at the edges of every column.
    for i in range(RACK.rows + 1):
        x = -lx + i * p
        boxes.append(([bar, ly, t / 2], [x, 0, top]))
    for j in range(RACK.columns + 1):
        y = -ly + j * p
        boxes.append(([lx, bar, t / 2], [0, y, top]))
    # A seat under each hole: four low walls round the tube's tip.
    for x, y in cell_centres((0.0, 0.0)):
        boxes += tip_seat(x, y, t)
    # Four legs at the corners.
    for sx in (-1, 1):
        for sy in (-1, 1):
            boxes.append(([0.004, 0.004, (h - 2 * t) / 2], [sx * (lx - 0.004), sy * (ly - 0.004), h / 2]))
    return boxes


def add_rack(spec: mujoco.MjSpec, name: str, xy: tuple[float, float]) -> mujoco.MjsBody:
    """A free rack on the table, so a careless arm can push it."""
    body = spec.worldbody.add_body(name=name, pos=[xy[0], xy[1], 0])
    body.add_freejoint(name=f"{name}_joint")
    boxes = _boxes()
    volume = sum(8 * s[0] * s[1] * s[2] for s, _ in boxes)
    for i, (half, pos) in enumerate(boxes):
        body.add_geom(
            name=f"{name}_box{i}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=half,
            pos=pos,
            density=RACK.mass / volume,
            friction=[RACK_PLASTIC.friction, 0.005, 0.0001],
            rgba=list(RACK_PLASTIC.rgba),
        )
    return body
