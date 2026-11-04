"""The turns and cells the put-back plans with, and how it chooses where each tube goes."""

from __future__ import annotations

import math

import numpy as np
import pytest
from carry.perceive import Table, Tube

from putback import place, workflow
from putback.upright import exp_rot, log_rot, rot_z, upright


@pytest.mark.parametrize(
    "v", [np.array([0.1, -0.2, 0.3]), np.array([0.0, 0.0, 2.5]), np.array([1.0, 0.5, -0.4])]
)
def test_turns_go_there_and_back(v):
    assert np.allclose(log_rot(exp_rot(v)), v, atol=1e-9)


def test_upright_stands_the_tube_up_and_keeps_its_spin():
    leaning = exp_rot(np.array([0.2, -0.1, 0.0])) @ rot_z(0.7)
    R = upright(leaning)
    assert np.allclose(R[:, 2], [0, 0, 1], atol=1e-9)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    # The shortest way up: the turn between them is only the lean.
    assert math.isclose(np.linalg.norm(log_rot(R @ leaning.T)), math.acos(leaning[2, 2]), abs_tol=1e-9)


def test_cells_are_two_rows_of_five_36_mm_apart():
    cells = place.cells(np.array([0.45, 0.22]))
    assert len(cells) == 10
    xs = sorted({round(c[0], 4) for c in cells})
    ys = sorted({round(c[1], 4) for c in cells})
    assert np.allclose(np.diff(xs), 0.036) and np.allclose(np.diff(ys), 0.036)
    assert np.isclose(np.mean(xs), 0.45) and np.isclose(np.mean(ys), 0.22)


def test_turns_start_with_none_and_cover_the_circle():
    assert place.TURNS[0] == 0
    assert sorted({t % 360 for t in place.TURNS}) == list(range(0, 360, 15))


def test_the_source_goes_back_to_its_own_cell_and_the_receiver_near_arm_a():
    src_cells = place.cells(np.array([0.45, 0.22]))
    picked = Tube(src_cells[8][:2].copy(), 0.206, 0.2, True, "source_rack")
    neighbour = Tube(src_cells[1][:2].copy(), 0.156, 0.15, True, "source_rack")
    table = Table(
        racks={"source_rack": np.array([0.45, 0.22]), "destination_rack": np.array([0.45, -0.36])},
        tubes=[picked, neighbour],
    )
    to_source, to_destination = workflow.options(table, picked, np.array([0.9, 0.0, 0.0]))
    assert to_source[0][0] == 8
    assert 1 not in [cell for cell, _, _ in to_source]  # taken by the neighbour
    assert len(to_destination) == 10
    distances = [np.linalg.norm(c[:2] - [0.9, 0.0]) for _, c, _ in to_destination]
    assert distances == sorted(distances)
