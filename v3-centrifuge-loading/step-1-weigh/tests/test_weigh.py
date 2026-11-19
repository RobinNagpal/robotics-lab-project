"""Step 1's programmed arm: the geometry helpers, and one whole weighing on seed 1."""

from __future__ import annotations

import math

import numpy as np
import pytest
from centrifuge_scene.racks import cell_centres
from centrifuge_scene.settings import LAYOUT

import test_scenes
import weigh
from motion import Arm, Blocked, fingers_down, turn_between


def test_fingers_down_points_down_and_closes_along_the_turn():
    r = fingers_down(math.radians(30))
    assert np.allclose(r[:, 2], [0, 0, -1])
    assert np.allclose(r[:, 1], [math.cos(math.radians(30)), math.sin(math.radians(30)), 0])
    assert np.isclose(np.linalg.det(r), 1.0)


def test_turns_half_a_circle_apart_are_the_same_grip():
    assert turn_between(0.0, math.pi) == pytest.approx(0.0)
    assert turn_between(0.0, math.radians(170)) == pytest.approx(math.radians(-10))


@pytest.fixture(scope="module")
def seed1():
    return test_scenes.build(1, changed=False)


def test_a_move_that_would_hit_the_rack_is_refused_before_moving(seed1):
    model, data, truth = seed1
    arm = Arm(model, data)
    q = data.qpos[arm.joints].copy()
    rack, cell = truth.cells["sample0"]
    x, y = cell_centres(getattr(LAYOUT, rack))[cell]
    with pytest.raises(Blocked):
        arm.move_line(np.array([x + 0.03, y + 0.03, 0.05]))  # down into the rack's top plate
    assert np.allclose(data.qpos[arm.joints], q)


def test_one_sample_is_weighed_exactly_and_put_back(seed1):
    model, data, truth = seed1
    name = "sample2"
    results, _ = weigh.run(model, data, [truth.cells[name]])
    w = results[0]
    assert w.ok, w.error
    assert w.grams == pytest.approx(truth.samples[name].mass * 1000, abs=0.05)
    rack, cell = truth.cells[name]
    xy = np.array(cell_centres(getattr(LAYOUT, rack))[cell])
    assert np.linalg.norm(data.body(name).xpos[:2] - xy) < 0.003
