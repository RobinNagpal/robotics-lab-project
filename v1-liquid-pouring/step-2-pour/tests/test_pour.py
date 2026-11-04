"""The lip goes where it is told, the pour camera sees the receiver, and counts turn into levels."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest
from lab_scene import scene as lab
from lab_scene.settings import IMAGE_SIZE, POUR_CAMERA, RECEIVERS, SOURCES

from pour import pour
from pour.tilt import Tilter, rot_y

POUR = np.array([0.45, 0.0, 0.35])


@pytest.fixture(scope="module")
def scene():
    model, truth = lab.build(0)
    data = mujoco.MjData(model)
    lab.reset(model, data)
    mujoco.mj_forward(model, data)
    return model, data, truth


@pytest.mark.parametrize("container", RECEIVERS)
def test_counts_turn_into_levels_and_back(container):
    counts, levels = pour.level_table(container)
    assert levels[0] == 0 and counts[0] == 0
    assert np.all(np.diff(counts) > 0) and np.all(np.diff(levels) > 0)
    n50 = np.interp(50, levels, counts)
    assert np.interp(n50, counts, levels) == pytest.approx(50, abs=0.5)


def test_the_window_is_2_3_and_4_balls():
    assert [pour.window_balls(c) for c in RECEIVERS] == [2, 3, 4]


def test_the_tipped_tube_puts_its_lip_at_the_point(scene):
    model, data, truth = scene
    source = SOURCES[1]
    tilter = Tilter(model, data, source.name, source.height, source.radius)
    point = np.array([0.46, 0.0, 0.37])
    for tilt in (0.0, 78.0, 90.0, 100.0):
        p, R = tilter.tube_pose(tilt, point)
        assert p + R @ tilter.lip_local == pytest.approx(point)
        # Tipped towards the receiver (-x): the tube's top leans that way.
        assert R[:, 2] @ rot_y(-math.radians(tilt)) @ tilter.R0[:, 2] == pytest.approx(1.0)
    p, R = tilter.tube_pose(90.0, point)
    rim = [
        p + R @ (np.array([math.cos(a), math.sin(a), 0]) * source.radius + [0, 0, source.height])
        for a in np.linspace(0, 2 * math.pi, 36)
    ]
    assert min(r[2] for r in rim) == pytest.approx(point[2], abs=1e-4)  # the lip is the rim's lowest point


def test_the_pour_camera_frames_the_pouring_point(scene):
    model, data, _ = scene
    u, v = pour.Camera.project(data, POUR)
    assert abs(u - IMAGE_SIZE[0] / 2) < 5
    assert 100 < v < 160  # the rim, above the middle: the tube hangs down from it
    bottom = pour.Camera.project(data, POUR - [0, 0, 0.2])
    assert bottom[1] < IMAGE_SIZE[1]
    assert (bottom[1] - v) / 0.2 == pytest.approx(IMAGE_SIZE[1] / 0.3, rel=0.1)  # about 0.6 mm per pixel
    assert model.camera(POUR_CAMERA.name).id >= 0
