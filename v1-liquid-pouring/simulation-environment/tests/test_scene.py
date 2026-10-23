"""The scene builds, the liquid stays where it is put, and the arm can lift a full beaker."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from lab_scene import kinematics, liquid, scene
from lab_scene.settings import LIQUID, RECEIVERS, SOURCES


@pytest.fixture(scope="module")
def settled():
    model, truth = scene.build(0)
    data = mujoco.MjData(model)
    scene.reset(model, data)
    scene.settle(model, data, 1.0)
    return model, data, truth


def test_everything_is_in_the_scene(settled):
    model, _, _ = settled
    for name in ("source_rack", "empty_rack", "filled_rack", "source_tray", "pouring_stand"):
        assert model.body(name).id > 0
    for c in SOURCES + RECEIVERS:
        assert model.body(c.name).id > 0
    for cam in ("overhead", "front", "wrist"):
        assert model.camera(cam).id >= 0


def test_container_weights_are_real(settled):
    model, _, _ = settled
    for c in SOURCES + RECEIVERS:
        assert model.body(c.name).mass[0] == pytest.approx(c.mass, rel=1e-6)


def test_no_liquid_is_spilled_while_standing(settled):
    model, data, truth = settled
    assert liquid.spilled(model, data, truth.containers) == 0


def test_levels_match_what_was_asked(settled):
    model, data, truth = settled
    for name, asked in truth.levels.items():
        assert liquid.level(model, data, truth.containers[name], name) == pytest.approx(asked, abs=0.1)


def test_receivers_start_empty(settled):
    model, data, truth = settled
    for c in RECEIVERS:
        assert len(liquid.balls_in(model, data, c, c.name)) == 0


def test_everything_stands_upright(settled):
    model, data, truth = settled
    for name in truth.containers:
        # The z axis of the container, in the world: 1 when upright. Tubes
        # lean a few degrees in their 30 mm rack cells.
        assert data.xmat[model.body(name).id][8] > 0.98, name


def test_resting_scene_runs_faster_than_real_time(settled):
    model, data, _ = settled
    # Only the arm is awake once the liquid has settled.
    assert data.nbody_awake < 30


def test_seeds_change_the_levels():
    _, a = scene.build(1)
    _, b = scene.build(2)
    assert a.levels != b.levels
    for t in (a, b):
        assert all(LIQUID.level[0] <= v <= LIQUID.level[1] for v in t.levels.values())


def test_arm_can_grip_and_lift_the_beaker(settled):
    model, truth = scene.build(0)
    data = mujoco.MjData(model)
    scene.reset(model, data)
    beaker = model.body("source_beaker").id
    c = truth.containers["source_beaker"]
    before = len(liquid.balls_in(model, data, c, "source_beaker"))
    x, y = data.xpos[beaker][:2]
    for z, grip in ((0.15, 0), (0.035, 0), (0.035, 255), (0.20, 255)):
        data.ctrl[:6] = kinematics.solve(model, data, np.array([x, y, z]))
        data.ctrl[6] = grip
        scene.settle(model, data, 1.0)
    assert data.xpos[beaker][2] > 0.15
    # Squeezing the beaker can flick out a ball or two from the top.
    assert len(liquid.balls_in(model, data, c, "source_beaker")) >= before - 5
