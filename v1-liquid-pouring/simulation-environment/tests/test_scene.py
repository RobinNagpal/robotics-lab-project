"""The scene builds, the liquid stays where it is put, and the arm can lift a full beaker."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from lab_scene import kinematics, liquid, scene
from lab_scene.racks import slot_positions
from lab_scene.settings import LAYOUT, LIQUID, RECEIVERS, SOURCES


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
    for cam in ("overhead", "front", "arm1/wrist", "arm2/wrist"):
        assert model.camera(cam).id >= 0


def test_two_arms_with_their_own_grippers(settled):
    model, data, _ = settled
    assert scene.ARM_NAMES == ("arm1", "arm2")
    ids = [scene.gripper_actuator_id(model, arm) for arm in scene.ARM_NAMES]
    assert len(set(ids)) == 2
    # Both hold their home pose, and face each other across the table.
    for arm in scene.ARM_NAMES:
        assert np.abs(data.qpos[scene.arm_joint_ids(model, arm)] - scene.HOME).max() < 1e-3
    assert data.body("arm1/base").xpos[0] < 0.1 < 0.8 < data.body("arm2/base").xpos[0]


def test_both_arms_reach_every_rack_cell_the_stand_and_the_tray(settled):
    model, data, _ = settled
    spots = [
        p[:2]
        for rack in ("source_rack", "empty_rack", "filled_rack")
        for p in slot_positions(getattr(LAYOUT, rack))
    ]
    spots += [
        LAYOUT.stand,
        (LAYOUT.tray[0] - 0.045, LAYOUT.tray[1]),
        (LAYOUT.tray[0] + 0.045, LAYOUT.tray[1]),
    ]
    work = mujoco.MjData(model)
    for arm in scene.ARM_NAMES:
        for x, y in spots:
            # From just above a cell's floor up to carrying a 200 mm tube clear of a rack.
            for z in (0.05, 0.30):
                target = np.array([x, y, z])
                work.qpos[:] = data.qpos
                q = kinematics.solve(model, data, target, arm)
                # Within what the joints can really turn, not just right on paper.
                limits = model.jnt_range[[model.joint(f"{arm}/{j}").id for j in scene.ARM_JOINTS]]
                assert np.all((limits[:, 0] <= q) & (q <= limits[:, 1])), (arm, x, y, z)
                work.qpos[scene.arm_joint_ids(model, arm)] = q
                mujoco.mj_kinematics(model, work)
                assert np.linalg.norm(work.site(scene.pinch_site(arm)).xpos - target) < 1e-3, (arm, x, y, z)


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
    # Only the two arms are awake once the liquid has settled.
    assert data.nbody_awake < 60


def test_seeds_change_the_levels():
    _, a = scene.build(1)
    _, b = scene.build(2)
    assert a.levels != b.levels
    for t in (a, b):
        assert all(LIQUID.level[0] <= v <= LIQUID.level[1] for v in t.levels.values())


@pytest.mark.parametrize("arm", ["arm1", "arm2"])
def test_arm_can_grip_and_lift_the_beaker(arm):
    model, truth = scene.build(0)
    data = mujoco.MjData(model)
    scene.reset(model, data)
    beaker = model.body("source_beaker").id
    c = truth.containers["source_beaker"]
    before = len(liquid.balls_in(model, data, c, "source_beaker"))
    x, y = data.xpos[beaker][:2]
    gripper = scene.gripper_actuator_id(model, arm)
    # Over the beaker at a safe height, down, close, and lift.
    kinematics.move_to(model, data, np.array([x, y, 0.30]), arm, seconds=2.0)
    kinematics.move_to(model, data, np.array([x, y, 0.035]), arm)
    data.ctrl[gripper] = 255
    scene.settle(model, data, 1.0)
    kinematics.move_to(model, data, np.array([x, y, 0.20]), arm)
    assert data.xpos[beaker][2] > 0.15
    # Squeezing the beaker can flick out a ball or two from the top.
    assert len(liquid.balls_in(model, data, c, "source_beaker")) >= before - 5
