"""The scene builds, the liquid stays where it is put, and each arm can lift a full source tube."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from lab_scene import kinematics, liquid, scene
from lab_scene.racks import slot_positions
from lab_scene.settings import LAYOUT, LIQUID, RACK, RECEIVERS, SOURCES


@pytest.fixture(scope="module")
def settled():
    model, truth = scene.build(0)
    data = mujoco.MjData(model)
    scene.reset(model, data)
    scene.settle(model, data, 1.0)
    return model, data, truth


def test_everything_is_in_the_scene(settled):
    model, _, _ = settled
    for name in scene.RACKS:
        assert model.body(name).id > 0
    names = {model.body(i).name for i in range(model.nbody)}
    for gone in ("source_tray", "pouring_stand", "filled_rack", "source_beaker", "source_flask"):
        assert gone not in names
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


def _reaches(model, data, work, arm, target, pointing=kinematics.DOWN):
    q = kinematics.solve(model, data, target, arm, pointing=pointing)
    # Within what the joints can really turn, not just right on paper.
    limits = model.jnt_range[[model.joint(f"{arm}/{j}").id for j in scene.ARM_JOINTS]]
    assert np.all((limits[:, 0] <= q) & (q <= limits[:, 1])), (arm, target)
    work.qpos[:] = data.qpos
    work.qpos[scene.arm_joint_ids(model, arm)] = q
    mujoco.mj_kinematics(model, work)
    site = work.site(scene.pinch_site(arm))
    assert np.linalg.norm(site.xpos - target) < 1e-3, (arm, target)
    assert np.linalg.norm(site.xmat.reshape(3, 3)[:, 2] - pointing) < 1e-2, (arm, target)


def test_both_arms_reach_every_rack_cell(settled):
    model, data, _ = settled
    spots = [p[:2] for rack in scene.RACKS for p in slot_positions(getattr(LAYOUT, rack))]
    work = mujoco.MjData(model)
    for arm in scene.ARM_NAMES:
        for x, y in spots:
            # From just above a cell's floor up to carrying a 200 mm tube clear of a rack.
            for z in (0.05, 0.30):
                _reaches(model, data, work, arm, np.array([x, y, z]))


def test_both_arms_reach_the_pour_point_from_above_and_from_the_side(settled):
    model, data, _ = settled
    work = mujoco.MjData(model)
    point = np.array(LAYOUT.pour_point)
    # Each arm can come down onto the point, or reach in sideways, fingers
    # pointing at the other arm, so the receiver's rim stays open from above.
    for arm, towards_other in (("arm1", [1.0, 0, 0]), ("arm2", [-1.0, 0, 0])):
        for pointing in (kinematics.DOWN, np.array(towards_other)):
            for dz in (-0.10, 0.0, 0.15):
                _reaches(model, data, work, arm, point + [0, 0, dz], pointing)


def test_nothing_stands_at_the_pour_point(settled):
    model, data, _ = settled
    x, y, _ = LAYOUT.pour_point
    for name in (*scene.RACKS, *(c.name for c in SOURCES + RECEIVERS)):
        assert np.hypot(*(data.body(name).xpos[:2] - [x, y])) > 0.15, name


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


def test_seeds_change_the_levels_and_the_cells():
    _, a = scene.build(1)
    _, b = scene.build(2)
    assert a.levels != b.levels
    assert (a.source_slots, a.receiver_slots) != (b.source_slots, b.receiver_slots)
    for t in (a, b):
        assert all(LIQUID.level[0] <= v <= LIQUID.level[1] for v in t.levels.values())


@pytest.mark.parametrize("seed", range(6))
def test_tubes_stand_far_enough_apart_for_the_open_fingers(seed):
    _, truth = scene.build(seed)
    for slots in (truth.source_slots, truth.receiver_slots):
        cells = [divmod(s, RACK.columns) for s in slots.values()]
        for i, (row1, col1) in enumerate(cells):
            for row2, col2 in cells[i + 1 :]:
                if row1 == row2:
                    assert abs(col1 - col2) >= 3, (seed, slots)
                else:
                    assert col1 != col2, (seed, slots)


def test_grip_command_matches_the_measured_finger_gap():
    assert scene.grip_command(0.0852, 0) == pytest.approx(0)
    assert scene.grip_command(0.0251, 0) == pytest.approx(180)
    # Narrower than the tube, but nowhere near fully closed.
    assert 185 < scene.grip_command(0.025) < 200


@pytest.mark.parametrize("arm", ["arm1", "arm2"])
def test_arm_can_grip_and_lift_a_full_source_tube(arm):
    model, truth = scene.build(0)
    data = mujoco.MjData(model)
    scene.reset(model, data)
    name = "source_tube_150"
    tube = model.body(name).id
    c = truth.containers[name]
    before = len(liquid.balls_in(model, data, c, name))
    x, y, bottom = data.xpos[tube]
    gripper = scene.gripper_actuator_id(model, arm)
    # Over the tube at a safe height, down to 3 cm below the rim (but with
    # the pads clear of the rack's top), close to just under its width, lift.
    kinematics.move_to(model, data, np.array([x, y, 0.35]), arm, seconds=1.5)
    rack_top = RACK.height
    kinematics.move_to(model, data, np.array([x, y, max(bottom + c.height - 0.03, rack_top + 0.03)]), arm)
    data.ctrl[gripper] = scene.grip_command(2 * c.radius)
    scene.settle(model, data, 0.8)
    kinematics.move_to(model, data, np.array([x, y, 0.40]), arm)
    assert data.xpos[tube][2] > 0.2
    assert data.xmat[tube][8] > 0.95
    assert len(liquid.balls_in(model, data, c, name)) >= before - 2
