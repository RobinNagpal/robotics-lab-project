"""Side grasps: directions follow the picture all the way round, the check sees the wrist behind the
gripper, the arm never folds onto itself, and a clear side grasp lifts a tube."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest
from lab_scene import scene as lab

from grasp import rules, scenes, side


def _arrow(a: float, size: int = 65) -> np.ndarray:
    """A half-line from the middle towards direction ``a``; the picture's up is the table's +y."""
    v, u = np.mgrid[0:size, 0:size].astype(float)
    x, y = u - size // 2, -(v - size // 2)
    along = x * math.cos(a) + y * math.sin(a)
    across = -x * math.sin(a) + y * math.cos(a)
    return (np.abs(across) < 0.7) & (along > 2)


def _direction(picture: np.ndarray) -> float:
    size = picture.shape[0]
    v, u = np.nonzero(picture)
    return math.degrees(math.atan2(-(v - size // 2).mean(), (u - size // 2).mean())) % 360


@pytest.mark.parametrize("k", [0, 5, 13, 22])
@pytest.mark.parametrize("quarter_turns", [1, 2, 3])
def test_turned_picture_turns_the_directions(k, quarter_turns):
    one = np.eye(side.BINS)[k]
    turned = np.rot90(_arrow(side.direction(k)), quarter_turns)
    expected = 15 * int(np.argmax(rules.rotate_bins(one, quarter_turns, step=side.STEP)))
    assert _direction(turned) == pytest.approx(expected, abs=1)


@pytest.mark.parametrize("k", [0, 4, 11, 19])
def test_mirrored_picture_mirrors_the_directions(k):
    one = np.eye(side.BINS)[k]
    mirrored = _arrow(side.direction(k))[:, ::-1]
    assert _direction(mirrored) == pytest.approx(
        15 * int(np.argmax(rules.mirror_bins(one, step=side.STEP))), abs=1
    )


@pytest.fixture(scope="module")
def look():
    model, data, scene = scenes.build(200_000)
    return model, data, scene, side.SideChecker(model, data)


def test_side_grasp_in_empty_space_is_clear_every_way(look):
    *_, checker = look
    away = np.array([0.45, 1.1])  # off the table, nothing near
    assert all(checker.clear(away, side.direction(k), 0.15, "none") for k in range(side.BINS))


def test_the_wrist_behind_the_gripper_is_checked(look):
    model, data, scene, checker = look
    # A tube 25 cm out along the way in: past the fingers and the gripper, where the wrist is.
    tube = scenes.tubes(scene)[0]
    xy, top = scenes.rim(data, tube)
    target = xy + np.array([0.25, 0.0])
    pinch = np.array([target[0], target[1], top - 0.03])
    assert tube.name in checker.touches_side(pinch, math.pi)
    checker.park()


def test_the_arm_never_folds_onto_itself(look):
    model, data, scene, _ = look
    work = mujoco.MjData(model)
    work.qpos[:] = data.qpos
    for t in scenes.tubes(scene)[:4]:
        xy, top = scenes.rim(data, t)
        for k in range(0, side.BINS, 6):
            target = np.array([xy[0], xy[1], side.grip_height(top) + side.DROP]) + side.OUT * side.along(
                side.direction(k)
            )
            arm = min(("arm1", "arm2"), key=lambda a: np.linalg.norm(data.body(f"{a}/base").xpos[:2] - xy))
            try:
                q = side.solve(model, work, target, arm, side.direction(k), elbow_up=True)
            except side.Unreachable:
                continue
            check = mujoco.MjData(model)
            check.qpos[:] = data.qpos
            check.qpos[lab.arm_joint_ids(model, arm)] = q
            mujoco.mj_kinematics(model, check)
            assert not side.folds_onto_itself(model, check, arm)


def test_clear_side_grasp_lifts_the_tube_with_full_physics(look):
    model, data, scene, checker = look
    for tube in scenes.tubes(scene):
        xy, top = scenes.rim(data, tube)
        clear = [k for k in range(side.BINS) if checker.clear(xy, side.direction(k), top, tube.name)]
        if not clear:
            continue
        result = side.run(200_000, tube.name, xy, side.direction(clear[len(clear) // 2]), top)
        if result.get("unreachable"):
            continue
        assert result["success"], result
        return
    pytest.fail("no reachable clear side grasp in the scene")


# ------------------------------------------------------------------ version 2


def test_preshaped_fingers_come_in_just_wider_than_the_tube(look):
    model, data, *_ = look
    checker = side.SideChecker(model, data, preshape=True)
    thickness = checker._pad_centres(1.0) - 2 * rules.TUBE_RADIUS
    gap = checker._pad_centres(checker.way_in_closing) - thickness
    checker.park()
    assert gap == pytest.approx(side.PRESHAPE_GAP, abs=0.0005)
    assert checker._pad_centres(0.0) - thickness > 0.08  # fully open (version 1): about 85 mm


def test_rack_turn_is_read_from_the_close_up(look):
    model, data, scene, _ = look
    camera = scenes.LookCamera(model, data)
    racks = [i.name for i in scene.instances if "rack" in i.name]
    errors = []
    for t in scenes.tubes(scene):
        xy, top = scenes.rim(data, t)
        rack = min(racks, key=lambda n: np.linalg.norm(data.body(n).xpos[:2] - xy))
        r = data.body(rack).xmat
        truth = math.atan2(r[3], r[0])
        found = side.rack_yaw(camera.look(xy, top, side.LOOK_HEIGHT)[0])
        # A quarter turn is the same rack.
        errors.append(abs((found - truth + math.pi / 4) % (math.pi / 2) - math.pi / 4))
    camera.close()
    assert np.median(np.degrees(errors)) < 1.0


def test_a_turned_camera_turns_the_picture(look):
    model, data, scene, _ = look
    camera = scenes.LookCamera(model, data)
    t = scenes.tubes(scene)[0]
    xy, top = scenes.rim(data, t)
    plain, _ = camera.look(xy, top, side.LOOK_HEIGHT)
    turned, calib = camera.look(xy, top, side.LOOK_HEIGHT, yaw=math.pi / 2)
    camera.close()
    # Turning the camera a quarter turn to the left turns what it sees a quarter turn to the right.
    assert np.abs(np.rot90(plain, -1) - turned).mean() < 0.002
    # And the calibration knows: the tube's rim still projects to the middle.
    u, v = calib.project(np.array([[xy[0], xy[1], top]]))[0]
    assert abs(u - 63.5) < 3 and abs(v - 63.5) < 3


def test_smoothing_prefers_the_middle_of_a_run_of_clear_directions():
    scores = np.zeros((side.BINS, 1, 1))
    scores[[3, 4, 5, 6, 7], 0, 0] = 1.0
    scores[3, 0, 0] = 1.01  # the edge of the run scores a hair higher on its own
    assert int(np.argmax(scores[:, 0, 0])) == 3
    assert int(np.argmax(side.smoothed(scores)[:, 0, 0])) in (4, 5, 6)
    # Directions wrap round: 23 is next to 0.
    one = np.zeros((side.BINS, 1, 1))
    one[0, 0, 0] = 1.0
    assert side.smoothed(one)[23, 0, 0] == pytest.approx(0.25)


# ------------------------------------------------------------------ version 3


def test_labels_at_each_grip_height(look):
    _, data, scene, checker = look
    t = scenes.tubes(scene)[0]
    xy, top = scenes.rim(data, t)
    labels = checker.labels(xy, top, t.name, heights=True)
    assert labels.shape == (side.BINS * len(side.HEIGHTS),)
    # Blocked at one height counts as blocked at every lower one.
    rows = labels.reshape(len(side.HEIGHTS), side.BINS)
    assert not (rows[1:] & ~rows[:-1]).any()
    assert (rows[0] == checker.labels(xy, top, t.name)).all()


def test_lower_grip_is_lower_but_never_under_the_rack_limit():
    assert side.grip_height(0.207, 0.05) == pytest.approx(0.157)
    assert side.grip_height(0.107, 0.07) == pytest.approx(side.RACK.height + side.ABOVE_RACK)


def test_each_height_is_smoothed_and_turned_on_its_own():
    scores = np.zeros((side.BINS * 3, 1, 1))
    scores[side.BINS - 1, 0, 0] = 1.0  # the last direction of the first height
    mixed = side.smoothed(scores)[:, 0, 0]
    # Its neighbours are the first height's 22nd and 0th, never the second height's 0th.
    assert mixed[side.BINS - 2] == pytest.approx(0.25) and mixed[0] == pytest.approx(0.25)
    assert mixed[side.BINS] == 0.0


def test_turned_training_picture_keeps_each_height_in_its_block(tmp_path):
    import json

    from grasp.net import Pictures

    tubes = [{"u": 64.0, "v": 64.0, "labels": [0] * (side.BINS * 3)}]
    tubes[0]["labels"][side.BINS + 1] = 1  # second height, direction 1
    (tmp_path / "samples.json").write_text(json.dumps([{"tubes": tubes}]))
    np.save(tmp_path / "heights.npy", np.zeros((1, 128, 128), np.float16))
    pictures = Pictures(tmp_path, augment=True, bins=side.BINS * 3, step=side.STEP)
    for _ in range(10):
        _, target, _ = pictures[0]
        on = np.nonzero(target[:, 64, 64].numpy() > 0.5)[0]
        assert len(on) == 1 and side.BINS <= on[0] < 2 * side.BINS
