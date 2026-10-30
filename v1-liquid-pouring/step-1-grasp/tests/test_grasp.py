"""The turn bins follow the picture when it is turned or mirrored, the fast check sees neighbours,
and a clear grasp lifts the tube with full physics."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from lab_scene.settings import RACK

from grasp import pick, rules, scenes, trial
from grasp.net import GraspNet


def _line(yaw: float, size: int = 65) -> np.ndarray:
    """A picture of a line through the middle along ``yaw``, as the close-up camera would see it.

    The camera has the table's x to the right and its y up the picture.
    """
    v, u = np.mgrid[0:size, 0:size].astype(float)
    x, y = u - size // 2, -(v - size // 2)
    return np.abs(-math.sin(yaw) * x + math.cos(yaw) * y) < 0.7


def _angle(picture: np.ndarray) -> float:
    """The direction of the line in a picture, from 0 to 180 degrees on the table."""
    size = picture.shape[0]
    v, u = np.nonzero(picture)
    x, y = u - size // 2, -(v - size // 2)
    w, vecs = np.linalg.eigh(np.cov(np.stack([x, y])))
    d = vecs[:, np.argmax(w)]
    return math.degrees(math.atan2(d[1], d[0])) % 180


@pytest.mark.parametrize("k", [0, 2, 5, 11])
@pytest.mark.parametrize("quarter_turns", [1, 2, 3])
def test_turned_picture_turns_the_bins(k, quarter_turns):
    one = np.eye(rules.BINS)[k]
    turned = _line(rules.bin_yaw(k))
    turned = np.rot90(turned, quarter_turns)
    assert _angle(turned) == pytest.approx(10 * int(np.argmax(rules.rotate_bins(one, quarter_turns))), abs=1)


@pytest.mark.parametrize("k", [0, 3, 9, 14])
def test_mirrored_picture_mirrors_the_bins(k):
    one = np.eye(rules.BINS)[k]
    mirrored = _line(rules.bin_yaw(k))[:, ::-1]
    assert _angle(mirrored) == pytest.approx(10 * int(np.argmax(rules.mirror_bins(one))), abs=1)


def test_grip_height_keeps_the_pads_off_the_rack():
    assert rules.grip_height(0.206) == pytest.approx(0.176)
    # A short tube: 3 cm below its rim would put the pads on the rack's top plate.
    assert rules.grip_height(0.106) == pytest.approx(RACK.height + 0.03)


@pytest.fixture(scope="module")
def look():
    model, data, scene = scenes.build(200_000)
    return model, data, scene, scenes.Checker(model, data)


def test_open_gripper_in_empty_space_is_clear(look):
    _, _, _, checker = look
    assert checker.clear(np.array([0.2, 0.7]), 0.0, 0.15, "none")


def test_a_neighbour_in_the_next_cell_blocks_the_turn_towards_it(look):
    model, data, scene, checker = look
    tubes = scenes.tubes(scene)
    pairs = [
        (a, b)
        for a in tubes
        for b in tubes
        if a.rack == b.rack
        and divmod(a.cell, RACK.columns)[0] == divmod(b.cell, RACK.columns)[0]
        and abs(a.cell - b.cell) == 1
    ]
    assert pairs, "seed has no tubes side by side"
    a, b = pairs[0]
    (xa, top), (xb, _) = scenes.rim(data, a), scenes.rim(data, b)
    towards = math.atan2(*(xb - xa)[::-1]) % math.pi
    assert not checker.clear(xa, towards, top, a.name)


def test_close_up_picture_does_not_show_the_test_gripper(look):
    model, data, scene, checker = look
    tube = scenes.tubes(scene)[0]
    xy, top = scenes.rim(data, tube)
    checker.labels(xy, top, tube.name)
    camera = scenes.LookCamera(model, data)
    heights, _ = camera.look(xy, top)
    camera.close()
    # The tallest tube's rim is at 0.206 m; anything higher is the gripper.
    assert heights.max() < 0.21


def test_closing_fingers_are_tested_too(look):
    _, _, _, checker = look
    assert np.abs(checker.closed).max() > 0.3
    # Closed on nothing in empty space touches nothing.
    assert not checker.touches(np.array([0.2, 0.7, 0.15]), 0.0, 1.0)


def test_network_scores_every_turn_at_every_pixel():
    out = GraspNet()(torch.zeros(2, 1, 128, 128))
    assert out.shape == (2, rules.BINS, 128, 128)


def test_programmed_rule_refuses_a_tube_boxed_in_on_all_sides():
    target = {"name": "t", "xy": [0.0, 0.0], "top": 0.15}
    around = [
        {"name": f"n{i}", "xy": [0.036 * math.cos(a), 0.036 * math.sin(a)], "top": 0.2}
        for i, a in enumerate(np.linspace(0, 2 * math.pi, 8, endpoint=False))
    ]
    assert pick.programmed(target, [target, *around], pick.Footprint()) is None
    assert pick.programmed(target, [target], pick.Footprint()) is not None


def test_clear_grasp_lifts_the_tube_with_full_physics(look):
    model, data, scene, checker = look
    for tube in scenes.tubes(scene):
        xy, top = scenes.rim(data, tube)
        clear = [k for k in range(rules.BINS) if checker.clear(xy, rules.bin_yaw(k), top, tube.name)]
        if clear:
            break
    else:
        pytest.fail("no tube in the scene has a clear turn")
    result = trial.run(200_000, tube.name, xy, rules.bin_yaw(clear[0]), top)
    assert result["success"], result
    assert result["slip"] < 0.002
