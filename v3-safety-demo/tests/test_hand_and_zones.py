import math

import mujoco
import numpy as np
import pytest
from centrifuge_scene import scene

import hand
import scene_plus
import zones


@pytest.fixture(scope="module")
def world():
    model, _ = scene_plus.build(0)
    data = mujoco.MjData(model)
    scene.reset(model, data)
    return model, data


def test_zone_of_point():
    assert zones.zone_of_point(np.array([0.5, 0.0, 0.2])) == zones.Zone.STOP
    assert zones.zone_of_point(np.array([1.0, 0.0, 0.2])) == zones.Zone.WARNING
    assert zones.zone_of_point(np.array([0.0, 1.3, 0.2])) == zones.Zone.CLEAR
    assert zones.zone_of_point(np.array([0.5, 0.0, 1.5])) == zones.Zone.CLEAR  # above the zones


def test_parked_person_is_clear(world):
    model, data = world
    hand.place(model, data, None, np.array([1.0, 0, 0, 0]))
    mujoco.mj_forward(model, data)
    t = zones.truth(model, data)
    assert t.zone == zones.Zone.CLEAR and not t.visible


def test_fingertips_count(world):
    """A wrist just outside the stop zone, pointing in: the fingers are inside it, so it is a stop."""
    model, data = world
    hand.set_look(model, hand.Look(1.0, "glove_blue", "coat_white"))
    q = hand.Reach(0, (1.5, 0, 0.3), (0.9, 0, 0.3), 1, 0, 1, 0, 0).quat()
    hand.place(model, data, np.array([zones.STOP_RADIUS + 0.05, 0.0, 0.3]), q)
    mujoco.mj_forward(model, data)
    t = zones.truth(model, data)
    assert t.zone == zones.Zone.STOP
    assert zones.STOP_RADIUS - 0.20 < t.reach < zones.STOP_RADIUS


def test_bigger_hand_reaches_further(world):
    model, data = world
    q = hand.Reach(0, (1.5, 0, 0.3), (0.9, 0, 0.3), 1, 0, 1, 0, 0).quat()
    reach = {}
    for size in (0.85, 1.15):
        hand.set_look(model, hand.Look(size, "skin_light", "coat_white"))
        hand.place(model, data, np.array([1.2, 0.0, 0.3]), q)
        mujoco.mj_forward(model, data)
        reach[size] = zones.truth(model, data).reach
    assert reach[1.15] < reach[0.85] - 0.03


def test_reach_timing_and_path():
    r = hand.Reach(1.0, (1.5, 0, 0.3), (0.5, 0, 0.2), 0.5, 1.0, 1.0, 0.1, 0.0)
    assert r.wrist(0.5) is None and r.wrist(r.t_end + 0.1) is None
    assert math.isclose(r.t_arrive, 1.0 + math.hypot(1.0, 0.1) / 0.5)
    assert np.allclose(r.wrist(r.t_arrive + 0.5), (0.5, 0, 0.2))
    assert np.allclose(r.wrist(r.t_end), (1.5, 0, 0.3))


def test_random_reaches(world):
    """Every drawn reach starts clear of the zones, never goes behind the arm, and keeps above the bench."""
    model, data = world
    rng = np.random.default_rng(0)
    hand.set_look(model, hand.Look(1.15, "skin_light", "coat_white"))  # the largest hand
    goals = {}
    for _ in range(60):
        goal, r = hand.draw_reach(rng, model, data, 0.0, 5.0)
        hand.place(model, data, np.array(r.start), r.quat())
        mujoco.mj_forward(model, data)
        assert zones.truth(model, data).zone == zones.Zone.CLEAR
        hand.place(model, data, np.array(r.target), r.quat())
        mujoco.mj_forward(model, data)
        t = zones.truth(model, data)
        goals.setdefault(goal, []).append(t.zone)
        assert r.target[0] >= -0.05
        assert r.target[2] >= hand.surface(model, data, r.target[0], r.target[1]) + hand.CLEARANCE
    assert all(z == zones.Zone.STOP for z in goals["stop"])
    assert all(z == zones.Zone.CLEAR for z in goals["outside"])


def test_distance_to_robot(world):
    model, data = world
    scene.reset(model, data)
    far = zones.distance_to_robot(model, data)
    pinch = data.site_xpos[model.site(scene.pinch_site()).id]
    hand.place(model, data, pinch + np.array([0.05, 0.0, 0.0]), np.array([1.0, 0, 0, 0]))
    mujoco.mj_forward(model, data)
    assert zones.distance_to_robot(model, data) < 0.0 < far
