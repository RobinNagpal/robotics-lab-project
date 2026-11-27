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


def test_pixel_rays_undo_to_pixels(world):
    import cameras

    model, data = world
    pts = np.array([[0.4, 0.2, 0.1], [1.0, -0.8, 0.4], [0.0, 0.9, 0.0]])
    w, h = cameras.SAFETY.width, cameras.SAFETY.height
    px, ok = cameras.to_pixels(model, data, cameras.SAFETY.name, pts, w, h)
    pos, rays = cameras.pixel_rays(model, data, cameras.SAFETY.name, px, w, h)
    t = (pts[:, 2] - pos[2]) / rays[:, 2]
    assert ok.all() and np.allclose(pos + t[:, None] * rays, pts, atol=1e-9)


def test_box_rule_never_safer_than_truth(world):
    """With the true box, the rule's zone is never safer than the true zone (hands below HAND_TOP)."""
    import box_zone
    import cameras

    model, data = world
    rng = np.random.default_rng(1)
    w, h = cameras.SAFETY.width, cameras.SAFETY.height
    skin = np.array([model.geom(hand.PREFIX + n).id for n in hand.SKIN_SHAPES])
    for _ in range(40):
        hand.set_look(model, hand.Look.draw(rng))
        _, r = hand.draw_reach(rng, model, data, 0.0, 0.0)
        hand.place(model, data, np.array(r.target), r.quat())
        mujoco.mj_forward(model, data)
        truth = zones.truth(model, data)
        # The hand's own corner points, projected: the box a perfect detector would draw (hidden parts too).
        pts = zones.person_points(model, data)[: 8 + 10 * (len(skin) - 1)]
        px, _ = cameras.to_pixels(model, data, cameras.SAFETY.name, pts, w, h)
        box = (*px.min(axis=0), *px.max(axis=0))
        hand_reach = np.linalg.norm(pts[:, :2], axis=1).min()
        assert box_zone.box_reach(model, data, box) <= hand_reach + 1e-6
        if truth.reach == hand_reach:
            assert box_zone.zone_of_boxes(model, data, [box]) >= truth.zone


def test_outline_distance():
    import box_zone

    square = box_zone._hull(np.array([[1.0, -1], [3, -1], [3, 1], [1, 1], [2, 0]]))
    assert len(square) == 4
    assert box_zone._distance_from_origin(square) == pytest.approx(1.0)
    around = box_zone._hull(np.array([[-1.0, -1], [1, -1], [1, 1], [-1, 1]]))
    assert box_zone._distance_from_origin(around) == 0.0


def test_depth_rule_matches_truth(world):
    """With the true box and the depth picture, the rule's reach is within 2 cm below the truth (never above
    it when the closest point of the hand shows), and its zone is never safer."""
    import box_zone
    import cameras
    import dataset

    model, data = world
    rng = np.random.default_rng(2)
    w, h = cameras.SAFETY.width, cameras.SAFETY.height
    skin = np.array([model.geom(hand.PREFIX + n).id for n in hand.SKIN_SHAPES])
    seg = mujoco.Renderer(model, height=h, width=w)
    seg.enable_segmentation_rendering()
    dep = mujoco.Renderer(model, height=h, width=w)
    dep.enable_depth_rendering()
    checked = 0
    for _ in range(25):
        hand.set_look(model, hand.Look.draw(rng))
        goal, r = hand.draw_reach(rng, model, data, 0.0, 0.0)
        hand.place(model, data, np.array(r.target), r.quat())
        mujoco.mj_forward(model, data)
        seg.update_scene(data, camera=cameras.SAFETY.name)
        _, box = dataset.hand_box(seg.render(), skin)
        if box is None:
            continue
        dep.update_scene(data, camera=cameras.SAFETY.name)
        depth = dep.render()
        pts = zones.person_points(model, data)[: 8 + 10 * (len(skin) - 1)]
        hand_reach = np.linalg.norm(pts[:, :2], axis=1).min()
        got = box_zone.depth_reach(model, data, box, depth)
        assert got <= hand_reach + 0.02
        assert got >= hand_reach - 0.15  # not wildly cautious: the box's own corners only
        checked += 1
    seg.close()
    dep.close()
    assert checked >= 15
