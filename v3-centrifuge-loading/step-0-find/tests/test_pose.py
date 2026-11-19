"""pose.py: from points on a surface to a tube's place, direction and size."""

from __future__ import annotations

import math

import numpy as np
import pytest

import pose
from bench import take_pictures
from evaluate_pose import CAMERAS, score, true_tubes, tube_masks


def half_cylinder(base, axis, radius, length, towards, n=2000, seed=0):
    """Points on the half of a cylinder's surface that faces ``towards`` (as a camera would see it)."""
    rng = np.random.default_rng(seed)
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    e1, e2 = pose.basis(axis)
    look = np.asarray(towards, float) - np.asarray(base, float)
    mid = math.atan2(look @ e2, look @ e1)
    angle = mid + rng.uniform(-math.pi / 2, math.pi / 2, n)
    s = rng.uniform(0, length, n)
    return base + np.outer(s, axis) + radius * (np.outer(np.cos(angle), e1) + np.outer(np.sin(angle), e2))


@pytest.mark.parametrize(
    "axis",
    [(0, 0, 1), (1, 0.3, 0), (0.2, 1, 0.05)],
    ids=["standing", "lying", "lying-other-way"],
)
def test_half_a_tube_gives_the_whole_tube(axis):
    base = np.array([0.4, 0.3, 0.004])
    p = pose.fit(half_cylinder(base, axis, 0.0146, 0.11, towards=(1.0, 0.5, 0.7)))
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    assert math.degrees(math.acos(min(1.0, abs(float(p.axis @ a))))) < 0.5
    assert pose.distance_to_line(p.centre, base, a) < 0.0005
    assert p.radius == pytest.approx(0.0146, abs=0.0005)
    assert p.length == pytest.approx(0.11, abs=0.003)


def test_two_views_of_one_tube_join_and_two_tubes_do_not():
    a = pose.fit(half_cylinder([0.4, 0.3, 0.004], (0, 0, 1), 0.0146, 0.11, towards=(1, 0.3, 0.5)), ("high",))
    b = pose.fit(half_cylinder([0.4, 0.3, 0.004], (0, 0, 1), 0.0146, 0.11, towards=(0.4, 1, 0.3)), ("low",))
    c = pose.fit(half_cylinder([0.4, 0.36, 0.004], (0, 0, 1), 0.0146, 0.11, towards=(0.4, 1, 0.3)), ("low",))
    assert pose.same_object(a, b)
    assert not pose.same_object(a, c)  # the next cell, 60 mm away
    assert len(pose.merge({"high": [a], "low": [b, c]})) == 2


def test_perfect_outlines_on_a_real_scene_put_the_tubes_on_their_axes():
    views, (model, data, truth) = take_pictures(0, CAMERAS)
    tubes = true_tubes(model, data, truth)
    masks = tube_masks(model, data, truth, views)
    per_camera = {
        c: pose.poses(views[c], [m for m in masks[c].values() if m.sum() >= 30], c) for c in CAMERAS
    }
    s = score(pose.merge(per_camera), tubes)
    assert s["made_up"] == 0
    assert all(t["found"] and t["stand_or_lie_right"] for t in s["tubes"])
    # One of seed 0's samples stands behind another, and the rack cameras see little more than its cap: its
    # fit tilts about 20 degrees (results.md, section 6). Every other tube is close.
    close = [t for t in s["tubes"] if t["axis_deg"] < 5 and t["place_mm"] < 3]
    assert len(close) >= len(s["tubes"]) - 1
