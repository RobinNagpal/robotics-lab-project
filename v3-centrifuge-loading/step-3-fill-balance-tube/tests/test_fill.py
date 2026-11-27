"""Step 3: the geometry of the turned hand and the pour, the second balance tube's new target, and one real
side pick of a water tube on seed 1."""

from __future__ import annotations

import math

import numpy as np
import pytest
import test_scenes
from centrifuge_scene import liquid
from centrifuge_scene.racks import cell_centres
from centrifuge_scene.settings import LAYOUT, TUBE
from motion import Arm
from weigh import tube_at

import evaluate  # Step 3's (it puts step-2-plan on the path for the solver; that folder has its own)
from fill import Filler
from moves import UP, HeldAt, angle_between, rot_about, side_rot, turn_part

solver = evaluate.solver  # Step 2's solver, as evaluate loads it


def test_turning_the_hand_part_way_goes_evenly_from_start_to_end():
    a = np.eye(3)
    b = rot_about(np.array([0.0, 1.0, 0.0]), math.radians(80))
    assert np.allclose(turn_part(a, b, 0.0), a)
    assert np.allclose(turn_part(a, b, 1.0), b)
    assert angle_between(a, turn_part(a, b, 0.25)) == pytest.approx(math.radians(20), abs=1e-6)


def test_a_side_grip_points_the_fingers_along_the_approach_and_closes_across_the_tube():
    approach = np.array([0.6, 0.8, 0.0])
    r = side_rot(approach)
    assert np.allclose(r[:, 2], approach)  # fingers
    assert abs(r[:, 1] @ UP) < 1e-9 and abs(r[:, 1] @ approach) < 1e-9  # the way they close: level, across
    assert np.isclose(np.linalg.det(r), 1.0)


def test_a_held_tube_and_the_hand_that_holds_it_agree_both_ways():
    r = side_rot(np.array([1.0, 0.0, 0.0]))
    held = HeldAt.planned(0, np.array([0.1, 0.2, 0.09]), r, np.array([0.1, 0.2, 0.005]), np.eye(3))
    turn = rot_about(np.array([0.0, 1.0, 0.0]), math.radians(70))
    tip, tube_rot = held.tube_pose(*held.pinch_for(np.array([0.4, 0.0, 0.2]), turn))
    assert np.allclose(tip, [0.4, 0.0, 0.2]) and np.allclose(tube_rot, turn)


@pytest.fixture(scope="module")
def seed1():
    model, data, truth = test_scenes.build(1, changed=False)
    return model, data, truth, Filler(Arm(model, data))


def test_the_pour_pose_puts_the_lip_just_above_and_short_of_the_balance_tubes_middle(seed1):
    model, data, truth, filler = seed1
    r = side_rot(np.array([1.0, 0.0, 0.0]))
    filler.arm.held = HeldAt.planned(0, np.array([0, 0, TUBE.height - 0.04]), r, np.zeros(3), np.eye(3))
    d, u = filler.pour_dir, np.cross(UP, filler.pour_dir)
    for tilt in (60.0, 90.0, 120.0):
        tip, tube_rot = filler.arm.held.tube_pose(*filler.pour_pinch(tilt))
        tipped = rot_about(u, math.radians(tilt))
        lip = tip + TUBE.height * (tipped @ UP) + TUBE.outline[-1][1] * (tipped @ d)
        rim = filler.rim()
        assert lip[2] - rim[2] == pytest.approx(0.020, abs=1e-9)
        assert (lip - rim) @ d == pytest.approx(-0.004, abs=1e-9)
        assert math.degrees(math.acos(tube_rot[2, 2])) == pytest.approx(tilt, abs=1e-6)
    filler.arm.held = None


def test_the_second_balance_tube_makes_up_for_the_first_ones_error():
    weights = {"sample0": 25.35, "sample1": 41.93}  # seed 3: two balance tubes
    plan = solver.plan_load(weights)
    first, second = plan.balance_tubes
    off = 0.30  # the first came out 0.3 g heavy
    new = evaluate.second_target(plan, weights, first.name, first.grams + off)
    masses = {t.hole: weights[t.name] for t in plan.tubes if not t.balance}
    masses |= {first.hole: first.grams + off}
    before = solver.imbalance(masses | {second.hole: second.grams})
    after = solver.imbalance(masses | {second.hole: new})
    assert after < before and after < 0.3


@pytest.mark.slow
def test_the_water_tube_is_picked_from_the_side_and_lifted_out_with_all_its_water(seed1):
    model, data, truth, filler = seed1
    xy = np.array(cell_centres(LAYOUT.supply_rack)[truth.cells["water0"][1]])
    before = len(liquid.balls_in(model, data, TUBE, "water0"))
    filler.pick_side(xy, tube_at(filler.arm, xy))
    filler.arm.wait(0.5)
    tube = model.body("water0").id
    assert data.xpos[tube][2] > 0.09  # lifted 10 cm, clear of the rack (6 cm tall)
    assert len(liquid.balls_in(model, data, TUBE, "water0")) == before
    # The opening is free: the fingers are below the rim.
    assert filler.arm.pinch()[2] < data.xpos[tube][2] + TUBE.height - 0.03
