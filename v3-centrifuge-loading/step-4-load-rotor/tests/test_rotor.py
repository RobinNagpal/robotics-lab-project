"""Step 4: the geometry of holding a tube on a rotor hole's axis (no physics)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from centrifuge_scene.settings import ROTOR, TUBE
from motion import CAP_DEPTH
from moves import TOP_DEPTH

from rotor import BACK, Grip, grip_at_hole


def hole_rot(phi: float) -> np.ndarray:
    """A hole's rotation as centrifuge.slot_frames builds it: z up and in at the rotor's tilt."""
    s, c = math.sin(ROTOR.tilt), math.cos(ROTOR.tilt)
    out = np.array([math.cos(phi), math.sin(phi), 0.0])
    z = np.array([-s * out[0], -s * out[1], c])
    x = np.array([-math.sin(phi), math.cos(phi), 0.0])
    return np.column_stack([x, np.cross(z, x), z])


@pytest.mark.parametrize("capped", [True, False])
def test_the_pinch_point_sits_on_the_axis_its_grip_depth_under_the_tubes_top(capped):
    grip = Grip(capped)
    rot = hole_rot(1.0)
    tip = np.array([0.5, -0.4, 0.09])
    pinch, hand = grip_at_hole(tip, rot, 30, grip)
    top = tip + (TUBE.capped_height if capped else TUBE.height) * rot[:, 2]
    depth = CAP_DEPTH if capped else TOP_DEPTH
    assert np.allclose(pinch, top - depth * rot[:, 2])
    assert np.allclose(hand[:, 2], -rot[:, 2])  # fingers point down the tube
    assert np.allclose(hand.T @ hand, np.eye(3), atol=1e-12)
    assert np.linalg.det(hand) == pytest.approx(1.0)


def test_turn_zero_closes_the_fingers_along_the_rim_and_ninety_across_it():
    rot = hole_rot(0.3)
    _, hand0 = grip_at_hole(np.zeros(3), rot, 0, Grip(True))
    _, hand90 = grip_at_hole(np.zeros(3), rot, 90, Grip(True))
    assert np.allclose(hand0[:, 1], rot[:, 0])  # the pinch site's y axis is the way the fingers close
    assert abs(hand90[:, 1] @ rot[:, 0]) < 1e-12
    assert abs(hand90[:, 1] @ rot[:, 2]) < 1e-12


def test_the_slide_starts_further_back_where_the_arm_can_reach():
    assert set(BACK) == set(range(ROTOR.slots))
    assert all(BACK[k] > ROTOR.depth for k in BACK)  # the tube starts clear of the hole
