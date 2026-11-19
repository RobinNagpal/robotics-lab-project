"""The solver on hand-made weights: each kind of job, and the rules it must keep."""

from __future__ import annotations

import pytest

from solver import (
    EMPTY_TUBE_G,
    MAX_WATER_G,
    TOLERANCE_G,
    imbalance,
    load_order,
    neighbours,
    plan_load,
)


def opposite(a: int, b: int) -> bool:
    return (a - b) % 6 == 3


def check_rules(plan) -> None:
    """Every plan: balanced, each tube has an empty neighbour when it goes in, balance tubes can be filled."""
    assert plan.imbalance_g <= TOLERANCE_G
    assert imbalance({t.hole: t.grams for t in plan.tubes}) == pytest.approx(plan.imbalance_g)
    loaded = set()
    for t in plan.tubes:
        assert sum(n in loaded for n in neighbours(t.hole)) < 2, plan.describe()
        loaded.add(t.hole)
    for t in plan.balance_tubes:
        assert EMPTY_TUBE_G < t.grams < EMPTY_TUBE_G + MAX_WATER_G


def test_two_pairs_go_opposite_each_other_with_no_balance_tube():
    plan = plan_load({"red": 40.10, "yellow": 40.31, "green": 25.02, "purple": 24.90})
    check_rules(plan)
    assert not plan.balance_tubes
    h = {t.name: t.hole for t in plan.tubes}
    assert opposite(h["red"], h["yellow"]) and opposite(h["green"], h["purple"])


def test_the_pairs_are_turned_so_their_small_differences_cancel():
    # Each pair is 0.6 g apart. Side by side the two leftovers would add up to 1.04 g; the solver must lay
    # them out so they cancel instead.
    plan = plan_load({"a": 40.6, "b": 40.0, "c": 30.6, "d": 30.0})
    check_rules(plan)
    assert not plan.balance_tubes
    assert plan.imbalance_g < 0.61


def test_three_equal_samples_go_in_a_triangle():
    plan = plan_load({"a": 35.00, "b": 35.20, "c": 34.90})
    check_rules(plan)
    assert not plan.balance_tubes
    assert sorted(t.hole % 2 for t in plan.tubes) in ([0, 0, 0], [1, 1, 1])  # every other hole


def test_the_odd_one_gets_a_balance_tube_opposite_filled_to_its_weight():
    plan = plan_load({"a": 30.00, "b": 30.20, "odd": 41.70})
    check_rules(plan)
    assert len(plan.balance_tubes) == 1
    h = {t.name: t.hole for t in plan.tubes}
    assert opposite(h["a"], h["b"])
    assert opposite(h["odd"], h["balance A"])
    assert plan.balance_tubes[0].grams == pytest.approx(41.7, abs=0.2)
    # One balance tube cancels only the pull along its own line; the pair's 0.2 g difference points 60 degrees
    # away from it, so 0.2 x sin 60 = 0.17 g is left.
    assert plan.imbalance_g == pytest.approx(0.2 * 3**0.5 / 2, abs=0.01)


def test_two_samples_far_apart_each_get_a_balance_tube():
    plan = plan_load({"light": 22.00, "heavy": 44.00})
    check_rules(plan)
    assert len(plan.balance_tubes) == 2
    assert plan.imbalance_g < 0.01


def test_balance_tubes_go_in_first_with_both_neighbours_empty():
    for weights in ({"light": 22.0, "heavy": 44.0}, {"a": 30.0, "b": 30.2, "odd": 41.7}):
        plan = plan_load(weights)
        assert all(t.balance for t in plan.tubes[: len(plan.balance_tubes)])
        assert not any(t.name.startswith("balance") for t in plan.tubes if t.name in plan.crowded)


def test_a_sample_too_light_for_any_balance_tube_has_no_plan():
    # Lighter than an empty tube: nothing can be filled to match it.
    assert plan_load({"tiny": 5.0}) is None


def test_six_tubes_cannot_be_ordered_but_five_can():
    assert load_order(set(range(6)), set()) is None
    order, _, _ = load_order({0, 1, 2, 3, 4}, set())
    assert sorted(order) == [0, 1, 2, 3, 4]
    assert order[-1] in (0, 4)  # the last one goes in beside the empty hole 5
