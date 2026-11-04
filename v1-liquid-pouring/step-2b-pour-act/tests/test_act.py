"""The demonstrations' noise, the examples cut from them, the policies' shapes, and the safety check."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco  # noqa: F401  (before PyTorch: act/__init__.py)
import numpy as np
import pytest
import torch
from pour import start

from act.data import CHUNK, DONE, Demos, Stats, ready
from act.model import make
from act.rollout import MAX_STEP, Guard
from demos.collect import STARTS
from demos.eyes import shrink
from demos.record import Wander


def test_wander_keeps_its_size():
    w = Wander(np.random.default_rng(0), np.array([0.75, 0.0015]))
    values = np.array([w.step(0.01).copy() for _ in range(200_000)])
    assert np.allclose(values.std(0), [0.75, 0.0015], rtol=0.15)
    # It wanders slowly: neighbouring values (0.01 s apart) are nearly the same.
    assert np.corrcoef(values[1:, 0], values[:-1, 0])[0, 1] > 0.95


def test_shrink_averages_blocks():
    image = np.zeros((4, 6, 3), np.uint8)
    image[:2, :2] = 200
    out = shrink(image, 2)
    assert out.shape == (2, 3, 3)
    assert out[0, 0, 0] == 200 and out[1, 1, 0] == 0


def _fake_demo(folder: Path, name: str, n: int, target: float, failure=None) -> None:
    """A demonstration of n ticks whose joint targets rise by 0.01 rad a tick; finished at the last 3."""
    folder.mkdir(parents=True, exist_ok=True)
    intent = np.repeat(np.arange(n, dtype=np.float32)[:, None] * 0.01, 6, axis=1)
    phase = np.full(n, 2)
    phase[-3:] = DONE
    np.savez(
        folder / f"{name}.npz",
        strip=np.zeros((n, 192, 48, 3), np.uint8),
        wide=np.zeros((n, 96, 128, 3), np.uint8),
        wrist=np.zeros((n, 72, 96, 3), np.uint8),
        qpos=intent,
        ctrl=intent,
        intent=intent,
        t=np.arange(n, dtype=np.float32) * 0.1,
        phase=phase,
        receiver=np.zeros(n, int),
        source=np.zeros(n, int),
        air=np.zeros(n, int),
        spilled=np.zeros(n, int),
        level=np.linspace(0, 50, n).astype(np.float32),
        tilt=np.zeros(n, np.float32),
        tilt_cmd=np.zeros(n, np.float32),
        lip=np.zeros((n, 3), np.float32),
    )
    result = {"target_pct": target, "size": 0.15, "noise": 0.0, "failure": failure}
    (folder / f"{name}.json").write_text(json.dumps(result))


@pytest.fixture(scope="module")
def demos(tmp_path_factory) -> Demos:
    folder = tmp_path_factory.mktemp("demos")
    _fake_demo(folder, "a", 25, 45.0)
    _fake_demo(folder, "b", 15, 55.0)
    _fake_demo(folder, "c", 15, 55.0, failure="Arm B cannot reach")
    return Demos(folder)


def test_failed_demonstrations_are_left_out(demos):
    assert demos.names == ["a", "b"] and len(demos) == 40


def test_actions_are_the_next_ticks_held_at_the_end(demos):
    actions = demos.actions(np.array([0, 20]))
    assert actions.shape == (2, CHUNK, 6)
    assert np.allclose(actions[0, :, 0], 0.01 * np.arange(1, CHUNK + 1))
    # Tick 20 of 25: four ticks left, then held at the last one.
    assert np.allclose(actions[1, :, 0], 0.01 * np.array([1, 2, 3, 4, 4, 4, 4, 4, 4, 4]))
    done = demos.done(np.array([20]))[0]
    assert done.tolist() == [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # ticks 22-24 finished, then past the end


def test_examples_do_not_reach_into_the_next_demonstration(demos):
    last_of_a = 24
    assert demos.future(np.array([last_of_a])).max() == last_of_a
    stack = demos.strip_stack(np.array([25]))  # the first tick of b: the picture repeated
    assert stack.shape == (1, 9, 192, 48)


@pytest.mark.parametrize("name", ["act", "mlp"])
def test_policies_give_a_chunk_of_targets(demos, name):
    stats = Stats.of(demos)
    net = make(name)
    b = ready(demos.batch(np.array([0, 5, 30]), stats), "cpu", change=True)
    out = net(b, actions=b["actions"])
    assert out["actions"].shape == (3, CHUNK, 6)
    assert out["done"].shape == (3, CHUNK)
    assert out["extra"].shape == (3, 3)
    net.eval()
    with torch.no_grad():
        assert net(b)["actions"].shape == (3, CHUNK, 6)


def test_the_strip_eye_stays_as_trained():
    net = make("act")
    assert not any(p.requires_grad for p in net.strip.parameters())
    net.train()
    assert not net.strip.net.training  # its batch norms keep the statistics they were trained with


@pytest.fixture(scope="module")
def pouring_point():
    path = next(STARTS.glob("seed150_size*.npy"), None)
    if path is None:
        pytest.skip("no saved starts (step-2-pour: make starts)")
    model, data, _, meta = start.load(path)
    return model, data, meta


def test_guard_lets_small_moves_through_and_limits_big_ones(pouring_point):
    model, data, meta = pouring_point
    guard = Guard(model, meta["source"], meta["receiver"])
    from lab_scene import scene as lab

    now = data.ctrl[lab.arm_actuator_ids(model, "arm1")].copy()
    assert not guard.touches(data, now)
    small = now + 0.001
    assert np.allclose(guard.check(data, now, small), small)
    # Half a radian at the shoulder is cut to MAX_STEP (or, if even that would touch, refused).
    big = guard.check(data, now, now + np.array([0.5, 0, 0, 0, 0, 0]))
    if guard.refused:
        assert np.allclose(big, now)
    else:
        assert np.isclose(np.abs(big - now).max(), MAX_STEP)
