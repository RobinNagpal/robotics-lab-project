"""The demo's parts must leave v3's scene, and its physics, exactly as they were."""

import mujoco
import numpy as np
import pytest
from centrifuge_scene import scene
from centrifuge_scene.settings import HOME

import hand
import scene_plus


@pytest.mark.parametrize("seed", [0, 3])
def test_v3_parts_unchanged(seed):
    assert scene_plus.same_as_v3(seed) == []


def _run(model, steps=1500):
    """Swing the arm towards the sample rack and step the physics; returns the final state."""
    data = mujoco.MjData(model)
    scene.reset(model, data)
    target = np.array(HOME) + np.array([0.6, 0.3, -0.4, 0.2, 0.0, 0.5])
    act = scene.arm_actuator_ids(model)
    for i in range(steps):
        data.ctrl[act] = np.array(HOME) + (target - np.array(HOME)) * min(1.0, i / 1000)
        scene.step(model, data)
    return data


def test_same_physics_with_person_parked():
    a = _run(scene._V3_BUILD(0)[0] if hasattr(scene, "_V3_BUILD") else scene_plus._V3_BUILD(0)[0])
    b = _run(scene_plus.build(0)[0])
    assert np.array_equal(a.qpos, b.qpos)


def test_same_physics_with_person_in_the_way():
    """The person's shapes don't collide: the arm passes through them as if they weren't there."""
    model_a, _ = scene_plus._V3_BUILD(0)
    model_b, _ = scene_plus.build(0)
    a = mujoco.MjData(model_a)
    b = mujoco.MjData(model_b)
    for m, d in ((model_a, a), (model_b, b)):
        scene.reset(m, d)
    hand.place(model_b, b, np.array([0.35, 0.10, 0.45]), np.array([1.0, 0, 0, 0]))
    act = scene.arm_actuator_ids(model_a)
    for i in range(1000):
        for m, d in ((model_a, a), (model_b, b)):
            d.ctrl[act] = np.array(HOME) + np.array([0.5, 0.2, 0, 0, 0, 0]) * min(1.0, i / 800)
            scene.step(m, d)
    assert np.allclose(a.qpos, b.qpos, atol=1e-9)


def test_patched_swaps_build_and_puts_it_back():
    v3 = scene.build
    with scene_plus.patched():
        model, _ = scene.build(1)
        assert model.body(hand.BODY).id > 0
    assert scene.build is v3


@pytest.mark.parametrize("seed", [0, 7])
def test_v3_cameras_see_the_same(seed):
    """v3's own cameras render the same pictures and the same depth in our scene (v3's Step 5 reads depth from
    the rotor camera; the person is parked out of sight)."""
    a, _ = scene_plus._V3_BUILD(seed)
    b, _ = scene_plus.build(seed)
    da, db = mujoco.MjData(a), mujoco.MjData(b)
    for m, d in ((a, da), (b, db)):
        scene.reset(m, d)
        mujoco.mj_forward(m, d)
    for camera in ("rotor", "overhead", "racks_high"):
        for depth in (False, True):
            pictures = []
            for m, d in ((a, da), (b, db)):
                with mujoco.Renderer(m, height=240, width=320) as r:
                    if depth:
                        r.enable_depth_rendering()
                    r.update_scene(d, camera=camera)
                    pictures.append(r.render().astype(float))
            assert np.abs(pictures[0] - pictures[1]).max() < (1e-4 if depth else 1.0), (camera, depth)
