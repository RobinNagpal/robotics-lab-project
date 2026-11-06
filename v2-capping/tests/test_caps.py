"""The screw caps: on tight, off only by turning anticlockwise 1.5 turns, then free.

Most tests use one capped tube held still (as Arm A would hold it) and a "hand" welded to the cap that
turns and rises as a gripper on the arm would: the full scene is only built to check the caps are on
and stay put.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest
from lab_scene import scene as lab
from lab_scene.glassware import add_container
from lab_scene.settings import RECEIVERS

from capping import caps, scene

TUBE = RECEIVERS[1]  # the 150 mm receiver


def one_tube(hand: bool = True) -> tuple[mujoco.MjModel, mujoco.MjData, float]:
    """One capped tube fixed to the floor, with the scene's own solver settings, and a hand on the cap."""
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.001
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.option.solver = mujoco.mjtSolver.mjSOL_CG
    spec.option.iterations = 100
    spec.option.ls_iterations = 10
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.1])
    tube = add_container(spec, spec.worldbody, TUBE, [0, 0, 0], free=False)
    caps.add_cap(spec, tube, TUBE, [0, 0, 0])
    seat = float(spec.body(caps.cap_name(TUBE.name)).pos[2])
    if hand:
        # The hand holds the cap on the thread, and the free cap once it is off (as fingers would).
        spec.worldbody.add_body(name="hand", pos=[0, 0, seat], mocap=True)
        for which, body in (("threaded", caps.follower_name(TUBE.name)), ("free", caps.cap_name(TUBE.name))):
            spec.add_equality(
                name=f"hand_{which}",
                type=mujoco.mjtEq.mjEQ_WELD,
                objtype=mujoco.mjtObj.mjOBJ_BODY,
                name1=body,
                name2="hand",
                data=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1] + [0] * (mujoco.mjNEQDATA - 11),
                solref=list(caps.SOLREF),
                solimp=list(caps.SOLIMP),
                active=which == "threaded",
            )
    model = spec.compile()
    return model, mujoco.MjData(model), seat


def turn(model, data, seat: float, turns_per_s: float, seconds: float) -> float | None:
    """Turn the hand (anticlockwise seen from above for a positive speed), rising with the thread.
    Returns when the cap came off the thread (s), or None."""
    angle, freed = 0.0, None
    for k in range(round(seconds / model.opt.timestep)):
        angle += turns_per_s * 2 * math.pi * model.opt.timestep
        rise = min(max(angle, 0.0), caps.OPEN_ANGLE) * caps.PITCH / (2 * math.pi)
        data.mocap_pos[0] = [0, 0, seat + rise]
        data.mocap_quat[0] = [math.cos(angle / 2), 0, 0, math.sin(angle / 2)]
        was_on = caps.on_thread(model, data, TUBE.name)
        caps.step(model, data)
        if was_on and not caps.on_thread(model, data, TUBE.name) and freed is None:
            freed = (k + 1) * model.opt.timestep
            data.eq_active[model.equality("hand_threaded").id] = 0
            data.eq_active[model.equality("hand_free").id] = 1
    return freed


def unstable(data) -> bool:
    return data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0


def test_the_cap_is_a_24_410_screw_cap():
    assert abs(caps.PITCH - 0.0254 / 8) < 1e-12
    assert caps.TURNS == 1.5
    assert 2 * caps.INNER_R > 0.025  # fits over the 25 mm tube
    assert 0.027 < 2 * (caps.OUTER_R + caps.RIB) < 0.030  # about 28 mm across


def test_pulling_straight_up_does_not_take_the_cap_off():
    model, data, seat = one_tube(hand=False)
    cap = caps.cap_body(model, data, TUBE.name)
    for _ in range(500):
        data.xfrc_applied[cap] = [0, 0, 10.0, 0, 0, 0]
        caps.step(model, data)
    assert caps.on_thread(model, data, TUBE.name)
    assert data.xpos[cap][2] - seat < 0.0015  # 10 N lifts it less than 1.5 mm
    assert not unstable(data)


def test_turning_anticlockwise_takes_it_off_after_one_and_a_half_turns():
    model, data, seat = one_tube()
    freed = turn(model, data, seat, turns_per_s=1.0, seconds=2.0)
    assert freed == pytest.approx(1.5, abs=0.05)
    assert not caps.on_thread(model, data, TUBE.name)
    # (After that the threaded copy is hidden and no longer counts: the free cap is the cap.)
    assert not unstable(data)


def test_turning_clockwise_does_nothing_it_is_already_tight():
    model, data, seat = one_tube()
    assert turn(model, data, seat, turns_per_s=-1.0, seconds=1.0) is None
    assert caps.on_thread(model, data, TUBE.name)
    assert abs(caps.turned(model, data, TUBE.name)) < 0.02  # forced a whole turn the tight way: 7 deg at most


def test_once_off_the_cap_lifts_away_freely():
    model, data, seat = one_tube()
    turn(model, data, seat, turns_per_s=1.0, seconds=1.8)
    z0 = data.mocap_pos[0][2]
    for i in range(500):
        data.mocap_pos[0][2] = z0 + 0.05 * (i + 1) / 500
        caps.step(model, data)
    cap = model.body(caps.cap_name(TUBE.name)).id
    assert data.xpos[cap][2] - seat > 0.05


def test_every_receiver_gets_a_cap_and_v1_is_left_as_it_was():
    plain, _ = lab.build(0)
    assert not caps.tubes_with_caps(plain)
    model, truth = scene.build(0)
    assert sorted(caps.tubes_with_caps(model)) == sorted(c.name for c in RECEIVERS)
    data = mujoco.MjData(model)
    scene.reset(model, data)
    caps.step(model, data, 500)
    for tube in scene.CAPPED:
        assert caps.on_thread(model, data, tube)
        cap, body = caps.cap_body(model, data, tube), model.body(tube).id
        rel = data.xmat[body].reshape(3, 3).T @ (data.xpos[cap] - data.xpos[body])
        assert np.hypot(rel[0], rel[1]) < 0.0005  # still centred on its tube
        assert abs(rel[2] - truth.containers[tube].height) < 0.003  # on the rim
    assert not unstable(data)


def test_off_the_thread_the_free_cap_takes_over_and_reset_puts_it_back():
    model, data, seat = one_tube()
    turn(model, data, seat, turns_per_s=1.0, seconds=1.8)
    free, threaded = caps.cap_name(TUBE.name), caps.follower_name(TUBE.name)
    assert caps.cap_body(model, data, TUBE.name) == model.body(free).id
    # The free cap can be touched and seen now, the threaded one not (both the geom and the body flags).
    assert model.geom_contype[model.geom(f"{free}_shell").id] == 1
    assert model.body_contype[model.body(free).id] == 1
    assert model.geom_contype[model.geom(f"{threaded}_shell").id] == 0
    assert model.geom_rgba[model.geom(f"{threaded}_visual").id, 3] == 0
    mujoco.mj_resetData(model, data)
    caps.reset(model, data)
    assert caps.on_thread(model, data, TUBE.name)
    assert model.geom_contype[model.geom(f"{threaded}_shell").id] == 1
    assert model.body_contype[model.body(free).id] == 0
