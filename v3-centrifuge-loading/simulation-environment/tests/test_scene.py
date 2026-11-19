"""Checks that the scene is built as described, and behaves like the real thing where the task needs it."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from centrifuge_scene import centrifuge, grading, kinematics, liquid, scene
from centrifuge_scene.racks import cell_centres, floor_height, rack_neighbours
from centrifuge_scene.scale import Balance, pan_top
from centrifuge_scene.settings import (
    BALANCE_TOLERANCE,
    LAYOUT,
    LID_CLOSED,
    ROTOR,
    SAMPLES,
    TUBE,
    WATER_CELLS,
    WATER_TUBES,
)


def awake(seed: int, scenario: str | None = None):
    model, truth = scene.build(seed, scenario=scenario, awake=True)
    data = mujoco.MjData(model)
    scene.reset(model, data)
    return model, data, truth


def tilt_deg(data: mujoco.MjData, model: mujoco.MjModel, body: str) -> float:
    return math.degrees(math.acos(min(1.0, data.xmat[model.body(body).id][8])))


# --- the job each seed sets ------------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", scene.SCENARIOS)
def test_each_job_has_the_samples_it_promises(scenario):
    _, truth = scene.build(10, scenario=scenario)
    balls = sorted(s.balls for s in truth.samples.values())
    counts = {"pairs": {2, 4}, "odd_one": {3}, "triangle": {3}, "mismatch": {2}}[scenario]
    assert len(balls) in counts
    assert all(SAMPLES.min_balls <= b <= SAMPLES.max_balls for b in balls)
    if scenario == "triangle":
        assert balls[-1] - balls[0] <= 2 * SAMPLES.pair_spread
    if scenario == "mismatch":
        assert balls[1] - balls[0] >= SAMPLES.odd_gap
    assert set(truth.blanks) == {"blank0", "blank1"}
    assert all(truth.capped[n] for n in truth.samples)
    assert not any(truth.capped[w] for w in WATER_TUBES)
    # Each water tube stands in its own corner, with nothing right beside it (room for a side grip).
    for water, cell in zip(WATER_TUBES, WATER_CELLS, strict=True):
        assert truth.cells[water] == ("supply_rack", cell)
        beside = {n for n in rack_neighbours(cell) if n // 4 == cell // 4 or n % 4 == cell % 4}
        assert not {truth.cells[t][1] for t in [*truth.blanks, *WATER_TUBES] if t != water} & beside


def test_no_job_needs_more_than_four_tubes_in_the_rotor():
    # A tube goes in only with an empty hole beside it (grasp-plan/docs/results.md). With a balance tube for
    # each sample that has no partner, every job fits in 4 tubes (a full rotor of 6 could not be loaded).
    for seed in range(40):
        _, truth = scene.build(seed)
        unpaired = {"pairs": 0, "odd_one": 1, "triangle": 0, "mismatch": 2}[truth.scenario]
        assert len(truth.samples) + unpaired <= 4, seed


def test_the_seed_picks_the_job_in_turn():
    assert [scene.build(s)[1].scenario for s in range(4)] == list(scene.SCENARIOS)


def test_the_built_liquid_matches_the_truth():
    model, truth = scene.build(1)
    data = mujoco.MjData(model)
    scene.reset(model, data)
    for name, sample in truth.samples.items():
        assert len(liquid.balls_in(model, data, TUBE, name)) == sample.balls
    assert len(liquid.balls_in(model, data, TUBE, "water0")) == truth.water_balls
    assert len(liquid.balls_in(model, data, TUBE, "blank0")) == 0


# --- racks and the balance ---------------------------------------------------------------------------


def test_tubes_stand_in_their_racks_and_nothing_spills():
    model, data, truth = awake(1)
    scene.settle(model, data, 1.5)
    for name in truth.tubes:
        assert tilt_deg(data, model, name) < 5.0, name
    assert liquid.spilled(model, data, truth.tubes) == 0


def test_the_balance_reads_each_tube_to_a_hundredth_of_a_gram():
    model, data, truth = awake(3)
    scene.settle(model, data, 0.5)
    balance = Balance(model, data)
    balance.tare()
    assert balance.grams() == 0.0
    on_pan = np.array([LAYOUT.scale[0], LAYOUT.scale[1], pan_top() + 0.0005])
    for name in [*truth.samples, "blank0"]:
        scene.move_tube(model, data, name, on_pan, np.eye(3))
        scene.settle(model, data, 1.5)
        # A real precision balance repeats to about 0.02 g; the simulated one stays within 0.05 g.
        assert balance.grams() == pytest.approx(grading.tube_mass(model, data, truth, name) * 1000, abs=0.05)
        rack, cell = truth.cells[name]
        x, y = cell_centres(getattr(LAYOUT, rack))[cell]
        scene.move_tube(model, data, name, np.array([x, y, floor_height() + 0.0005]), np.eye(3))
        scene.settle(model, data, 0.5)
    assert balance.grams() == pytest.approx(0.0, abs=0.01)


# --- the rotor and lid ------------------------------------------------------------------------------------


def load_rotor(model, data, truth, names):
    for k, name in enumerate(names):
        site = data.site(scene.slot_site(k))
        rot = site.xmat.reshape(3, 3).copy()
        scene.move_tube(model, data, name, site.xpos + rot[:, 2] * 0.0005, rot)


def test_a_capped_tube_seats_in_every_hole_and_the_lid_shuts_over_them():
    model, data, truth = awake(0)  # four samples, and both blanks: a full rotor
    names = [*truth.samples, "blank0", "blank1"]
    load_rotor(model, data, truth, names)
    scene.settle(model, data, 1.5)
    seated = grading.tubes_in_rotor(model, data, truth)
    assert seated == dict(enumerate(names))
    assert liquid.spilled(model, data, truth.tubes) == 0
    centrifuge.close_lid(model, data)
    scene.settle(model, data, 0.3)
    lid = model.body("centrifuge/lid").id
    touching = [
        c
        for c in data.contact[: data.ncon]
        if lid in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2])
    ]
    assert not touching
    assert data.qpos[centrifuge.lid_qpos_adr(model)] == pytest.approx(LID_CLOSED, abs=0.01)


def test_an_open_balance_tube_as_full_as_the_job_needs_does_not_spill_in_the_rotor():
    model, data, truth = awake(2)
    w = model.body("water0").id
    ids = liquid.ball_body_ids(model)
    local = (data.xpos[ids] - data.xpos[w]) @ data.xmat[w].reshape(3, 3)
    inside = sorted(
        (
            b
            for b, p in zip(ids, local, strict=True)
            if liquid.inner_radius(TUBE, p[2]) > 0 and np.hypot(p[0], p[1]) < TUBE.radius
        ),
        key=lambda b: data.xpos[b][2],
    )
    keep = 106  # the most water a balance tube needs (see settings.Samples)
    for k, b in enumerate(inside[keep:]):  # the rest are moved out of the way, onto the table
        adr = model.jnt_qposadr[model.body_jntadr[b]]
        data.qpos[adr : adr + 3] = [-0.5 + 0.008 * (k % 60), -0.9 - 0.008 * (k // 60), 0.004]
    mujoco.mj_forward(model, data)
    load_rotor(model, data, truth, ["water0"])
    scene.settle(model, data, 2.0)
    assert len(liquid.balls_in(model, data, TUBE, "water0")) == keep


def push_lid(model, data, newtons: float, seconds: float) -> bool:
    lid = model.body("centrifuge/lid").id
    hinge = model.joint("centrifuge/lid").id
    for _ in range(int(seconds / model.opt.timestep)):
        if centrifuge.update_latch(model, data):
            break
        data.qfrc_applied[:] = 0
        point = data.site("centrifuge/lid").xpos.copy()
        lever = point - data.xanchor[hinge]
        force = np.cross(data.xaxis[hinge], lever)
        force *= newtons / np.linalg.norm(force)
        if np.cross(lever, force) @ data.xaxis[hinge] > 0:  # push the way that closes it
            force = -force
        mujoco.mj_applyFT(model, data, force, np.zeros(3), point, lid, data.qfrc_applied)
        scene.step(model, data)
    data.qfrc_applied[:] = 0
    return centrifuge.update_latch(model, data)


def test_a_firm_push_on_the_lid_shuts_and_latches_it_and_releasing_lets_it_pop_up():
    model, truth = scene.build(0)
    data = mujoco.MjData(model)
    scene.reset(model, data)
    assert centrifuge.lid_angle(model, data) > 1.6  # starts open
    assert push_lid(model, data, newtons=20, seconds=2.0)
    scene.settle(model, data, 0.5)
    assert centrifuge.lid_angle(model, data) < 0.05
    centrifuge.release_latch(model, data)
    scene.settle(model, data, 2.0)
    # The gas spring lifts it clear of the latch; lifting it the rest of the way is the arm's job.
    assert centrifuge.lid_angle(model, data) > 0.1
    assert not centrifuge.update_latch(model, data)


# --- reach ---------------------------------------------------------------------------------------------


def reaches(model, data, target, pointing) -> bool:
    q = kinematics.solve(model, data, np.asarray(target), pointing=np.asarray(pointing))
    work = mujoco.MjData(model)
    work.qpos[:] = data.qpos
    work.qpos[scene.arm_joint_ids(model)] = q
    mujoco.mj_kinematics(model, work)
    site = work.site(scene.pinch_site())
    along = site.xmat.reshape(3, 3)[:, 2] @ (np.asarray(pointing) / np.linalg.norm(pointing))
    return bool(np.linalg.norm(site.xpos - target) < 1e-3 and along > 0.9998)


def test_the_arm_reaches_every_cell_the_balance_and_every_hole_along_its_axis():
    model, truth = scene.build(0)
    data = mujoco.MjData(model)
    scene.reset(model, data)
    grip = TUBE.capped_height - 0.025  # fingers 25 mm under the top of the cap
    down = [0, 0, -1]
    for rack in scene.RACKS:
        for x, y in cell_centres(getattr(LAYOUT, rack)):
            assert reaches(model, data, [x, y, floor_height() + grip], down)
    assert reaches(model, data, [LAYOUT.scale[0], LAYOUT.scale[1], pan_top() + grip], down)
    for k in range(ROTOR.slots):
        site = data.site(scene.slot_site(k))
        z = site.xmat.reshape(3, 3)[:, 2]
        for back in (0.0, 0.06):  # seated, and 6 cm back along the hole's axis
            assert reaches(model, data, site.xpos + z * (grip + back), -z), k


# --- cameras -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(4))
def test_both_rack_cameras_see_every_tube(seed):
    """Step 0 needs every tube in view: counted from the renderer's picture of which object is at each pixel."""
    model, truth = scene.build(seed)
    data = mujoco.MjData(model)
    scene.reset(model, data)
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        renderer.enable_segmentation_rendering()
        for camera in ("racks_high", "racks_low"):
            renderer.update_scene(data, camera=camera)
            seg = renderer.render()
            geoms = seg[..., 0][seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)]
            bodies = model.geom_bodyid[geoms]
            for name in truth.tubes:
                assert np.sum(bodies == model.body(name).id) > 300, (camera, name)


# --- the balance number ----------------------------------------------------------------------------------


def test_imbalance_is_zero_for_opposite_pairs_and_triangles_and_one_tube_on_its_own():
    assert grading.imbalance({0: 0.040}) == pytest.approx(0.040)
    assert grading.imbalance({0: 0.040, 3: 0.040}) == pytest.approx(0.0, abs=1e-12)
    assert grading.imbalance({0: 0.035, 2: 0.035, 4: 0.035}) == pytest.approx(0.0, abs=1e-12)
    assert grading.imbalance({0: 0.040, 3: 0.039}) == pytest.approx(0.001)
    assert grading.imbalance({1: 0.040, 4: 0.0405}) <= BALANCE_TOLERANCE
