"""Perception finds the roles, sizes and liquid; the pair always matches; moves are straight and checked;
and one full run brings both tubes to the pouring point, held from the side, one arm moving at a time."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest
from grasp import scenes, side
from lab_scene import scene as lab

from carry import episode, look, motion
from carry.perceive import Perception, Table, Tube


@pytest.fixture(scope="module")
def scene():
    model, truth = lab.build(0)
    data = mujoco.MjData(model)
    episode.prepare(model, data)
    return model, data, truth


@pytest.fixture(scope="module")
def table(scene):
    model, data, _ = scene
    return Perception().table(model, data)


def test_perception_finds_every_tube_with_its_size_liquid_and_rack(scene, table):
    model, data, truth = scene
    assert set(table.racks) == {"source_rack", "empty_rack", "destination_rack"}
    assert len(table.tubes) == 6
    for t in table.tubes:
        name = min(truth.containers, key=lambda n: np.linalg.norm(data.body(n).xpos[:2] - t.xy))
        assert np.linalg.norm(data.body(name).xpos[:2] - t.xy) < 0.005
        assert t.size == pytest.approx(truth.containers[name].height)
        assert t.liquid == name.startswith("source")
        assert t.rack == ("source_rack" if name.startswith("source") else "empty_rack")


def test_pair_has_one_size_and_falls_back_when_a_tube_is_boxed_in(scene, table):
    model, data, truth = scene
    coordinator = episode.Coordinator(model, data, perception=None, net=episode.load_net())
    coordinator.containers = truth.containers
    choice = coordinator.choose(table, [0.15, 0.2, 0.1])
    assert choice.receiver.size == choice.source.size == 0.15
    assert choice.receiver.rack == "empty_rack" and choice.source.rack == "source_rack"

    # Pretend the medium receiver is boxed in: the next size is taken instead.
    real = coordinator.grasp_of

    def boxed(camera, tube: Tube, arm: str, role: str):
        return None if tube.size == 0.15 and tube.rack == "empty_rack" else real(camera, tube, arm, role)

    coordinator.grasp_of = boxed
    choice = coordinator.choose(table, [0.15, 0.2, 0.1])
    assert choice.receiver.size == choice.source.size == 0.2
    assert choice.tried == [0.15]


def test_no_pair_when_sizes_do_not_match():
    table = Table(
        racks={},
        tubes=[
            Tube(np.zeros(2), 0.106, 0.1, False, "empty_rack"),
            Tube(np.ones(2), 0.206, 0.2, True, "source_rack"),
        ],
    )
    model, truth = lab.build(0)
    coordinator = episode.Coordinator(model, mujoco.MjData(model), perception=None, net=episode.load_net())
    coordinator.containers = truth.containers
    assert coordinator.choose(table, [0.1, 0.15, 0.2]) is None


def test_a_move_is_a_straight_line(scene):
    model, data, _ = scene
    planner = motion.Planner(model)
    start = motion.pinch(data, "arm1")
    end = np.array([0.30, 0.10, 0.35])
    path = planner.solve_line(data, "arm1", start, end, 0.0, math.pi / 2)
    work = mujoco.MjData(model)
    work.qpos[:] = data.qpos
    line = (end - start) / np.linalg.norm(end - start)
    for q in path:
        work.qpos[lab.arm_joint_ids(model, "arm1")] = q
        mujoco.mj_kinematics(model, work)
        p = work.site(lab.pinch_site("arm1")).xpos - start
        assert np.linalg.norm(p - (p @ line) * line) < 0.002


def test_a_move_through_a_rack_is_refused_before_the_arm_moves(scene):
    model, data, _ = scene
    planner = motion.Planner(model)
    rack = data.body("source_rack").xpos
    start = np.array([rack[0], rack[1] - 0.2, 0.04])
    with pytest.raises(motion.Blocked):
        to_start = planner.solve_line(data, "arm2", motion.pinch(data, "arm2"), start, 0.0, 0.0)
        through = planner.solve_line(data, "arm2", start, start + [0, 0.4, 0], 0.0, 0.0)
        planner.check(data, "arm2", to_start + through, None)


def test_full_run_brings_both_tubes_to_the_pouring_point():
    r = episode.run_episode(0, [0.15, 0.2, 0.1], Perception(), episode.load_net())
    assert r["done"], r["failure"]
    assert r["pair_matches"] and r["picked_from_right_racks"]
    assert r["receiver_error_mm"] < 5 and r["source_error_mm"] < 5
    assert max(r["lean_deg"].values()) < 5
    assert r["balls_lost"] == 0
    assert not any(r["touched"].values())
    # One arm moves at a time: Arm A closed on its tube only after Arm B parked at the waiting
    # spot, and Arm B moved in only after Arm A said it was ready.
    at = {msg: t for t, _, msg in r["log"]}
    closed = {arm: t for t, arm, what in r["timeline"] if what == "closing"}
    assert closed[episode.RECEIVER_ARM] > at["waiting"]
    assert at["ready"] <= at["in place"] - episode.SETTLE


def test_close_up_turned_to_the_rack(scene, table):
    """Turned by the rack's angle read from the plain close-up, the rack lines up with the picture."""
    model, data, _ = scene
    camera = look.Camera(model, data)
    tube = table.in_rack("empty_rack")[0]
    height = tube.top + side.LOOK_HEIGHT
    yaw = side.rack_yaw(camera.heights(tube.xy, height)[0])
    turned, calib = camera.heights(tube.xy, height, yaw)
    camera.close()
    assert abs(math.degrees(side.rack_yaw(turned))) < 1.5
    u, v = calib.project(np.array([[*tube.xy, tube.top]]))[0]
    assert abs(u - 63.5) < 3 and abs(v - 63.5) < 3


def test_fingers_come_in_opened_just_wider_than_the_tube():
    model, truth = lab.build(0)
    data = mujoco.MjData(model)
    episode.prepare(model, data)
    closing = episode._preshape_closing(model, data)
    fingers = np.array(
        [model.jnt_qposadr[j] for j in range(model.njnt) if model.joint(j).name.startswith("arm2/gripper/")]
    )
    # The real gripper, told to open 35 mm, is where the planner thinks it is (within a few percent).
    reached = np.linalg.norm(data.qpos[fingers]) / np.linalg.norm(scenes._closed_fingers())
    assert reached == pytest.approx(closing, abs=0.05)
