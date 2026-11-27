"""The person, the safety camera and the detectors working with the guard (phase 3)."""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
import scene_plus
from centrifuge_scene import scene
from centrifuge_scene.settings import HOME

import guard as G
import sensing
import zones
from detector import Frame
from safety_log import SafetyLog

HERE = Path(__file__).resolve().parent.parent
V1 = HERE / "runs" / "v1" / "weights" / "best.pt"
TEST = HERE / "data" / "hands"


@pytest.fixture(scope="module")
def model():
    m, _ = scene_plus.build(0)
    return m


@pytest.fixture
def data(model):
    d = mujoco.MjData(model)
    scene.reset(model, d)
    return d


@pytest.fixture(autouse=True)
def installed():
    G.install()
    yield
    G.uninstall()


def step(model, data, seconds):
    for _ in range(int(seconds / model.opt.timestep)):
        scene.step(model, data)


def test_person_reaches_in_and_leaves(model, data):
    person = sensing.Person(model, data, np.random.default_rng(0), first=(0.1, 0.2), pause=(0.5, 1.0))
    seen = []
    for _ in range(12000):  # 12 s
        person.before_physics(model, data)
        mujoco.mj_step(model, data)
        seen.append(person.points(model, data) is not None)
    assert any(seen) and not all(seen)  # in sight during reaches, parked between them
    assert len(person.visits) >= 2
    v = person.visits[0]
    assert v.end > v.start and v.reach < zones.WARNING_RADIUS + 0.5


def test_camera_frames_are_noisy_and_carry_the_truth(model, data):
    person = sensing.Person(model, data, np.random.default_rng(1), reaches=False)
    cam = sensing.SafetyCamera(person, np.random.default_rng(2))
    a = cam.capture(model, data, 1)
    mujoco.mj_step(model, data)
    b = cam.capture(model, data, 2)
    cam.close()
    assert a.image.shape == (600, 800, 3) and a.depth.shape == (600, 800)
    assert not np.array_equal(a.image, b.image)  # the watchdog must not see a frozen camera
    assert a.truth["zone"] == "clear" and a.truth["hand"] is None
    # The bench's top is 1.9 m below the camera (things standing on it are nearer), the floor 2.8 m.
    assert 1.0 < float(np.median(a.depth[250:350, 350:450])) < 1.95
    assert float(np.median(a.depth[:20, :20])) == pytest.approx(2.8, abs=0.02)


def test_perfect_detector_uses_the_fixed_zones():
    det = sensing.perfect_detector()
    inside = np.array([[0.5, 0.0, 0.2]])
    warning = np.array([[1.0, 0.0, 0.2]])
    assert det.detect(Frame(1, 0.0, None, {"hand": inside, "arm": None})).zone == "stop"
    assert det.detect(Frame(1, 0.0, None, {"hand": warning, "arm": None})).zone == "warning"
    assert det.detect(Frame(1, 0.0, None, {"hand": None, "arm": None})).zone == "clear"


def test_guard_holds_while_the_person_is_in_the_stop_zone(model, data):
    """A person who only reaches into the stop zone: the arm is held whenever they are there (allowing the
    delay), and moves again once they are gone."""
    person = sensing.Person(model, data, np.random.default_rng(3), first=(0.2, 0.3), pause=(30, 40), goals=("stop",))
    cam = sensing.SafetyCamera(person, np.random.default_rng(4), render=False)
    g = G.attach(G.Guard(model, data, cam, sensing.perfect_detector(), SafetyLog(), limits=G.loose()))
    g.before_physics.append(person.before_physics)
    act = scene.arm_actuator_ids(model)
    goal = np.array(HOME) + np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0])
    for k in range(1, 6001):  # a 6 s v3-style move
        data.ctrl[act] = np.array(HOME) + (goal - np.array(HOME)) * k / 6000
        scene.step(model, data)
    person.finish(float(data.time))
    v = person.visits[0]
    assert v.deepest == "stop" and v.entered_stop is not None
    assert g.held_steps > 0
    assert v.contact_moving == 0
    holds = [e for e in g.log.entries if e.get("action") == G.HOLD and e.get("kind") == "detector"]
    assert holds and holds[0]["t"] <= v.entered_stop + G.FRAME_HZ**-1 + G.LATENCY + 1e-6


@pytest.mark.skipif(not V1.exists() or not (TEST / "meta" / "test.jsonl").exists(), reason="needs v1 and data")
def test_yolo_detector_finds_a_test_hand():
    from PIL import Image

    import dataset

    rows = [json.loads(line) for line in (TEST / "meta" / "test.jsonl").read_text().splitlines()]
    r = next(r for r in rows if r["zone"] == "STOP" and r["hand_pixels"] > 800)
    image = np.asarray(Image.open(TEST / "images" / "test" / f"{r['name']}.jpg").convert("RGB"))
    depth = dataset.load_depth(TEST / "depth" / "test" / f"{r['name']}.png")
    det = sensing.YoloDetector(V1.read_bytes(), "v1", {"threshold": 0.2, "imgsz": 640})
    d = det.detect(Frame(1, 0.0, image, None, depth=depth))
    assert d.zone == "stop" and d.boxes and 0.2 <= d.score <= 1.0
    empty = next(r for r in rows if r["goal"] == "nobody")
    image = np.asarray(Image.open(TEST / "images" / "test" / f"{empty['name']}.jpg").convert("RGB"))
    assert det.detect(Frame(2, 0.1, image, None)).zone == "clear"
