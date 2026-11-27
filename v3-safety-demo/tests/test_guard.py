"""The guard on v3's real scene, with a small stand-in "script" that moves the arm the way v3's steps do:
write joint targets into data.ctrl, then call scene.step through the module.
"""

import math

import mujoco
import numpy as np
import pytest
from centrifuge_scene import scene
from centrifuge_scene.settings import HOME

import guard as G
from detector import CLEAR, WARNING, Detection, Frame, PerfectDetector, TruthCamera
from safety_log import SafetyLog, verify
from standin import ScriptedHand, Visit

LOOSE = G.loose()
GOAL = np.array(HOME) + np.array([0.4, 0.0, 0.0, 0.0, 0.0, 0.3])  # turn at the shoulder and the wrist


@pytest.fixture(scope="module")
def model():
    m, _ = scene.build(0)
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


def make_guard(model, data, hand=None, camera=None, detector=None, limits=LOOSE, **kw):
    g = G.Guard(
        model,
        data,
        camera or TruthCamera(hand),
        detector or PerfectDetector(),
        SafetyLog(),
        limits=limits,
        **{"latency": 0.0, **kw},
    )
    return G.attach(g)


def move(model, data, goal, seconds=1.0):
    """v3-style: slide the six joint targets to ``goal``, one target per scene.step call."""
    act = scene.arm_actuator_ids(model)
    start = data.ctrl[act].copy()
    n = int(seconds / model.opt.timestep)
    for k in range(1, n + 1):
        data.ctrl[act] = start + (goal - start) * k / n
        scene.step(model, data)  # looked up through the module, as v3 does
    return n


def joints(model, data):
    return data.qpos[scene.arm_joint_ids(model)].copy()


# --- the wrapper itself -------------------------------------------------------------------------------------


def test_nothing_near_is_exactly_v3(model):
    plain = mujoco.MjData(model)
    scene.reset(model, plain)
    move(model, plain, GOAL)
    scene.step(model, plain, 200)
    guarded = mujoco.MjData(model)
    scene.reset(model, guarded)
    g = make_guard(model, guarded)
    move(model, guarded, GOAL)
    scene.step(model, guarded, 200)
    assert np.array_equal(plain.qpos, guarded.qpos) and np.array_equal(plain.qvel, guarded.qvel)
    assert g.physics_steps == g.ticks and g.held_steps == 0


def test_data_not_attached_passes_through(model, data):
    other = mujoco.MjData(model)
    scene.reset(model, other)
    make_guard(model, data)
    scene.step(model, other, 10)
    assert other.time == pytest.approx(10 * model.opt.timestep)


def test_v3_reads_back_its_own_targets(model, data):
    hand = ScriptedHand([Visit(0.0, 100.0, 0.27)])  # warning all along: every tick is spread out
    make_guard(model, data, hand)
    act = scene.arm_actuator_ids(model)
    target = data.ctrl[act] + 0.001
    data.ctrl[act] = target
    scene.step(model, data)
    assert np.array_equal(data.ctrl[act], target)


def test_fk_matches_the_scene(model, data):
    fk = G.ArmFK()
    data.qpos[scene.arm_joint_ids(model)] = GOAL
    mujoco.mj_kinematics(model, data)
    pts = fk.points(GOAL)
    assert np.allclose(pts[0], data.site_xpos[model.site(scene.pinch_site()).id], atol=1e-9)
    assert np.allclose(pts[1], data.xpos[model.body("arm/forearm_link").id], atol=1e-9)


# --- behaviour: slow, hold, carry on -----------------------------------------------------------------------


def test_warning_zone_runs_at_a_quarter_speed(model, data):
    g = make_guard(model, data, ScriptedHand([Visit(0.0, 100.0, 0.27)]))
    n = move(model, data, GOAL, 0.5)
    assert g.physics_steps == pytest.approx(4 * n, abs=2)
    assert np.allclose(data.ctrl[scene.arm_actuator_ids(model)], GOAL)
    assert {e["zone"] for e in g.log.entries if e.get("kind") == "detector"} == {WARNING}


def test_stop_zone_holds_then_carries_on_after_one_second(model, data):
    g = make_guard(model, data, ScriptedHand([Visit(0.25, 0.75, 0.05)]), trace=True)  # frames 0.3 to 0.7
    move(model, data, GOAL, 1.5)
    hold = [e for e in g.log.entries if e.get("action") == G.HOLD]
    assert hold[0]["t"] == pytest.approx(0.3, abs=0.001)
    # The last stop-zone frame is at 0.7 s: the arm carries on at the first frame 1 s after it.
    back = next(e for e in g.log.entries if e.get("t", 0) > 0.8 and e.get("action") == G.RUN)
    assert back["t"] == pytest.approx(1.7, abs=0.002)
    assert g.held_steps * model.opt.timestep == pytest.approx(1.4, abs=0.01)
    # While held, the arm came to rest, and only after being told.
    stop = g.stops[0]
    assert stop.seconds is not None and stop.seconds < 0.05 and stop.distance < 0.005
    assert np.allclose(data.ctrl[scene.arm_actuator_ids(model)], GOAL)
    assert G.replay(g.log.entries) == []
    assert verify(g.log.entries, head=g.log.head) == []


def test_held_call_returns_only_when_the_arm_may_move(model, data):
    g = make_guard(model, data, ScriptedHand([Visit(0.0, 0.45, 0.05)]))  # stop-zone frames 0.0 to 0.4
    before = joints(model, data)
    data.ctrl[scene.arm_actuator_ids(model)] = GOAL  # a big jump in one tick, held back
    held = []
    g.before_physics.append(lambda m, d: held.append(joints(m, d)) if g.holding else None)
    scene.step(model, data)  # one tick of v3's script
    assert g.ticks == 1
    assert g.held_steps == pytest.approx(1400, abs=2)  # held until 1 s after the last stop-zone frame (0.4 s)
    # The arm stayed put while held. The hold came before the tick began, so the arm eased back to where it
    # stood (the shortest ease, 50 ms), then the tick itself ran.
    assert np.abs(np.array(held) - before).max() < 1e-3
    assert g.catch_up_steps * model.opt.timestep == pytest.approx(0.05, abs=0.002)
    assert np.array_equal(g.applied[:6], GOAL)


def test_warning_after_stop_steps_down_to_slow(model, data):
    g = make_guard(model, data, ScriptedHand([Visit(0.0, 0.25, 0.05), Visit(0.25, 3.05, 0.27)]))
    move(model, data, GOAL, 0.5)
    actions = [e["action"] for e in g.log.entries if e.get("kind") == "detector"]
    assert actions[0] == G.HOLD and G.SLOWED in actions and G.RUN not in actions


# --- the watchdog
# --------------------------------------------------------------------------------------------


class Faulty:
    """A camera that goes wrong from frame ``start`` on, in one of several ways."""

    def __init__(self, how, start=5, stop=10**9):
        self.how, self.start, self.stop = how, start, stop
        self.inner = TruthCamera(None)
        self.frames = []
        self.rng = np.random.default_rng(0)

    def capture(self, model, data, number):
        f = self.inner.capture(model, data, number)
        img = self.rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)  # sensor noise: every picture differs
        f = Frame(f.number, f.time, img, f.truth)
        self.frames.append(f)
        if not self.start <= number < self.stop:
            return f
        match self.how:
            case "missing":
                return None
            case "dead":
                raise OSError("no camera")
            case "late":  # 200 ms delay
                old = [x for x in self.frames if x.time <= data.time - 0.2 + 1e-9]
                return old[-1] if old else None
            case "frozen":
                return self.frames[self.start - 1]
            case "same_picture":
                first = self.frames[self.start - 1]
                return Frame(f.number, f.time, first.image, f.truth)
            case "not_a_frame":
                return "rubbish"
        return f


@pytest.mark.parametrize("how", ["missing", "dead", "late", "frozen", "same_picture", "not_a_frame"])
def test_camera_faults_hold_the_arm(model, data, how):
    cam = Faulty(how, start=5, stop=8)  # frames 5 to 7 go wrong (0.4 to 0.7 s)
    g = make_guard(model, data, camera=cam)
    move(model, data, GOAL, 1.0)
    faults = [e for e in g.log.entries if e.get("zone") == G.FAULT]
    assert faults, how
    first = faults[0]["t"]
    assert first <= 0.6 + 0.002  # caught within two frames
    assert all(e["action"] == G.HOLD for e in faults)
    # The arm was held: no tick of the script ran between the fault and 1 s after the last fault.
    assert g.held_steps * model.opt.timestep >= 1.0
    assert G.replay(g.log.entries) == []
    assert np.allclose(data.ctrl[scene.arm_actuator_ids(model)], GOAL)  # and the job carried on after


def test_dead_camera_from_the_start_never_lets_the_arm_move(model, data):
    cam = Faulty("dead", start=1)
    g = make_guard(model, data, camera=cam)
    before = joints(model, data)
    act = scene.arm_actuator_ids(model)
    data.ctrl[act] = before + 0.05
    g.estop_at = 0.5  # the only way out of the hold here
    with pytest.raises(G.SafetyStop):
        scene.step(model, data)
    assert np.abs(joints(model, data) - before).max() < 1e-3
    assert g.ticks == 1 and g.held_steps >= 499


class BadDetector:
    version = "perfect-1"

    def __init__(self, how, start=0.25, stop=0.55):  # bad answers for frames 0.3, 0.4, 0.5
        self.how, self.start, self.stop = how, start, stop
        self.inner = PerfectDetector()
        self.now = 0.0

    def detect(self, frame):
        if not self.start <= frame.time < self.stop:
            return self.inner.detect(frame)
        match self.how:
            case "crash":
                raise RuntimeError("model blew up")
            case "zone":
                return Detection("banana", 0.5, self.version)
            case "nan":
                return Detection(CLEAR, math.nan, self.version)
            case "score":
                return Detection(CLEAR, 2.0, self.version)
            case "version":
                return Detection(CLEAR, 1.0, "v-unknown")
            case "type":
                return {"zone": CLEAR}
            case "slow":
                self.now += 1.0  # the fake clock: one second per call
                return Detection(CLEAR, 1.0, self.version)


@pytest.mark.parametrize("how", ["crash", "zone", "nan", "score", "version", "type", "slow"])
def test_detector_faults_hold_the_arm(model, data, how):
    det = BadDetector(how)
    clock_calls = []

    def clock():
        clock_calls.append(1)
        return det.now

    g = make_guard(model, data, detector=det, clock=clock)
    move(model, data, GOAL, 1.0)
    faults = [e for e in g.log.entries if e.get("zone") == G.FAULT]
    assert faults and faults[0]["t"] == pytest.approx(0.3, abs=0.002), how
    assert all(e["action"] == G.HOLD for e in faults)
    # Last fault at 0.5 s: the arm carries on at 1.5 s (fault ended at 0.6 s, then 1 s of good frames).
    back = next(e for e in g.log.entries if e.get("t", 0) > 0.6 and e.get("action") == G.RUN)
    assert back["t"] == pytest.approx(1.6, abs=0.002)
    assert G.replay(g.log.entries) == []


# --- the normal layer
# ----------------------------------------------------------------------------------------


def test_too_fast_command_is_stopped(model, data):
    g = make_guard(model, data, limits=G.loose(joint_speed=1.0))
    move(model, data, GOAL, 1.0)  # 0.4 rad in 1 s: fine
    with pytest.raises(G.SafetyStop, match="rad/s"):
        move(model, data, np.array(HOME), 0.2)  # 0.4 rad in 0.2 s: 2 rad/s
    g.hold(1.0)
    assert g.stops[-1].action == G.STOP and g.stops[-1].seconds is not None
    assert any(e.get("kind") == "normal" and e["action"] == G.STOP for e in g.log.entries)
    # Latched: the script can't move the arm again until an operator resets it.
    before = joints(model, data)
    with pytest.raises(G.SafetyStop):
        scene.step(model, data)
    assert np.abs(joints(model, data) - before).max() < 1e-3
    g.reset_stop()  # the operator's reset: the arm may move again, from where it was held
    move(model, data, GOAL, 1.0)
    assert np.allclose(joints(model, data), GOAL, atol=0.02)
    assert G.replay(g.log.entries) == []


def test_tool_speed_limit(model, data):
    make_guard(model, data, limits=G.loose(tool_speed=0.05))
    with pytest.raises(G.SafetyStop, match="m/s"):
        move(model, data, GOAL, 0.5)


def test_leaving_the_work_area_is_stopped(model, data):
    fk = G.ArmFK()
    home, goal = fk.points(np.array(HOME))[0], fk.points(GOAL)[0]
    axis = int(np.argmax(np.abs(goal - home)))  # the way the gripper moves most
    lo, hi = [-5.0] * 3, [5.0] * 3
    if goal[axis] > home[axis]:
        hi[axis] = float(home[axis] + goal[axis]) / 2  # the box's edge halfway along the move
    else:
        lo[axis] = float(home[axis] + goal[axis]) / 2
    lim = G.loose(box_min=tuple(lo), box_max=tuple(hi))
    g = make_guard(model, data, limits=lim)
    with pytest.raises(G.SafetyStop, match="work area"):
        move(model, data, GOAL, 1.0)
    g.hold(0.5)
    assert g.stops and g.stops[0].seconds is not None


def test_emergency_stop(model, data):
    g = make_guard(model, data)
    g.estop_at = 0.25
    with pytest.raises(G.SafetyStop, match="emergency"):
        move(model, data, GOAL, 1.0)
    assert data.time == pytest.approx(0.25, abs=0.002)


def test_normal_layer_works_when_the_detector_is_broken(model, data):
    """Separation: with the detector crashing on every frame, the limits and the emergency stop still act."""
    det = BadDetector("crash", start=0.0, stop=1e9)
    g = make_guard(model, data, detector=det, limits=G.loose(joint_speed=0.5))
    act = scene.arm_actuator_ids(model)
    data.ctrl[act] = data.ctrl[act] + 0.1  # a jump: 0.1 rad in one tick
    with pytest.raises(G.SafetyStop, match="rad/s"):
        scene.step(model, data)
    assert g.normal.latched
    assert all(
        e["action"] in (G.HOLD, G.STOP) for e in g.log.entries if "action" in e and e["kind"] != "guard"
    )


def test_hold_mid_move_eases_back_onto_the_path(model, data):
    """A hold in the middle of a move: the arm stops where it is, then eases forward to v3's targets."""
    g = make_guard(model, data, ScriptedHand([Visit(0.45, 0.55, 0.05)]))  # one stop-zone frame at 0.5 s
    move(model, data, GOAL, 1.0)
    assert 0.05 < g.catch_up_steps * model.opt.timestep < 0.5
    assert g.stops[0].seconds < 0.05
    scene.step(model, data, 1000)  # the arm's servo settles slowly, guarded or not
    assert np.allclose(joints(model, data), GOAL, atol=0.002)


def test_each_joint_has_its_own_limit(model, data):
    """The last wrist joint may turn faster than the others (v3 turns it at 3.47 rad/s in Step 3)."""
    lim = G.loose(joint_speed=(1.0, 1.0, 1.0, 1.0, 1.0, 4.0))
    make_guard(model, data, limits=lim)
    wrist = np.array(HOME) + np.array([0, 0, 0, 0, 0, 0.6])
    move(model, data, wrist, 0.3)  # the last joint at 2 rad/s: allowed
    with pytest.raises(G.SafetyStop, match="joint 1 "):
        move(model, data, wrist + np.array([0.6, 0, 0, 0, 0, 0]), 0.3)  # the first joint at 2 rad/s: not


def test_the_elbow_has_its_own_box(model, data):
    elbow = G.ArmFK().points(np.array(HOME))[1]
    lim = G.loose(elbow_min=tuple(elbow - 0.01), elbow_max=tuple(elbow + 0.01))
    make_guard(model, data, limits=lim)
    with pytest.raises(G.SafetyStop, match="elbow"):
        move(model, data, np.array(HOME) + np.array([0, 0.3, 0, 0, 0, 0]), 1.0)


def test_v3_home_is_inside_the_set_limits():
    pts = G.ArmFK().points(np.array(HOME))
    lim = G.Limits()
    assert np.all(pts[0] >= lim.box_min) and np.all(pts[0] <= lim.box_max)
    assert np.all(pts[1] >= lim.elbow_min) and np.all(pts[1] <= lim.elbow_max)


# --- the detection delay, the start-up check, the measured speeds, the boxes in the log --------------------


def test_decisions_take_effect_one_latency_after_the_frame(model, data):
    """With the real 0.1 s delay, a hand in the stop zone from 0.25 s is first seen in the frame taken at
    0.301 s (frames from the first step, every 0.1 s), and that frame's hold takes effect at 0.401 s.
    """
    g = make_guard(model, data, ScriptedHand([Visit(0.25, 0.75, 0.05)]), latency=G.LATENCY)
    move(model, data, GOAL, 1.5)
    hold = [e for e in g.log.entries if e.get("kind") == "detector" and e.get("action") == G.HOLD]
    assert hold[0]["taken"] == pytest.approx(0.301, abs=1e-6)
    assert hold[0]["t"] == pytest.approx(0.401, abs=1e-6)


def test_no_watchdog_faults_with_the_delay(model, data):
    """Decisions keep coming once a frame period apart with the delay too: no false 'no fresh frame'."""
    g = make_guard(model, data, latency=G.LATENCY)
    move(model, data, GOAL, 2.0)
    assert not [e for e in g.log.entries if e.get("zone") == G.FAULT]
    assert g.held_steps == 0 and g.slow_steps == 0


def test_start_up_check_holds_a_hand_already_there(model, data):
    """A hand already in the stop zone when the job starts: the arm never moves until it has been gone 1 s."""
    g = make_guard(model, data, ScriptedHand([Visit(0.0, 0.5, 0.05)]), latency=G.LATENCY)
    start = joints(model, data)
    act = scene.arm_actuator_ids(model)
    data.ctrl[act] = GOAL
    scene.step(model, data)  # one tick of a v3 script asking for a big move
    first = g.log.entries[1]
    assert first["kind"] == "detector" and first["t"] == 0.0 and first["action"] == G.HOLD
    assert data.time > 1.4  # the call returned only after the hand had been gone for 1 s (and the delay)
    # All that time the arm stood still; it has only just started on the tick (one physics step).
    assert g.held_steps > 1400
    assert np.abs(joints(model, data) - start).max() < 0.01


def test_measured_overspeed_latches_a_stop(model, data):
    """The normal layer also watches the arm's measured speed: a joint really turning too fast stops it,
    whatever the commands said.
    """
    g = make_guard(model, data, limits=G.Limits())
    q = data.qpos[g.joints].copy()
    for k in range(20):  # the shoulder turning at 4 rad/s, twice the measured limit (1.6 x 1.25)
        data.qpos[g.joints[0]] = q[0] + 0.004 * k
        g._tool = data.site_xpos[g.site].copy()
        g._check_measured()
    assert g.normal.latched and "measured" in g.normal.latched


def test_hand_boxes_are_logged(model, data):
    class Boxes:
        version = "boxes-1"

        def detect(self, frame):
            return Detection("warning", 0.9, self.version, boxes=((10.0, 20.0, 30.0, 40.0),))

    g = make_guard(model, data, detector=Boxes())
    move(model, data, GOAL, 0.3)
    logged = [e for e in g.log.entries if e.get("kind") == "detector"]
    assert logged and logged[-1]["boxes"] == [[10.0, 20.0, 30.0, 40.0]]
