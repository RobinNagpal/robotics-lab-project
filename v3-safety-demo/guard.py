"""The safety layer around v3's arm: it slows or holds the arm between simulation steps.

**How it reaches the arm without changing v3.** Every v3 step moves the arm the same way: it writes joint
targets
into ``data.ctrl`` and calls ``scene.step(model, data, n)``, looked up through the ``scene`` module.
``install()``
replaces that module attribute with a wrapper. For a ``data`` the wrapper has been told about (``attach``),
each
call becomes one "tick" of v3's script, and the guard decides how the tick is carried out:

- **run**: one physics step with v3's targets, exactly as v3 does it (so with nothing near, the run is the
  same
  to the last bit);
- **slow**: the tick is spread over 4 physics steps (25% speed), the targets moving a quarter of the way each
  step;
- **hold**: the targets are set to where the arm is at that moment, so it stops at once (in about 20 ms in the
  simulation), and physics keeps running; the call does not return until the arm may move again. Then the arm
  eases back onto v3's path (at most 0.5 rad/s) and v3's script carries on exactly where it was paused.

After each call ``data.ctrl`` again holds v3's own targets, so v3's code reads back what it wrote. Any other
``data`` (v3's scratch copies) passes straight through.

**The two layers.**

- The **normal layer** (no ML) checks every arm command (each tick's joint targets): joint speed and gripper
  speed over the last 50 ms within limits, and the gripper and elbow inside the work-area box. Any breach, and
  the
  emergency stop, latch a **stop**: the arm holds and ``SafetyStop`` is raised out of v3's script. It needs an
  operator's reset (``reset_stop``). Nothing from the ML side can change or skip it.
- The **ML layer** reads camera frames (10 a second) and asks the detector for the zone: warning gives slow,
  stop
  zone gives hold, and the arm carries on only after 1 s with no hand (``Behaviour``). The **watchdog** holds
  the
  arm whenever a frame is missing, late (older than 150 ms), frozen (its number or time not moving on, or the
  same
  picture 3 times running), or the detector crashes, gives an answer that makes no sense, or takes too long.
  It
  carries on only after 1 s of good frames with no hand.

What the arm does is the strictest of the two: stop beats hold beats slow beats run.

Every decision goes into the safety log (``safety_log.py``), and ``replay`` rebuilds the decisions from the
log
alone.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np
from centrifuge_scene import scene

from detector import ZONES, Detection, Frame, image_digest

# --- settings ----------------------------------------------------------------------------------------------

SLOW = 0.25  # speed in the warning zone
CLEAR_FOR = 1.0  # s with no hand (and good frames) before the arm carries on
FRAME_HZ = 10.0  # camera frames per simulated second
MAX_FRAME_AGE = 0.15  # s: a newer frame than this must have been checked, or the arm holds
FROZEN_FRAMES = 3  # the same picture this many times running counts as a frozen camera
DETECTOR_BUDGET = 0.25  # s (wall clock) the detector may take per frame
STOPPED = 0.01  # rad/s: the arm counts as stopped when no joint turns faster than this
STOP_WATCH = 3.0  # s: the longest a stop is timed
CATCH_UP_SPEED = 0.5  # rad/s: easing back onto v3's path after a hold (v3's own moves go up to 0.7)

RUN, SLOWED, HOLD, STOP = "run", "slow", "hold", "stop"
SPEED = {RUN: 1.0, SLOWED: SLOW, HOLD: 0.0, STOP: 0.0}
FAULT = "fault"  # the zone logged when there is no trustworthy answer


@dataclass(frozen=True)
class Limits:
    """The normal layer's limits, on the commands v3 sends (speeds measured over ``window`` seconds of
    commands).

    Set from what v3's own 20 jobs use, with room to spare (docs/safety-layer.md, "Setting the limits"):

    - joints 1 to 5 reach 1.25 rad/s at most; the last wrist joint reaches 3.47 rad/s (a quick turn of the
      gripper
      in Step 3, likely the cap's screw-on turns, while the gripper itself hardly moves);
    - the point between the fingers reaches 0.88 m/s (as the arm swings upright going home);
    - the boxes are the space the gripper and the elbow use over all 20 jobs, plus 5 cm.
    """

    joint_speed: float | tuple[float, ...] = (1.6, 1.6, 1.6, 1.6, 1.6, 4.0)  # rad/s, per arm joint
    tool_speed: float = 1.1  # m/s, the point between the fingers
    window: float = 0.05  # s
    # The work-area boxes (m, world frame): for the point between the fingers, and for the elbow.
    box_min: tuple[float, float, float] = (-0.48, -0.74, 0.03)
    box_max: tuple[float, float, float] = (0.75, 0.68, 1.29)
    elbow_min: tuple[float, float, float] = (-0.34, -0.27, 0.38)
    elbow_max: tuple[float, float, float] = (0.38, 0.34, 0.64)


def loose(**changes) -> Limits:
    """Limits wide open, but for ``changes``: for measuring what a job needs, and for tests of one limit."""
    wide = dict(
        joint_speed=100.0,
        tool_speed=100.0,
        box_min=(-9.0,) * 3,
        box_max=(9.0,) * 3,
        elbow_min=(-9.0,) * 3,
        elbow_max=(9.0,) * 3,
    )
    return Limits(**{**wide, **changes})


class SafetyStop(RuntimeError):
    """The normal layer stopped the arm (a limit was broken, or the emergency stop). Needs an operator's
    reset."""


# --- the normal layer (no ML) -----------------------------------------------------------------------------


class ArmFK:
    """Where the gripper's pinch point and the elbow would be for given joint angles.

    Uses a copy of v3's arm on its own (``scene.arm_with_gripper``), which stands at the world's origin
    exactly
    as it does in the scene, so it is quick and never touches the scene's ``data``.
    """

    NAMES = ("tool", "elbow")

    def __init__(self) -> None:
        self.model = scene.arm_with_gripper().compile()
        self.data = mujoco.MjData(self.model)
        self.q = np.array([self.model.joint(n).qposadr[0] for n in scene.ARM_JOINTS])
        self.site = self.model.site(scene.PINCH_SITE).id
        self.elbow = self.model.body("forearm_link").id

    def points(self, q: np.ndarray) -> np.ndarray:
        self.data.qpos[self.q] = q
        mujoco.mj_kinematics(self.model, self.data)
        return np.vstack([self.data.site_xpos[self.site], self.data.xpos[self.elbow]])


class NormalLayer:
    """Speed limits and the work-area box, checked on every arm command, and the latched stop."""

    def __init__(self, limits: Limits, dt: float, q0: np.ndarray, fk: ArmFK | None = None):
        self.limits = limits
        self.fk = fk or ArmFK()
        self.dt = dt
        self.lo = np.array([limits.box_min, limits.elbow_min])  # rows: tool, elbow
        self.hi = np.array([limits.box_max, limits.elbow_max])
        self.joint_limit = np.broadcast_to(np.asarray(limits.joint_speed, dtype=float), (len(q0),))
        n = max(1, int(round(limits.window / dt)))
        p0 = self.fk.points(q0)
        # Filled with where the arm is told to be now: the first command is checked against it.
        self.q_hist = deque([np.array(q0, dtype=float)] * (n + 1), maxlen=n + 1)
        self.p_hist = deque([p0] * (n + 1), maxlen=n + 1)
        self.latched: str | None = None
        # The largest values seen (for setting the limits from v3's own runs).
        self.peak = {"joint_speed": 0.0, "tool_speed": 0.0}
        self.joint_peaks = np.zeros(len(q0))  # per joint, rad/s
        self.peak_where: dict[str, dict] = {}  # for each peak and box edge: the tick, the command, the points
        self.ticks = 0
        self.box_seen = np.array([p0.min(0), p0.max(0)])  # per point kind: [min xyz, max xyz] over tool+elbow
        self.tool_seen = np.array([p0[0], p0[0]])
        self.elbow_seen = np.array([p0[1], p0[1]])

    def reseed(self, q: np.ndarray) -> None:
        """Forget the commands so far: the next is checked against ``q`` (after an operator's reset)."""
        q = np.array(q, dtype=float)
        p = self.fk.points(q)
        for _ in range(self.q_hist.maxlen):
            self.q_hist.append(q)
            self.p_hist.append(p)

    def check(self, q: np.ndarray) -> str | None:
        """What is wrong with the arm command ``q`` (six joint targets), or None."""
        q = np.array(q, dtype=float)
        if not np.all(np.isfinite(q)):
            return "joint targets are not numbers"
        pts = self.p_hist[-1] if np.array_equal(q, self.q_hist[-1]) else self.fk.points(q)
        self.q_hist.append(q)
        self.p_hist.append(pts)
        span = (len(self.q_hist) - 1) * self.dt
        joint = float(np.abs(q - self.q_hist[0]).max()) / span
        tool = float(np.linalg.norm(pts[0] - self.p_hist[0][0])) / span
        self.ticks += 1
        per_joint = np.abs(q - self.q_hist[0]) / span
        self.joint_peaks = np.maximum(self.joint_peaks, per_joint)
        where = None
        if (
            joint > self.peak["joint_speed"]
            or tool > self.peak["tool_speed"]
            or np.any(pts[0] > self.tool_seen[1])
        ):
            where = {
                "tick": self.ticks,
                "q": np.round(q, 4).tolist(),
                "q_window_start": np.round(self.q_hist[0], 4).tolist(),
                "tool": np.round(pts[0], 4).tolist(),
                "elbow": np.round(pts[1], 4).tolist(),
            }
        if joint > self.peak["joint_speed"]:
            self.peak_where["joint_speed"] = where
        if tool > self.peak["tool_speed"]:
            self.peak_where["tool_speed"] = where
        if pts[0][2] > self.tool_seen[1][2]:
            self.peak_where["tool_high"] = where
        self.peak["joint_speed"] = max(self.peak["joint_speed"], joint)
        self.peak["tool_speed"] = max(self.peak["tool_speed"], tool)
        self.tool_seen = np.array(
            [np.minimum(self.tool_seen[0], pts[0]), np.maximum(self.tool_seen[1], pts[0])]
        )
        self.elbow_seen = np.array(
            [np.minimum(self.elbow_seen[0], pts[1]), np.maximum(self.elbow_seen[1], pts[1])]
        )
        for name, p, lo, hi in zip(ArmFK.NAMES, pts, self.lo, self.hi, strict=True):
            if np.any(p < lo) or np.any(p > hi):
                return f"the {name} would leave the work area, at {np.round(p, 3).tolist()} m"
        over = per_joint > self.joint_limit
        if over.any():
            k = int(np.argmax(per_joint - self.joint_limit))
            return f"joint {k + 1} would turn at {per_joint[k]:.2f} rad/s (limit {self.joint_limit[k]:g})"
        if tool > self.limits.tool_speed:
            return f"the gripper would move at {tool:.2f} m/s (limit {self.limits.tool_speed})"
        return None


# --- the ML layer's behaviour ----------------------------------------------------------------------------


class Behaviour:
    """What the arm should do, from the zones seen: hold for 1 s after a stop zone or a fault, slow for 1 s
    after
    a warning, otherwise run. A pure function of (time, zone) pairs, so the log can replay it."""

    def __init__(self, clear_for: float = CLEAR_FOR):
        self.clear_for = clear_for
        self.last_stop = -math.inf
        self.last_warning = -math.inf

    def update(self, t: float, zone: str) -> str:
        if zone in (STOP, FAULT):
            self.last_stop = t
        elif zone == "warning":
            self.last_warning = t
        if t - self.last_stop < self.clear_for - 1e-6:
            return HOLD
        if t - self.last_warning < self.clear_for - 1e-6:
            return SLOWED
        return RUN


# --- the guard ---------------------------------------------------------------------------------------------


@dataclass
class StopEvent:
    """One time the arm was told to hold or stop while moving: how long it took to stop and how far it
    went."""

    t: float  # when the decision was made (simulation s)
    action: str
    reason: str
    seconds: float | None = (
        None  # until no joint turned faster than STOPPED (None: resumed first, or timed out)
    )
    distance: float = 0.0  # m the pinch point travelled meanwhile
    peak_speed: float = 0.0  # m/s of the pinch point at the decision


class Guard:
    """The safety layer for one scene (``model``, ``data``)."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        camera,
        detector,
        log,
        *,
        limits: Limits = Limits(),
        frame_hz: float = FRAME_HZ,
        max_frame_age: float = MAX_FRAME_AGE,
        frozen_frames: int | None = FROZEN_FRAMES,
        detector_budget: float = DETECTOR_BUDGET,
        clock: Callable[[], float] = time.perf_counter,
        trace: bool = False,
    ):
        self.model, self.data = model, data
        self.camera, self.detector, self.log = camera, detector, log
        self.frame_period = 1.0 / frame_hz
        self.max_frame_age = max_frame_age
        self.frozen_frames = frozen_frames
        self.detector_budget = detector_budget
        self.clock = clock
        self.act = np.concatenate([scene.arm_actuator_ids(model), [scene.gripper_actuator_id(model)]])
        self.joints = scene.arm_joint_ids(model)
        self.dofs = np.array([model.joint(f"{scene.ARM_NAME}/{n}").dofadr[0] for n in scene.ARM_JOINTS])
        self.site = model.site(scene.pinch_site()).id
        self.dt = model.opt.timestep
        self.normal = NormalLayer(limits, self.dt, data.ctrl[self.act[:6]].copy())
        self.behaviour = Behaviour()
        self.ml_action = RUN
        self.action = RUN
        self.fault: str | None = None
        self.applied = data.ctrl[self.act].copy()  # what the actuators are being given
        self.last = self.applied.copy()  # v3's targets at the end of the last tick
        self.progress = 0.0
        self.holding = False
        self.resume: tuple[np.ndarray, float] | None = (
            None  # (where the arm stopped, how far back on the path)
        )
        self.catch_up_steps = 0
        self.next_frame = float(data.time)
        self.frame_no = 0
        self.last_frame: tuple[int, float, str | None] | None = None
        self.same_pictures = 0
        self.last_good = -math.inf
        # Hooks: called before every physics step (Track A moves its hand here); test-only command changes.
        self.before_physics: list[Callable[[mujoco.MjModel, mujoco.MjData], None]] = []
        self.inject: Callable[[Guard, np.ndarray], np.ndarray] | None = None
        self.estop_at: float | None = None
        self.ticks = 0
        self.physics_steps = 0
        self.held_steps = 0
        self.slow_steps = 0
        self.stops: list[StopEvent] = []
        self._watch: StopEvent | None = None
        self._tool = data.site_xpos[self.site].copy()
        self.tool_speed = 0.0
        self.trace: list[tuple] | None = [] if trace else None
        self.log.append(kind="guard", t=float(data.time), action=RUN, reason="started", model=self.version)

    @property
    def version(self) -> str:
        return getattr(self.detector, "version", "none")

    # --- v3's ticks
    # ------------------------------------------------------------------------------------------

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            self._tick()

    def _tick(self) -> None:
        d = self.data
        desired = d.ctrl[self.act].copy()
        if self.inject is not None:
            desired = np.array(self.inject(self, desired), dtype=float)
        self.ticks += 1
        if self.normal.latched is None:
            why = self.normal.check(desired[:6])
            if why:
                self._latch(f"normal layer: {why}")
        while True:
            self._sense()
            s = SPEED[self.action]
            if s == 0.0:
                self._hold_step()
                if self.normal.latched:
                    raise SafetyStop(self.normal.latched)
                continue
            if self.resume is not None:
                # Back from a hold: ease from where the arm stopped onto v3's path before v3's script carries
                # on.
                frm, phase = self.resume
                nominal = self.last + (desired - self.last) * self.progress
                seconds = max(0.05, 1.5 * float(np.abs(nominal[:6] - frm[:6]).max()) / CATCH_UP_SPEED)
                phase = min(1.0, phase + s * self.dt / seconds)
                self.applied = frm + (nominal - frm) * _smooth(phase)
                d.ctrl[self.act] = self.applied
                self._physics()
                self.catch_up_steps += 1
                self.resume = None if phase >= 1.0 else (frm, phase)
                continue
            self.progress += s
            if self.progress >= 1.0 - 1e-9:
                self.applied = desired.copy()
                d.ctrl[self.act] = desired
                self._physics()
                self.progress = max(0.0, self.progress - 1.0)
                if s < 1.0:
                    self.slow_steps += 1
                break
            self.applied = self.last + (desired - self.last) * self.progress
            d.ctrl[self.act] = self.applied
            self._physics()
            self.slow_steps += 1
        self.last = desired
        d.ctrl[self.act] = desired  # v3 reads back the targets it wrote

    def _hold_step(self) -> None:
        """One physics step with the arm held where it is (the gripper keeps its last command: a held tube
        stays
        held)."""
        if not self.holding:
            self.holding = True
            self.applied = self.applied.copy()
            self.applied[:6] = self.data.qpos[self.joints]
        self.resume = None
        self.data.ctrl[self.act] = self.applied
        self._physics()
        self.held_steps += 1

    def hold(self, seconds: float) -> None:
        """Keep the simulation running with the arm held (after a SafetyStop, to see it come to rest)."""
        for _ in range(int(round(seconds / self.dt))):
            self._sense()
            self._hold_step()

    def _physics(self) -> None:
        for f in self.before_physics:
            f(self.model, self.data)
        _ORIGINAL["step"](self.model, self.data, 1)
        self.physics_steps += 1
        tool = self.data.site_xpos[self.site]
        moved = float(np.linalg.norm(tool - self._tool))
        self.tool_speed = moved / self.dt
        self._tool = tool.copy()
        w = self._watch
        if w is not None:
            w.distance += moved
            if float(np.abs(self.data.qvel[self.dofs]).max()) < STOPPED:
                w.seconds = float(self.data.time) - w.t
                self._watch = None
            elif self.data.time - w.t > STOP_WATCH:
                self._watch = None

    # --- the operator's buttons ----------------------------------------------------------------------------

    def emergency_stop(self, reason: str = "button pressed") -> None:
        self._latch(f"emergency stop: {reason}")

    def reset_stop(self) -> None:
        if self.normal.latched:
            self.log.append(
                kind="normal", t=float(self.data.time), event="reset", action=RUN, reason="operator reset"
            )
            self.normal.latched = None
            self.normal.reseed(self.applied[:6])
            self.last = self.applied.copy()
            self.progress = 0.0

    def _latch(self, reason: str) -> None:
        if self.normal.latched:
            return
        self.normal.latched = reason
        self.log.append(kind="normal", t=float(self.data.time), event="stop", action=STOP, reason=reason)
        self._set_action(float(self.data.time), reason)

    # --- the camera, the detector and the watchdog --------------------------------------------------------

    def _sense(self) -> None:
        t = float(self.data.time)
        if self.estop_at is not None and t >= self.estop_at:
            self.estop_at = None
            self.emergency_stop("scheduled")
        if t >= self.next_frame - 1e-9:
            self.next_frame += self.frame_period
            if self.next_frame <= t:
                self.next_frame = t + self.frame_period
            self.frame_no += 1
            self._frame(t)
        if self.fault is None and t - self.last_good > self.max_frame_age + 1e-9:
            self._bad(t, None, "watchdog: no fresh camera frame", kind="watchdog")
        self._set_action(t, self.fault or "")

    def _frame(self, t: float) -> None:
        try:
            frame = self.camera.capture(self.model, self.data, self.frame_no)
        except Exception as e:  # noqa: BLE001 - any camera failure is a fault, never a crash of the guard
            return self._bad(t, None, f"camera failed: {type(e).__name__}")
        if frame is None:
            return self._bad(t, None, "frame missing")
        why = self._check_frame(t, frame)
        if why:
            return self._bad(t, frame, why)
        start = self.clock()
        try:
            det = self.detector.detect(frame)
        except Exception as e:  # noqa: BLE001
            return self._bad(t, frame, f"detector crashed: {type(e).__name__}")
        took = self.clock() - start
        why = self._check_detection(det, took)
        if why:
            return self._bad(t, frame, why, det)
        if self.fault:
            # The fault lasted until now: the arm carries on only after CLEAR_FOR of good frames.
            self.behaviour.update(t, FAULT)
            self.log.append(
                kind="watchdog",
                t=t,
                frame=frame.number,
                model=self.version,
                zone=FAULT,
                action=self._effective(HOLD),
                reason=f"fault ended ({self.fault})",
            )
            self.fault = None
        self.last_good = frame.time
        self.ml_action = self.behaviour.update(t, det.zone)
        action = self._effective(self.ml_action)
        self.log.append(
            kind="detector",
            t=t,
            frame=frame.number,
            model=det.model_version,
            score=float(det.score),
            zone=det.zone,
            action=action,
            reason="",
        )
        if self.trace is not None:
            dist = getattr(self.camera, "last_distance", None)
            self.trace.append((round(t, 3), action, det.zone, round(self.tool_speed, 4), dist))

    def _check_frame(self, t: float, frame) -> str | None:
        if not isinstance(frame, Frame):
            return "camera gave something that is not a frame"
        last = self.last_frame
        if last and frame.number <= last[0]:
            return f"frozen camera: frame number {frame.number} did not move on"
        if last and frame.time <= last[1]:
            return "frozen camera: frame time did not move on"
        if frame.time > t + 1e-9:
            return "frame time is in the future"
        age = t - frame.time
        digest = image_digest(frame.image) if frame.image is not None else None
        self.same_pictures = self.same_pictures + 1 if digest and last and digest == last[2] else 0
        self.last_frame = (frame.number, frame.time, digest)
        if age > self.max_frame_age + 1e-9:
            return f"late frame: {age * 1000:.0f} ms old"
        if self.frozen_frames and self.same_pictures >= self.frozen_frames - 1:
            return f"frozen camera: the same picture {self.same_pictures + 1} times running"
        return None

    def _check_detection(self, det, took: float) -> str | None:
        if not isinstance(det, Detection):
            return "detector output makes no sense: not a detection"
        if det.zone not in ZONES:
            return f"detector output makes no sense: zone {str(det.zone)[:20]!r}"
        s = det.score
        if not isinstance(s, (int, float, np.floating)) or not math.isfinite(s) or not 0.0 <= s <= 1.0:
            return f"detector output makes no sense: score {str(s)[:20]}"
        if det.model_version != self.version:
            return (
                f"detector output makes no sense: says model "
                f"{str(det.model_version)[:20]}, loaded {self.version}"
            )
        if took > self.detector_budget:
            return f"detector too slow: {took * 1000:.0f} ms"
        return None

    def _bad(self, t: float, frame, reason: str, det=None, kind: str = "detector") -> None:
        self.fault = reason
        self.behaviour.update(t, FAULT)
        self.ml_action = HOLD
        score = getattr(det, "score", None)
        self.log.append(
            kind=kind,
            t=t,
            frame=frame.number if isinstance(frame, Frame) else self.frame_no,
            model=self.version,
            score=score if isinstance(score, (int, float)) else None,
            zone=FAULT,
            action=self._effective(HOLD),
            reason=reason,
        )
        self._set_action(t, reason)
        if self.trace is not None:
            self.trace.append((round(t, 3), self._effective(HOLD), FAULT, round(self.tool_speed, 4), None))

    # --- the strictest of the two layers ------------------------------------------------------------------

    def _effective(self, ml_action: str) -> str:
        if self.normal.latched:
            return STOP
        if self.fault:
            return HOLD
        return ml_action

    def _set_action(self, t: float, reason: str) -> None:
        new = self._effective(self.ml_action)
        if new == self.action:
            return
        if SPEED[new] == 0.0 and SPEED[self.action] > 0.0:
            self._watch = StopEvent(
                t, new, reason or "hand in the stop zone", peak_speed=round(self.tool_speed, 4)
            )
            self.stops.append(self._watch)
        elif SPEED[new] > 0.0:
            self._watch = None
        if SPEED[new] > 0.0 and self.holding:
            self.holding = False
            self.resume = (self.applied.copy(), 0.0)
        self.action = new

    def summary(self) -> dict:
        n = self.normal
        return {
            "sim_seconds": round(float(self.data.time), 3),
            "ticks": self.ticks,
            "physics_steps": self.physics_steps,
            "held_seconds": round(self.held_steps * self.dt, 3),
            "slow_seconds": round(self.slow_steps * self.dt, 3),
            "catch_up_seconds": round(self.catch_up_steps * self.dt, 3),
            "frames": self.frame_no,
            "stopped_by_normal_layer": n.latched,
            "peak_joint_speed": round(n.peak["joint_speed"], 4),
            "peak_tool_speed": round(n.peak["tool_speed"], 4),
            "joint_peaks": np.round(n.joint_peaks, 4).tolist(),
            "peak_where": n.peak_where,
            "tool_box_seen": np.round(n.tool_seen, 4).tolist(),
            "elbow_box_seen": np.round(n.elbow_seen, 4).tolist(),
            "stops": [s.__dict__ for s in self.stops],
            "log_entries": len(self.log.entries),
            "log_head": self.log.head,
        }


def _smooth(s: float) -> float:
    return s * s * (3 - 2 * s)


# --- the wrapper around v3's scene.step ------------------------------------------------------------------

_ORIGINAL: dict[str, Callable] = {"step": scene.step}
_GUARDS: dict[int, Guard] = {}


def _guarded_step(model: mujoco.MjModel, data: mujoco.MjData, n: int = 1) -> None:
    g = _GUARDS.get(id(data))
    if g is None or g.data is not data:
        return _ORIGINAL["step"](model, data, n)
    g.step(n)


def install() -> None:
    """Route v3's ``scene.step`` through the guards. Data not ``attach``-ed is stepped as before."""
    if scene.step is not _guarded_step:
        _ORIGINAL["step"] = scene.step
        scene.step = _guarded_step


def uninstall() -> None:
    scene.step = _ORIGINAL["step"]
    _GUARDS.clear()


def attach(guard: Guard) -> Guard:
    _GUARDS[id(guard.data)] = guard
    return guard


def detach(guard: Guard) -> None:
    if _GUARDS.get(id(guard.data)) is guard:
        del _GUARDS[id(guard.data)]


# --- rebuilding the decisions from the log ---------------------------------------------------------------


def replay(entries: list[dict], clear_for: float = CLEAR_FOR) -> list[str]:
    """Rebuild every logged decision from the log's own zones and events; return where they disagree."""
    b = Behaviour(clear_for)
    latched = False
    wrong = []
    for e in entries:
        kind = e.get("kind")
        if kind == "normal":
            latched = e.get("event") == "stop"
            continue
        zone = e.get("zone")
        if zone is None:
            continue
        action = b.update(e["t"], zone)
        if zone == FAULT:
            action = HOLD
        if latched:
            action = STOP
        if action != e.get("action"):
            wrong.append(f"entry {e.get('seq')}: logged {e.get('action')}, rebuilt {action}")
    return wrong
