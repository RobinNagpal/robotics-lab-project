"""Step 2's programmed pour, recorded as a demonstration: what the learned pour sees, and what Arm B does.

The pour is Step 2's own (step-2-pour/pour/pour.py, read with the true ball count): move in, find the
flow, pour slowly, tip back early, settle, top up if short. Its settings are imported from there, so
the two stay the same. What is new here is what is saved, every 0.1 s (one "tick"):

    seen        the strip, the wide picture and the wrist picture (demos/eyes.py);
                Arm B's six joint angles and the six joint targets it is being given
    done        Arm B's joint targets the expert means at that moment (``intent``): the action to copy
    graded      level, balls in the receiver / source / air / spilled, tilt, phase, lip position

Noise (DART). A copied policy makes small mistakes, and then sees states the expert never showed it.
DART (Laskey et al., 2017) is the usual remedy without querying the expert later: while recording,
the expert's commands are nudged by a slowly wandering error (``noise``: the tilt by about 0.5
degrees, the lip by about 1 mm towards or away from the receiver and sideways), but the action saved is
the expert's own, without the nudge. So the demonstrations show how to get back on course from small
mistakes. The nudge starts once the lip is in place; moving in is not nudged.

The lip is never nudged down: the first try (0.75 degrees, 1.5 mm in every direction) brought the two
grippers together, about 20 mm apart at this tilt, and 20 of 118 pours jammed or ran away. A pour in
which Arm B's joints stay more than JAMMED from their targets for a second is stopped and left out
(tipping back fast, a clean pour lags by up to 0.4 rad, but only for a moment).

    python -m demos.record ../step-2-pour/data/starts/seed100_size150.npy --target 50 --noise 1
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import mujoco
import numpy as np
from carry import motion
from lab_scene import scene as lab
from pour import start
from pour.pour import (
    BACK_RATE,
    CONTROL,
    FIND_RATE,
    FLOW_WINDOW,
    FRAME,
    MAX_TILT,
    MAX_TOPUPS,
    MOVE_IN_TILT,
    PHASES,
    POUR_RATE,
    REACTION,
    REGRIP_SHARE,
    REST_TILT,
    SETTLE,
    TOPUP_RATE,
    Balls,
    Blocked,
    level_table,
    move_in_path,
    tube_tilt,
    window_balls,
)
from pour.tilt import ARM, Tilter

from .eyes import Eyes

TIME_LIMIT = 90.0  # s of simulated time, as in Step 2
DONE_HOLD = 1.0  # s held still at the end, so the policy sees what "finished" looks like
NOISE_TILT = 0.5  # degrees: how far the nudge moves the tilt, typically (one standard deviation)
NOISE_LIP = 0.001  # m: the same for the lip, towards the receiver and sideways (never up or down)
JAMMED = 0.3  # rad: Arm B this far from its joint targets for JAMMED_FOR, once in place, is stuck
JAMMED_FOR = 1.0  # s
NOISE_TIME = 0.5  # s: how long a nudge lasts before it wanders elsewhere


class Wander:
    """A slowly wandering error (an Ornstein-Uhlenbeck process): ``scale`` typical size, ``span`` s memory."""

    def __init__(self, rng: np.random.Generator, scale: np.ndarray, span: float = NOISE_TIME):
        self.rng, self.scale, self.span = rng, np.asarray(scale, float), span
        self.value = self.rng.normal(size=self.scale.shape) * self.scale

    def step(self, dt: float) -> np.ndarray:
        a = dt / self.span
        self.value += -a * self.value + self.scale * math.sqrt(2 * a) * self.rng.normal(size=self.scale.shape)
        return self.value


def run(path: Path, target_pct: float = 50.0, noise: float = 0.0, seed: int = 0) -> tuple[dict, dict]:
    """One recorded pour from a saved start. ``noise`` scales the nudges (0: none, 1: the sizes above)."""
    model, data, containers, meta = start.load(path)
    receiver, source = meta["receiver"], meta["source"]
    rc = containers[receiver]
    counts, levels = level_table(rc)
    target = int(round(np.interp(target_pct, levels, counts)))
    balls = Balls(model, containers)
    balls.track(data, receiver, source)
    tilter = Tilter(model, data, source, containers[source].height, containers[source].radius)
    actuators = lab.arm_actuator_ids(model, ARM)
    eyes = Eyes(model, data, receiver, rc.height)
    rng = np.random.default_rng(seed)
    nudge = Wander(rng, noise * np.array([NOISE_TILT, NOISE_LIP, NOISE_LIP, 0.0])) if noise > 0 else None

    rbody = data.body(receiver)
    rim = rbody.xpos + rbody.xmat.reshape(3, 3)[:, 2] * rc.height
    keys = [
        "strip",
        "wide",
        "wrist",
        "qpos",
        "ctrl",
        "intent",
        "t",
        "phase",
        "receiver",
        "source",
        "air",
        "spilled",
        "level",
        "tilt",
        "tilt_cmd",
        "lip",
    ]
    ticks: dict[str, list] = {k: [] for k in keys}
    log: list[tuple[float, str]] = []
    t0, wall = data.time, time.time()
    first = balls.count(data, receiver, source)
    steps_per_control = round(CONTROL / model.opt.timestep)
    steps_per_frame = round(FRAME / model.opt.timestep)

    # What the expert means Arm B's joints to be: during the move in, the targets it gives; once the
    # lip is in place, the joints for the planned tilt and lip without the nudge (solved at each tick).
    s = {"phase": "move in", "tilt_cmd": 0.0, "place": None, "intent": None, "count": 0}

    def tick() -> None:
        seen = eyes.look(data)
        if s["intent"] is None:
            if s["place"] is None:  # moving in: the targets given are the intent
                s["intent"] = seen["ctrl"].copy()
            else:
                s["intent"], _ = tilter.solve(data, s["tilt_cmd"], s["place"], seen["ctrl"], iterations=30)
        c = balls.count(data, receiver, source)
        for k in ("strip", "wide", "wrist", "qpos", "ctrl"):
            ticks[k].append(seen[k])
        ticks["intent"].append(s["intent"].copy())
        ticks["t"].append(data.time - t0)
        ticks["phase"].append(PHASES.index(s["phase"]))
        for k in ("receiver", "source", "air", "spilled"):
            ticks[k].append(c[k])
        ticks["level"].append(float(np.interp(c["receiver"], counts, levels)))
        ticks["tilt"].append(tube_tilt(data, source))
        ticks["tilt_cmd"].append(s["tilt_cmd"])
        ticks["lip"].append(tilter.lip(data, source) - rim)
        if s["place"] is None:
            s["intent"] = None  # still moving in: read it again next tick

    def step(phase: str) -> None:
        s["phase"] = phase
        mujoco.mj_step(model, data)
        s["count"] += 1
        if s["count"] % steps_per_frame == 0:
            tick()
        if data.time - t0 > TIME_LIMIT:
            raise Blocked("time limit")
        if s["place"] is not None and s["count"] % steps_per_control == 0:
            gap = np.abs(data.qpos[tilter.joints] - data.ctrl[actuators]).max()
            s["behind"] = s.get("behind", 0) + 1 if gap > JAMMED else 0
            if s["behind"] * CONTROL >= JAMMED_FOR:
                raise Blocked(f"jammed: Arm B {gap:.2f} rad from its targets for {JAMMED_FOR} s")

    failure, stopped_at, topups = None, None, 0
    history: list[tuple[float, int]] = []
    try:
        path_q, place = move_in_path(tilter, data, source, receiver, rc.height)
        for _ in motion.follow(model, data, ARM, path_q, speed=0.08):
            step("move in")
        s["place"], s["tilt_cmd"], s["intent"] = place, MOVE_IN_TILT, None
        q = data.ctrl[actuators].copy()
        log.append((data.time - t0, "in position"))

        def command() -> None:
            """Solve and give Arm B's joint targets for the planned tilt and lip, nudged if noisy."""
            nonlocal q
            tilter.regrip(data, source, REGRIP_SHARE)
            tilt, lip = s["tilt_cmd"], place
            if nudge is not None:
                n = nudge.step(CONTROL)
                q_new, miss = tilter.solve(data, tilt + n[0], place + n[1:], q, iterations=20)
                if miss > 2e-3:
                    # The nudged pose is out of reach (near the end of a joint's travel): drop the
                    # nudge and start it again from nothing, rather than lose the pour.
                    nudge.value[:] = 0.0
                    s["dropped"] = s.get("dropped", 0) + 1
            if nudge is None or miss > 2e-3:
                q_new, miss = tilter.solve(data, tilt, lip, q, iterations=20)
            if miss > 2e-3:
                raise Blocked(f"Arm B cannot reach tilt {tilt:.1f}")
            q = q_new
            data.ctrl[actuators] = q
            # Without the nudge the targets given are the intent; with it, the clean ones are solved at
            # the next tick.
            s["intent"] = None if nudge is not None else q.copy()

        def hold(phase: str, seconds: float) -> None:
            """Keep still; with noise, the nudge keeps wandering, as it would for a real arm."""
            for _ in range(round(seconds / CONTROL)):
                if nudge is not None:
                    command()
                for _ in range(steps_per_control):
                    step(phase)

        def tip_to(goal: float, rate: float, phase: str, until=None) -> bool:
            while True:
                if until is not None and until():
                    return True
                if abs(goal - s["tilt_cmd"]) < 1e-6:
                    return False
                s["tilt_cmd"] += float(np.clip(goal - s["tilt_cmd"], -rate * CONTROL, rate * CONTROL))
                command()
                for _ in range(steps_per_control):
                    step(phase)
                history.append((data.time, balls.count(data, receiver, source)["receiver"]))

        def flow() -> float:
            if len(history) < 2:
                return 0.0
            t_now, n_now = history[-1]
            t_old, n_old = next(h for h in history if h[0] >= t_now - FLOW_WINDOW)
            return (n_now - n_old) / max(t_now - t_old, CONTROL)

        before = first["receiver"]

        def arrived() -> bool:
            return bool(history) and history[-1][1] > before

        def enough() -> bool:
            return bool(history) and history[-1][1] + flow() * REACTION >= target

        while True:
            if not arrived():
                tip_to(MAX_TILT, FIND_RATE, "find flow", until=lambda: arrived() or enough())
                log.append((data.time - t0, f"flow at {s['tilt_cmd']:.1f} deg"))
            if not enough() and not tip_to(MAX_TILT, POUR_RATE, "pour", until=enough):
                log.append((data.time - t0, "reached the largest tilt"))
            stopped_at = s["tilt_cmd"]
            log.append((data.time - t0, f"tip back from {stopped_at:.1f} deg, {history[-1][1]} balls"))
            tip_to(REST_TILT, BACK_RATE, "tip back")
            hold("settle", SETTLE)
            n = balls.count(data, receiver, source)["receiver"]
            short = target - n
            if short <= window_balls(rc) or topups >= MAX_TOPUPS:
                break
            topups += 1
            log.append((data.time - t0, f"top up: {short} short"))
            tip_to(stopped_at - 2.0, TOPUP_RATE, "top up")
            history.clear()
            history.append((data.time, n))
            before = n - 1  # already where the balls flow: pour slowly from here
        hold("done", DONE_HOLD)
    except Blocked as e:
        failure = str(e)
    eyes.close()
    end = balls.count(data, receiver, source)
    result = {
        "start": path.stem,
        "seed": meta["seed"],
        "size": rc.height,
        "receiver": receiver,
        "source_tube": source,
        "target_pct": target_pct,
        "target_balls": target,
        "noise": noise,
        "nudges_dropped": s.get("dropped", 0),  # control steps where the nudged pose was out of reach
        "failure": failure,
        "balls": end,
        "level_pct": round(float(np.interp(end["receiver"], counts, levels)), 2),
        "within_window": failure is None and abs(end["receiver"] - target) <= window_balls(rc),
        "source_start": first["source"],
        "topups": topups,
        "stopped_at_deg": stopped_at,
        "sim_seconds": round(data.time - t0, 2),
        "wall_seconds": round(time.time() - wall, 1),
        "ticks": len(ticks["t"]),
        "log": log,
    }
    return result, ticks


def save(ticks: dict, result: dict, out: Path) -> None:
    """One .npz with every tick, and a .json beside it with how the pour went."""
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.array(v) for k, v in ticks.items()}
    for k in ("strip", "wide", "wrist"):
        arrays[k] = arrays[k].astype(np.uint8)
    for k in ("qpos", "ctrl", "intent", "lip", "t", "level", "tilt", "tilt_cmd"):
        arrays[k] = arrays[k].astype(np.float32)
    np.savez_compressed(out.with_suffix(".npz"), **arrays)
    out.with_suffix(".json").write_text(json.dumps(result, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("start", type=Path)
    parser.add_argument("--target", type=float, default=50.0)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--out", type=Path, default=None, help="save the demonstration here (no suffix)")
    args = parser.parse_args()
    result, ticks = run(args.start, args.target, noise=args.noise)
    if args.out is not None:
        save(ticks, result, args.out)
    print(json.dumps(result, indent=1, default=str))


if __name__ == "__main__":
    main()
