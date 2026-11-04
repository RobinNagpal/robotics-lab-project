"""The programmed pour: Arm B tips the source until the receiver holds the target number of balls.

It starts where Step 1b ends (pour.start) and watches the true number of balls in the receiver,
as a perfect scale under it would. The steps, from docs/solutions/step-2-pour-approaches.md:

    move in      lift, tip to 78 degrees and bring the lip 15 mm from the receiver's axis, 20 mm
                 above its rim: no balls leave yet (the first leave at 81 to 88 degrees)
    find flow    tip on at about 2 degrees a second until the first balls land
    pour         tip on slowly, about 0.7 degrees a second
    stop early   tip back as soon as count + flow x REACTION reaches the target: the balls already
                 on their way still land
    tip back     to 70 degrees, at 40 degrees a second
    settle       wait, count again, and top up if short

While it pours, the pour camera's picture of the receiver is saved every 0.1 s with what the
simulation knows at that moment (balls in the receiver, in the air, spilled; the tilt), as the
training pictures for the level network.

    python -m pour.pour data/starts/seed100_size150.npy --target 50
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from carry import motion
from lab_scene import glassware, liquid
from lab_scene import scene as lab
from lab_scene.settings import IMAGE_SIZE, POUR_CAMERA

from . import start
from .tilt import ARM, LIP_OUT, LIP_UP, Tilter

MOVE_IN_TILT = 78.0  # degrees: tipped this far before the lip goes to its place; no flow yet
LIFT = 0.05  # m: the lip comes in this far above its place, then down
REST_TILT = 70.0  # degrees: tipped back to here, below where the flow starts
MAX_TILT = 110.0
FIND_RATE, POUR_RATE, BACK_RATE = 2.0, 0.7, 40.0  # degrees per second
TOPUP_RATE = 10.0  # back up to just short of where it stopped
REACTION = 0.6  # s of flow still to land after tipping back starts
# Pouring with the level network: the first balls count as landed once it reads this much (%).
# It reads an empty receiver as about 0.8 %, 1.5 % at most in 99 of 100 test pictures.
ARRIVED_PCT = 2.0
FLOW_WINDOW = 0.5  # s over which the flow is measured
SETTLE = 1.0  # s
MAX_TOPUPS = 3
CONTROL = 0.01  # s between joint target updates
# Each update, this share of the change in where the tube sits in the fingers is taken in (tilt.regrip):
# all of it at once would chase the shaking of the balls.
REGRIP_SHARE = 0.1
FRAME = 0.1  # s between saved pictures
TIME_LIMIT = 90.0  # s of simulated time for the whole pour
# The receiver's picture: this wide and tall (pixels), from this far above its rim. At the pour
# camera's 0.6 mm per pixel it covers 30 mm above the rim (the lip and the falling balls) down to
# 210 mm below it (the bottom of a large tube).
CROP_W, CROP_H, CROP_ABOVE = 96, 384, 48
PHASES = ("move in", "find flow", "pour", "tip back", "settle", "top up", "done")


@dataclass
class Balls:
    """Counts the balls in each place, the way liquid.balls_in does, for all balls at once."""

    model: mujoco.MjModel
    containers: dict
    ids: np.ndarray = field(init=False)

    def __post_init__(self):
        self.ids = liquid.ball_body_ids(self.model)
        self.tables = {}
        for name, c in self.containers.items():
            z = np.linspace(0, c.height, 400)
            self.tables[name] = (z, np.array([glassware.inner_radius(c, h) for h in z]))

    def inside(self, data: mujoco.MjData, name: str) -> np.ndarray:
        body = data.body(name)
        local = (data.xpos[self.ids] - body.xpos) @ body.xmat.reshape(3, 3)
        z, r = self.tables[name]
        room = np.interp(local[:, 2], z, r, left=0.0, right=0.0)
        return (room > 0) & (np.hypot(local[:, 0], local[:, 1]) < room)

    def track(self, data: mujoco.MjData, receiver: str, source: str) -> None:
        """Follow only the balls in the source and the receiver now, not those in the other tubes."""
        self.mine = self.inside(data, receiver) | self.inside(data, source)

    def count(self, data: mujoco.MjData, receiver: str, source: str) -> dict[str, int]:
        rec = self.inside(data, receiver) & self.mine
        src = self.inside(data, source) & self.mine
        loose = ~rec & ~src & self.mine
        low = data.xpos[self.ids][:, 2] < 0.12  # fallen to the table or the racks
        return {
            "receiver": int(rec.sum()),
            "source": int(src.sum()),
            "air": int((loose & ~low).sum()),
            "spilled": int((loose & low).sum()),
        }


def level_table(container) -> tuple[np.ndarray, np.ndarray]:
    """Ball counts against level (%), to turn a count into a level."""
    levels = np.linspace(0, 1, 201)
    counts = np.array([liquid.ball_count(container, x) for x in levels], float)
    # Several levels give the same count: keep the lowest, so an empty tube reads 0 %.
    counts, first = np.unique(counts, return_index=True)
    return counts, levels[first] * 100


class Camera:
    """The pour camera's picture, cut down to the receiver."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, receiver: str, height: float):
        self.renderer = mujoco.Renderer(model, IMAGE_SIZE[1], IMAGE_SIZE[0])
        body = data.body(receiver)
        rim = body.xpos + body.xmat.reshape(3, 3)[:, 2] * height
        u, v = self.project(data, rim)
        self.x0 = int(np.clip(round(u) - CROP_W // 2, 0, IMAGE_SIZE[0] - CROP_W))
        self.y0 = int(np.clip(round(v) - CROP_ABOVE, 0, IMAGE_SIZE[1] - CROP_H))

    @staticmethod
    def project(data: mujoco.MjData, p: np.ndarray) -> tuple[float, float]:
        cam = data.camera(POUR_CAMERA.name)
        R = cam.xmat.reshape(3, 3)
        local = R.T @ (p - cam.xpos)
        f = IMAGE_SIZE[1] / 2 / math.tan(math.radians(POUR_CAMERA.fovy) / 2)
        return IMAGE_SIZE[0] / 2 + f * local[0] / -local[2], IMAGE_SIZE[1] / 2 - f * local[1] / -local[2]

    def full(self, data: mujoco.MjData) -> np.ndarray:
        # With other renderers open (pictures for the docs), draw with this one's own graphics context.
        self.renderer._gl_context.make_current()
        self.renderer.update_scene(data, camera=POUR_CAMERA.name)
        return self.renderer.render()

    def crop(self, image: np.ndarray) -> np.ndarray:
        return image[self.y0 : self.y0 + CROP_H, self.x0 : self.x0 + CROP_W].copy()


class Blocked(RuntimeError):
    pass


def move_in_path(
    tilter: Tilter, data, source: str, receiver: str, rec_height: float
) -> tuple[list, np.ndarray]:
    """Joint targets from the end of Step 1b to the pour position, checked for collisions with no physics."""
    body = data.body(receiver)
    axis = body.xmat.reshape(3, 3)[:, 2]
    rim = body.xpos + axis * rec_height
    # From the rim's centre, not the tube's base: the receiver leans a few degrees in Arm A's fingers,
    # which puts the rim about 5 mm to the side of its base, and 5 mm to the side spills about 15 %.
    place = rim + [LIP_OUT, 0.0, LIP_UP]
    first = tilter.lip(data, source)
    above = place + [0, 0, LIFT]
    q = data.qpos[tilter.joints].copy()
    waypoints = []
    # Up and over while tipping, then straight down.
    for s in np.linspace(0, 1, 40)[1:]:
        lip = first + (above - first) * s
        lip[2] = max(lip[2], first[2] + LIFT * math.sin(math.pi * s / 2))
        waypoints.append((MOVE_IN_TILT * s, lip))
    for s in np.linspace(0, 1, 11)[1:]:
        waypoints.append((MOVE_IN_TILT, above + (place - above) * s))
    path = []
    for tilt, lip in waypoints:
        q, miss = tilter.solve(data, tilt, lip, q)
        if miss > 1e-3:
            raise Blocked(f"Arm B cannot reach tilt {tilt:.0f} with the lip at {np.round(lip, 3)}")
        hit = tilter.clashes(data, q, tilt, lip, source, ("arm2", receiver))
        if hit:
            raise Blocked(f"moving in at tilt {tilt:.0f}: {hit[:2]}")
        path.append(q)
    return path, place


def tube_tilt(data, name: str) -> float:
    """How far the tube's axis is from upright, degrees."""
    return math.degrees(math.acos(np.clip(data.body(name).xmat.reshape(3, 3)[2, 2], -1, 1)))


def run(
    path: Path,
    target_pct: float = 50.0,
    record: bool = True,
    find_rate: float = FIND_RATE,
    pour_rate: float = POUR_RATE,
    reaction: float = REACTION,
    pictures: Path | None = None,
    movie: list | None = None,
    reader=None,
    rule: str = "flow",
    sim: dict | None = None,
    on_frame=None,
) -> tuple[dict, dict]:
    """One pour from a saved start. Returns (what happened, the saved frames and their labels).

    ``pictures``: save the camera's whole picture at the start of each phase there. ``movie``: a list
    that collects the whole picture every 0.2 s, for animations.

    ``reader``: what the pour reads the receiver with. None reads the true ball count (a perfect
    scale). Otherwise the level network (level.reader.NetReader): it is shown the pour camera's
    picture every 0.1 s and gives the level, the flow and the balls in the air, turned into balls.
    ``rule``: when to tip back. "flow": reading + flow x ``reaction`` reaches the target (the
    programmed pour's rule); "air": reading + balls in the air reaches it (needs the network).
    ``sim``: if given, the model, the simulation and the scene's notes are left in it, so that the
    next step (putting the tubes back) can go on from where the pour ends. ``on_frame(data, phase)``, if
    given, is called every 0.1 s, for pictures.
    """
    model, data, containers, meta = start.load(path)
    receiver, source = meta["receiver"], meta["source"]
    rc = containers[receiver]
    counts, levels = level_table(rc)
    target = int(round(np.interp(target_pct, levels, counts)))
    balls = Balls(model, containers)
    balls.track(data, receiver, source)
    tilter = Tilter(model, data, source, containers[source].height, containers[source].radius)
    actuators = lab.arm_actuator_ids(model, ARM)
    camera = Camera(model, data, receiver, rc.height) if (record or reader is not None) else None
    if reader is not None:
        reader.start(rc)
    if rule == "air" and reader is None:
        raise ValueError("the air rule needs the level network's reading")
    if sim is not None:
        sim.update(model=model, data=data, containers=containers, meta=meta, balls=balls, tilter=tilter)
    rbody = data.body(receiver)
    rim = rbody.xpos + rbody.xmat.reshape(3, 3)[:, 2] * rc.height
    frames: dict[str, list] = {
        k: []
        for k in (
            "image",
            "t",
            "receiver",
            "source",
            "air",
            "spilled",
            "level",
            "surface",
            "tilt",
            "tilt_cmd",
            "phase",
            "lip",
            "turned",
        )
    }
    log: list[tuple[float, str]] = []
    first = balls.count(data, receiver, source)
    before = first["receiver"]  # balls in the receiver before this round of pouring
    wall = time.time()
    t0 = data.time
    history: list[tuple[float, int]] = []
    snapshots = {}
    if reader is not None:
        frames["reading"] = []  # what the network read: level, flow, air (%, % per s, %)

    def frame(phase: str, tilt_cmd: float, image: np.ndarray | None):
        c = balls.count(data, receiver, source)
        if record:
            if pictures is not None and phase not in snapshots:
                snapshots[phase] = image
            if movie is not None and len(frames["t"]) % 2 == 0:
                movie.append((image, phase))
            frames["image"].append(camera.crop(image))
            frames["t"].append(data.time - t0)
            for k in ("receiver", "source", "air", "spilled"):
                frames[k].append(c[k])
            frames["level"].append(float(np.interp(c["receiver"], counts, levels)))
            frames["surface"].append(100 * liquid.level(model, data, rc, receiver))
            frames["tilt"].append(tube_tilt(data, source))
            frames["tilt_cmd"].append(tilt_cmd)
            frames["phase"].append(PHASES.index(phase))
            # Where the lip is, from the receiver's rim centre (m).
            frames["lip"].append(tilter.lip(data, source) - rim)
            frames["turned"].append(tilter.turned(data, source))
            if reader is not None:
                frames["reading"].append(reader.last)

    steps_per_control = round(CONTROL / model.opt.timestep)
    steps_per_frame = round(FRAME / model.opt.timestep)
    step_count = 0

    def step(phase: str, tilt_cmd: float):
        nonlocal step_count
        mujoco.mj_step(model, data)
        step_count += 1
        if step_count % steps_per_frame == 0:
            image = camera.full(data) if camera is not None else None
            if reader is not None:
                reader.see(camera.crop(image))
            frame(phase, tilt_cmd, image)
            if on_frame is not None:
                on_frame(data, phase)
        if data.time - t0 > TIME_LIMIT:
            raise Blocked("time limit")

    failure = None
    phase = "move in"
    tilt_cmd = 0.0
    stopped_at = None
    topups = 0
    topping = False
    if sim is not None:
        sim["lip_start"] = tilter.lip(data, source)  # where the lip was, the source upright
    try:
        path_q, place = move_in_path(tilter, data, source, receiver, rc.height)
        if sim is not None:
            sim["place"] = place
            sim["move_in"] = [data.qpos[tilter.joints].copy(), *path_q]  # Arm B's way in, from the start
        for _ in motion.follow(model, data, ARM, path_q, speed=0.08):
            step(phase, tilt_cmd)
        tilt_cmd = MOVE_IN_TILT
        q = data.ctrl[actuators].copy()
        log.append((data.time - t0, "in position"))

        def tip_to(goal: float, rate: float, phase_: str, until=None):
            """Tip at ``rate`` towards ``goal``; stop early when ``until()`` says so."""
            nonlocal tilt_cmd, q
            while True:
                if until is not None and until():
                    return True
                if abs(goal - tilt_cmd) < 1e-6:
                    return False
                dt = CONTROL
                tilt_cmd += float(np.clip(goal - tilt_cmd, -rate * dt, rate * dt))
                tilter.regrip(data, source, REGRIP_SHARE)
                q_new, miss = tilter.solve(data, tilt_cmd, place, q, iterations=20)
                if miss > 2e-3:
                    raise Blocked(f"Arm B cannot reach tilt {tilt_cmd:.1f}")
                q = q_new
                data.ctrl[actuators] = q
                for _ in range(steps_per_control):
                    step(phase_, tilt_cmd)
                history.append((data.time, reading()))

        def reading() -> float:
            """Balls in the receiver now: counted, or the network's latest reading turned into balls."""
            if reader is None:
                return balls.count(data, receiver, source)["receiver"]
            return reader.balls

        def flow() -> float:
            if reader is not None:
                return reader.flow_balls
            if len(history) < 2:
                return 0.0
            t_now, n_now = history[-1]
            old = [h for h in history if h[0] >= t_now - FLOW_WINDOW]
            t_old, n_old = old[0]
            return (n_now - n_old) / max(t_now - t_old, CONTROL)

        def arrived() -> bool:
            if reader is not None:  # the network reads an empty tube as about 0.8 %
                return topping or reader.level >= ARRIVED_PCT
            return bool(history) and history[-1][1] > before

        def enough() -> bool:
            if not history:
                return False
            n = history[-1][1]
            if rule == "air":
                return n + reader.air_balls >= target
            return n + flow() * reaction >= target

        while True:
            phase = "find flow" if not arrived() else "pour"
            if phase == "find flow":
                tip_to(MAX_TILT, find_rate, phase, until=lambda: arrived() or enough())
                log.append((data.time - t0, f"flow at {tilt_cmd:.1f} deg"))
            if not enough():
                phase = "pour"
                if not tip_to(MAX_TILT, pour_rate, phase, until=enough):
                    log.append((data.time - t0, "reached the largest tilt"))
            stopped_at = tilt_cmd
            log.append((data.time - t0, f"tip back from {tilt_cmd:.1f} deg, {history[-1][1]:.1f} balls"))
            phase = "tip back"
            tip_to(REST_TILT, BACK_RATE, phase)
            phase = "settle"
            for _ in range(round(SETTLE / model.opt.timestep)):
                step(phase, tilt_cmd)
            # Settled: counted, or the network's readings over the last half second averaged.
            n = reading() if reader is None else reader.settled_balls()
            history.append((data.time, n))
            short = target - n
            if short <= window_balls(rc) or topups >= MAX_TOPUPS:
                break
            topups += 1
            log.append((data.time - t0, f"top up: {short:.1f} short"))
            phase = "top up"
            tip_to(stopped_at - 2.0, TOPUP_RATE, phase)
            history.clear()
            history.append((data.time, n))
            # Already where the balls flow: pour slowly from here.
            before = n - 1
            topping = True
        phase = "done"
        for _ in range(round(0.5 / model.opt.timestep)):
            step(phase, tilt_cmd)
    except Blocked as e:
        failure = str(e)
    if sim is not None:
        sim["tilt"] = tilt_cmd  # where the source is left tipped
    end = balls.count(data, receiver, source)
    pct = float(np.interp(end["receiver"], counts, levels))
    result = {
        "start": path.stem,
        "seed": meta["seed"],
        "size": rc.height,
        "receiver": receiver,
        "target_pct": target_pct,
        "target_balls": target,
        "failure": failure,
        "balls": end,
        "level_pct": round(pct, 2),
        "within_window": failure is None and abs(end["receiver"] - target) <= window_balls(rc),
        "source_start": first.get("source"),
        "topups": topups,
        "stopped_at_deg": stopped_at,
        "sim_seconds": round(data.time - t0, 2),
        "wall_seconds": round(time.time() - wall, 1),
        "settings": {"find_rate": find_rate, "pour_rate": pour_rate, "reaction": reaction},
        "log": log,
    }
    if reader is not None:
        result["read_by"] = reader.name
        result["rule"] = rule
        result["read_level_pct"] = round(float(reader.settled_level()), 2)  # its last settled reading
    if pictures is not None:
        from PIL import Image

        pictures.mkdir(parents=True, exist_ok=True)
        for k, img in snapshots.items():
            Image.fromarray(img).save(pictures / f"{path.stem}_{k.replace(' ', '_')}.png")
    return result, frames


def window_balls(container) -> int:
    """How many balls 2 % of the tube is (the 48 to 52 % window): 2, 3 or 4 for small, medium, large."""
    counts, levels = level_table(container)
    return int(round((np.interp(52, levels, counts) - np.interp(48, levels, counts)) / 2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("start", type=Path)
    parser.add_argument("--target", type=float, default=50.0)
    parser.add_argument("--pictures", type=Path, default=None)
    parser.add_argument("--no-record", action="store_true")
    args = parser.parse_args()
    result, frames = run(args.start, args.target, record=not args.no_record, pictures=args.pictures)
    print(json.dumps(result, indent=1, default=str))


if __name__ == "__main__":
    main()
