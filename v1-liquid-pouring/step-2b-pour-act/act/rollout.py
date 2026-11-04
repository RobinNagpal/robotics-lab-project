"""The learned pour on the real arms: the policy moves Arm B, from the end of Step 1b to the end of the pour.

The loop, every tick (0.1 s of simulated time):

    1. the eyes look (demos/eyes.py): the pour camera's strip and whole picture, Arm B's wrist camera,
       Arm B's joints and the targets it is being given
    2. the policy gives Arm B's joint targets for the next tick (act/policy.py)
    3. the safety check (``Guard``) may refuse them: then Arm B holds where it is told now
    4. over the tick, the targets slide from the old to the new ones, updated every 0.01 s as in
       Step 2, while the simulation runs; the arm's own controllers follow them

The simulation waits while the policy thinks, so the pictures and the joints always belong to the same
moment. The pour ends when the policy says "finished" three ticks running, or after 60 s. Then the
balls are given 1 s to settle and counted (the counting is only for grading: the policy never sees it).

Nothing else is programmed: no inverse kinematics, no tilt or lip plan, no ball count, no correction for
the tube turning in the fingers. Two things still come from the simulator, as in Step 2: where the
receiver strip is cut from (the receiver's rim at the start), and the safety check's tube positions.

    python -m act.rollout ../step-2-pour/data/starts/seed150_size200.npy --policy act --target 50
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from lab_scene import scene as lab
from pour import start
from pour.pour import CONTROL, FRAME, SETTLE, Balls, level_table, tube_tilt, window_balls
from pour.tilt import ARM

from demos.eyes import Eyes

from .policy import Policy

TIME_LIMIT = 60.0  # s of simulated time (the programmed pour takes about 20)
FINISHED_TICKS = 3  # "finished" this many ticks running ends the pour
MAX_STEP = 0.10  # rad: the most any joint target may move in one tick (moving in, the expert's is 0.08)
TOUCH = 0.0005  # m: closer than this counts as touching


class Guard:
    """Refuses joint targets that would put Arm B, or the tube it holds, into Arm A or the receiver.

    It is checked as Step 2 checks its moves: set there, no physics. The tube is carried along with the
    fingers as it sits in them now. The balls are left out.
    """

    def __init__(self, model: mujoco.MjModel, source: str, receiver: str):
        self.model, self.source = model, source
        self.work = mujoco.MjData(model)
        self.joints = lab.arm_joint_ids(model, ARM)
        self.site = model.site(lab.pinch_site(ARM)).id
        self.tube_adr = model.jnt_qposadr[model.body(source).jntadr[0]]
        bodies = [model.body(model.geom_bodyid[g]).name for g in range(model.ngeom)]
        self.mine = np.array([b.startswith(f"{ARM}/") or b == source for b in bodies])
        self.theirs = np.array([b.startswith("arm2/") or b == receiver for b in bodies])
        self.refused = 0

    def touches(self, data: mujoco.MjData, q: np.ndarray) -> bool:
        m, w = self.model, self.work
        site_R = data.site_xmat[self.site].reshape(3, 3)
        body = data.body(self.source)
        # The tube in the pinch point's frame, now.
        rel_p = site_R.T @ (body.xpos - data.site_xpos[self.site])
        rel_R = site_R.T @ body.xmat.reshape(3, 3)
        w.qpos[:] = data.qpos
        w.qpos[self.joints] = q
        mujoco.mj_kinematics(m, w)
        R = w.site_xmat[self.site].reshape(3, 3)
        w.qpos[self.tube_adr : self.tube_adr + 3] = w.site_xpos[self.site] + R @ rel_p
        mujoco.mju_mat2Quat(w.qpos[self.tube_adr + 3 : self.tube_adr + 7], (R @ rel_R).flatten())
        mujoco.mj_kinematics(m, w)  # positions only, then the contacts: no forces needed
        mujoco.mj_collision(m, w)
        for c in w.contact[: w.ncon]:
            a, b = c.geom1, c.geom2
            if c.dist < -TOUCH and ((self.mine[a] and self.theirs[b]) or (self.mine[b] and self.theirs[a])):
                return True
        return False

    def check(self, data: mujoco.MjData, now: np.ndarray, new: np.ndarray) -> np.ndarray:
        """The targets to use: ``new`` limited to MAX_STEP per joint, or ``now`` if they would touch."""
        step = new - now
        biggest = np.abs(step).max()
        if biggest > MAX_STEP:
            new = now + step * (MAX_STEP / biggest)
        if self.touches(data, new):
            self.refused += 1
            return now
        return new


def run(
    path: Path,
    policy: Policy,
    target_pct: float = 50.0,
    guard: bool = True,
    movie: list | None = None,
    sim: dict | None = None,
) -> tuple[dict, dict]:
    """One learned pour from a saved start. Returns (what happened, what it did every tick).

    ``sim``: if given, the model, the simulation and the scene's notes are left in it, as Step 2's
    programmed pour does, so that the next step (putting the tubes back) can go on from where the
    pour ends.
    """
    model, data, containers, meta = start.load(path)
    receiver, source = meta["receiver"], meta["source"]
    rc = containers[receiver]
    counts, levels = level_table(rc)
    target = int(round(np.interp(target_pct, levels, counts)))
    balls = Balls(model, containers)
    balls.track(data, receiver, source)
    if sim is not None:
        sim.update(model=model, data=data, containers=containers, meta=meta, balls=balls)
    eyes = Eyes(model, data, receiver, rc.height)
    check = Guard(model, source, receiver) if guard else None
    actuators = lab.arm_actuator_ids(model, ARM)
    rbody = data.body(receiver)
    rim = rbody.xpos + rbody.xmat.reshape(3, 3)[:, 2] * rc.height
    policy.start(target_pct)
    steps_per_control = round(CONTROL / model.opt.timestep)
    controls_per_tick = round(FRAME / CONTROL)
    first = balls.count(data, receiver, source)
    trace: dict[str, list] = {
        k: []
        for k in ["t", "level", "tilt", "finished", "lip", "ctrl", "read_level", "read_flow", "read_air"]
    }
    t0, wall = data.time, time.time()
    failure, finished_run, think = None, 0, 0.0
    while True:
        seen = eyes.look(data)
        if movie is not None and len(trace["t"]) % 2 == 0:
            movie.append(seen["full"])
        tic = time.time()
        new, finished = policy.act(seen)
        think += time.time() - tic
        c = balls.count(data, receiver, source)
        trace["t"].append(data.time - t0)
        trace["level"].append(float(np.interp(c["receiver"], counts, levels)))
        trace["tilt"].append(tube_tilt(data, source))
        trace["finished"].append(finished)
        trace["lip"].append(_lip(data, source, containers[source]) - rim)
        trace["ctrl"].append(seen["ctrl"])
        for k, v in zip(("read_level", "read_flow", "read_air"), policy.extra, strict=True):
            trace[k].append(v)
        finished_run = finished_run + 1 if finished > 0.5 else 0
        if finished_run >= FINISHED_TICKS:
            break
        if data.time - t0 > TIME_LIMIT:
            failure = "time limit: never said finished"
            break
        now = data.ctrl[actuators].copy()
        if check is not None:
            new = check.check(data, now, new)
        for j in range(1, controls_per_tick + 1):
            data.ctrl[actuators] = now + (new - now) * j / controls_per_tick
            for _ in range(steps_per_control):
                mujoco.mj_step(model, data)
    end_pour = data.time - t0
    for _ in range(round(SETTLE / model.opt.timestep)):
        mujoco.mj_step(model, data)
    eyes.close()
    end = balls.count(data, receiver, source)
    tilts = np.array(trace["tilt"])
    result = {
        "start": path.stem,
        "seed": meta["seed"],
        "size": rc.height,
        "policy": policy.name,
        "target_pct": target_pct,
        "target_balls": target,
        "failure": failure,
        "balls": end,
        "level_pct": round(float(np.interp(end["receiver"], counts, levels)), 2),
        "error_balls": end["receiver"] - target,
        "within_window": failure is None and abs(end["receiver"] - target) <= window_balls(rc),
        "source_start": first["source"],
        "max_tilt_deg": round(float(tilts.max()), 1),
        "end_tilt_deg": round(float(tilts[-1]), 1),
        "refused": check.refused if check is not None else None,
        "sim_seconds": round(end_pour, 2),
        "think_ms": round(1000 * think / max(1, len(trace["t"])), 1),
        "wall_seconds": round(time.time() - wall, 1),
    }
    return result, trace


def _lip(data: mujoco.MjData, source: str, container) -> np.ndarray:
    """The source's lowest rim point (where the balls leave), in the world."""
    body = data.body(source)
    R = body.xmat.reshape(3, 3)
    axis = R[:, 2]
    down = np.array([0.0, 0.0, -1.0])
    side = down - axis * (down @ axis)
    n = np.linalg.norm(side)
    side = side / n if n > 1e-9 else R[:, 0]
    return body.xpos + axis * container.height + side * container.radius


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("start", type=Path)
    parser.add_argument("--policy", default="act", help="a run in runs/")
    parser.add_argument("--target", type=float, default=50.0)
    parser.add_argument("--ensemble", type=float, default=None)
    parser.add_argument("--no-guard", action="store_true")
    args = parser.parse_args()
    import torch

    torch.set_num_threads(4)
    result, _ = run(
        args.start, Policy(args.policy, ensemble=args.ensemble), args.target, guard=not args.no_guard
    )
    print(json.dumps(result, indent=1, default=str))


if __name__ == "__main__":
    main()
