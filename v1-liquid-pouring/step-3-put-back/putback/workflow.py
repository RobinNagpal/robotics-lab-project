"""The whole task on one scene: Step 0's table, Step 1b's pick and carry, the pour read by the level
network (Step 2), the source back in its cell (Step 3), the receiver in the destination rack (Step 4),
and the check (Step 5).

Step 1b is not run again here: each scene starts from a saved Step 1b run (step-2-pour/data/starts, made
by `make starts` there), where both tubes are already held at the pouring point. Step 0 is run again on
the scene as it was before anything moved, which gives exactly the table Step 1b worked from: the racks'
places and where the source stood.

    python -m putback.workflow --starts seed150_size200 seed151_size150 --workers 4
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
from pathlib import Path

import mujoco
import numpy as np
from carry.episode import RECEIVER_ARM, SOURCE_ARM, Watch, prepare
from carry.motion import Blocked
from carry.perceive import Perception
from lab_scene import scene as lab
from lab_scene.racks import slot_positions
from pour import pour

from .place import PutBack, cells, untip
from .upright import lean

ROOT = Path(__file__).resolve().parents[2]
STARTS = ROOT / "step-2-pour" / "data" / "starts"
FLOOR = slot_positions((0.0, 0.0))[0][2]  # where a tube's bottom rests in a cell
IN_CELL = 0.005  # m: a tube's bottom this close to a cell's centre is in it
# ...and leaning no more than this (degrees): a 25 mm tube in a 30 mm cell with 54 mm walls can lean
# about 5 to 7 degrees against a wall, and comes to rest so when let go.
STANDING = 8.0
SETTLE = 1.0  # s at the end, before the check


def step0(seed: int, source: str, perception: Perception) -> dict:
    """Step 0 on the scene before anything moved, as Step 1b saw it."""
    model, truth = lab.build(seed)
    data = mujoco.MjData(model)
    prepare(model, data)
    table = perception.table(model, data)
    # The tube Step 1b picked: it remembers where that was. (Matched here by where the simulator says
    # the source stood, because the saved run kept only the tube's name.)
    stood = data.body(source).xpos[:2]
    picked = min(table.in_rack("source_rack"), key=lambda t: np.linalg.norm(t.xy - stood))
    return {"table": table, "picked": picked, "truth": truth}


def options(table, picked, receiver_base: np.ndarray) -> tuple[list, list]:
    """Where each tube may go, best first: (cell, where its bottom rests, rack). The source: its own
    cell, then the free cells nearest it. The receiver: the free cells of the destination rack nearest
    Arm A's base first (far out, a tube held low is beyond the arm's reach)."""
    taken = [t.xy for t in table.tubes if t is not picked]

    def free(cell: np.ndarray) -> bool:
        return all(np.linalg.norm(cell[:2] - xy) > 0.015 for xy in taken)

    src = cells(table.racks["source_rack"])
    own = int(np.argmin([np.linalg.norm(c[:2] - picked.xy) for c in src]))
    source = [(own, src[own], "source_rack")]
    others = [i for i, c in enumerate(src) if i != own and free(c)]
    others.sort(key=lambda i: np.linalg.norm(src[i][:2] - src[own][:2]))
    source += [(i, src[i], "source_rack") for i in others]
    dst = cells(table.racks["destination_rack"])
    order = sorted(
        (i for i, c in enumerate(dst) if free(c)),
        key=lambda i: np.linalg.norm(dst[i][:2] - receiver_base[:2]),
    )
    return source, [(i, dst[i], "destination_rack") for i in order]


def where(data: mujoco.MjData, tube: str, rack: str) -> dict:
    """Is the tube standing in a cell of ``rack``, and which."""
    rack_xy = data.body(rack).xpos[:2]
    slots = np.array(slot_positions(tuple(rack_xy)))
    bottom = data.body(tube).xpos
    d = np.linalg.norm(slots[:, :2] - bottom[:2], axis=1)
    i = int(np.argmin(d))
    return {
        "cell": i,
        "off_mm": round(float(d[i]) * 1000, 1),
        "lean_deg": round(lean(data, tube), 1),
        "height_mm": round(float(bottom[2] - FLOOR) * 1000, 1),
        "in_cell": bool(d[i] < IN_CELL and lean(data, tube) < STANDING and abs(bottom[2] - FLOOR) < 0.004),
    }


def run_one(
    start: Path, target: float = 50.0, rule: str = "flow", perception=None, reader=None, on_frame=None
) -> dict:
    """The whole task from one saved Step 1b run. ``on_frame(data, stage)``, if given, is called every
    0.1 s of simulated time, for pictures."""
    from level.reader import NetReader

    wall = time.time()
    meta = json.loads(start.with_suffix(".json").read_text())
    perception = perception or Perception()
    seen = step0(meta["seed"], meta["source"], perception)
    sim: dict = {}
    poured, _ = pour.run(
        start, target, record=False, reader=reader or NetReader(), rule=rule, sim=sim, on_frame=on_frame
    )
    return put_back(start, meta, seen, poured, sim, on_frame=on_frame, wall=wall)


def put_back(
    start: Path, meta: dict, seen: dict, poured: dict, sim: dict, on_frame=None, wall: float | None = None
) -> dict:
    """Steps 3 to 5, from where the pour ends: both tubes back in racks, then the check.

    ``seen`` is Step 0's table (``step0``), ``poured`` what the pour reported, and ``sim`` the
    simulation the pour left (with ``move_in``, Arm B's way in, which the source leaves by backwards).
    Any pour that fills these in can be followed by this.
    """
    wall = time.time() if wall is None else wall
    source, receiver = meta["source"], meta["receiver"]
    poured.pop("settings", None)
    model, data = sim["model"], sim["data"]
    result = {
        "start": start.stem,
        "seed": meta["seed"],
        "size": poured["size"],
        "pour": poured,
        "failure": None,
    }
    names = {"receiver": receiver, "source": source}
    watch = Watch(model, data, names)
    if on_frame is not None:
        every = round(0.1 / model.opt.timestep)
        plain = watch.step
        count = [0]

        def filmed() -> None:
            plain()
            count[0] += 1
            if count[0] % every == 0:
                on_frame(data, put.log[-1][1] + " " + put.log[-1][2] if put.log else "tipping back")

        watch.step = filmed
    lifted = math.inf
    put = PutBack(model, data, seen["table"].tallest_rim, watch.step)
    to_source, to_destination = options(seen["table"], seen["picked"], data.body(f"{RECEIVER_ARM}/base").xpos)
    plans = {}
    try:
        if poured["failure"]:
            raise Blocked(f"pour: {poured['failure']}")
        # What already touched when the pour ended (with medium tubes the two grippers do, lightly) is
        # kept apart from what the arms touch while putting the tubes back.
        for _ in range(5):
            watch.step()
        result["touching_after_pour"] = sorted((arm, what) for arm, t in watch.touched.items() for what in t)
        lifted = untip(sim, watch.step)
        put.say(SOURCE_ARM, "upright")
        plans["source"] = put.plan(SOURCE_ARM, source, to_source)
        put.run(SOURCE_ARM, source, plans["source"])
        plans["receiver"] = put.plan(RECEIVER_ARM, receiver, to_destination)
        put.run(RECEIVER_ARM, receiver, plans["receiver"])
    except Blocked as e:
        result["failure"] = str(e)
    for _ in range(int(SETTLE / model.opt.timestep)):
        watch.step()
    # Step 5: the check, from the simulator (the arms never see these).
    end = sim["balls"].count(data, receiver, source)
    counts, levels = pour.level_table(sim["containers"][receiver])
    own = int(seen["truth"].source_slots[source])
    src_at = where(data, source, "source_rack")
    dst_at = where(data, receiver, "destination_rack")
    result.update(
        {
            "balls_end": end,
            "level_pct": round(float(np.interp(end["receiver"], counts, levels)), 2),
            "level_in_window": abs(end["receiver"] - poured["target_balls"])
            <= pour.window_balls(sim["containers"][receiver]),
            "balls_lost_putting_back": int(poured["balls"]["receiver"] - end["receiver"]),
            "spilled": end["spilled"] + end["air"],
            "source": {**src_at, "own_cell": src_at["cell"] == own, "plan": _plan(plans.get("source"))},
            "receiver": {**dst_at, "plan": _plan(plans.get("receiver"))},
            # Touched while putting back. The first move away from the pour may start touching (the
            # pour's own contact, see place.untip); what it touches before it ends is the pour's.
            "touched": sorted({(arm, what) for arm, _, what, t in watch.parts if t > lifted}),
            "touched_leaving_pour": sorted({(arm, what) for arm, _, what, t in watch.parts if t <= lifted}),
            "log": put.log,
            "sim_seconds": round(data.time, 1),
            "wall_seconds": round(time.time() - wall, 1),
        }
    )
    result["done"] = bool(
        result["failure"] is None
        and result["level_in_window"]
        and src_at["in_cell"]
        and result["source"]["own_cell"]
        and dst_at["in_cell"]
        and not result["touched"]
    )
    return result


def _plan(plan) -> dict | None:
    if plan is None:
        return None
    keys = ("cell", "turn", "share", "lift", "tried")
    return {**{k: getattr(plan, k) for k in keys}, "seconds": round(plan.seconds, 1)}


def _job(args) -> dict:
    import torch

    torch.set_num_threads(1)
    start, target, rule, out = args
    try:
        result = run_one(start, target, rule)
    except Exception as e:  # a crash in one scene should not stop the others
        result = {"start": start.stem, "failure": f"crashed: {e!r}", "done": False}
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{start.stem}.json").write_text(json.dumps(result, indent=1, default=str))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--starts", nargs="+", required=True, help="saved Step 1b runs, e.g. seed150_size200")
    parser.add_argument("--target", type=float, default=50.0)
    parser.add_argument("--rule", choices=("flow", "air"), default="flow")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("runs/workflow"))
    args = parser.parse_args()
    jobs = [(STARTS / f"{s}.npy", args.target, args.rule, args.out / args.rule) for s in args.starts]
    t0 = time.time()
    with mp.get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
        for r in pool.imap_unordered(_job, jobs):
            line = {k: r.get(k) for k in ("start", "done", "level_pct", "failure", "spilled")}
            line["source"] = {k: r.get("source", {}).get(k) for k in ("in_cell", "own_cell", "off_mm")}
            line["receiver"] = {k: r.get("receiver", {}).get(k) for k in ("in_cell", "off_mm")}
            print(f"{time.time() - t0:.0f} s {json.dumps(line)}", flush=True)


if __name__ == "__main__":
    main()
