"""Bench test before the pour is built: tip a water tube at a steady rate over the balance tube and record.

Every 0.1 s: the tilt, the balance's reading, and (from the simulator, for this study only) how many balls are
in the balance tube, still in the water tube, in the air and spilled. This shows at what tilt water starts to
flow, how fast it flows as the tilt grows, how far the reading lags the balls that have landed, and whether
anything misses the balance tube.

    cd step-3-fill-balance-tube
    P="pixi run --manifest-path ../simulation-environment/pixi.toml"
    PYTHONPATH=$PWD:$PWD/../step-1-weigh:$PWD/../simulation-environment \\
        $P python tilt_study.py --seed 1 --rate 0.7 --fast-until 76
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import test_scenes
from centrifuge_scene import liquid
from centrifuge_scene.racks import cell_centres, floor_height
from centrifuge_scene.scale import pan_top
from centrifuge_scene.settings import LAYOUT, LIQUID, TUBE
from motion import Arm
from weigh import tube_at

from fill import MAX_TILT, MOVE_IN_TILT, Filler

OUT = Path(__file__).parent / "results"


def counter(model, receiver: str, source: str):
    ids = liquid.ball_body_ids(model)

    def inside(data, name):
        b = model.body(name).id
        local = (data.xpos[ids] - data.xpos[b]) @ data.xmat[b].reshape(3, 3)
        r = np.array([liquid.inner_radius(TUBE, z) for z in local[:, 2]])
        return (r > 0) & (np.hypot(local[:, 0], local[:, 1]) < r)

    mine = None

    def count(data):
        nonlocal mine
        rec, src = inside(data, receiver), inside(data, source)
        if mine is None:  # only the balls in these two tubes when counting starts
            mine = rec | src
        rec, src = rec & mine, src & mine
        loose = ~rec & ~src & mine
        low = data.xpos[ids][:, 2] < pan_top() + 0.01
        return int(rec.sum()), int(src.sum()), int((loose & ~low).sum()), int((loose & low).sum())

    return count


def tube_tilt(data, body: int) -> float:
    """The tube's real tilt from upright, degrees (from the simulator: for this study only)."""
    return float(np.degrees(np.arccos(np.clip(data.xmat[body].reshape(3, 3)[2, 2], -1, 1))))


def lip_error(f, data, body: int, tilt: float) -> list[float]:
    """Where the water tube's lip really is, against where the plan puts it: mm along the pour direction,
    across it, and up (from the simulator: for this study only)."""
    import math

    from centrifuge_scene.settings import TUBE as T

    from moves import UP, rot_about

    d = f.pour_dir
    u = np.cross(UP, d)
    tilt_rot = rot_about(u, math.radians(tilt))
    pos, rot = f.pour_pinch(tilt)
    tip, tube_rot = f.arm.held.tube_pose(pos, rot)
    lip_plan = tip + T.height * (tilt_rot @ UP) + T.outline[-1][1] * (tilt_rot @ d)
    local = tube_rot.T @ (lip_plan - tip)
    real = data.xpos[body] + data.xmat[body].reshape(3, 3) @ local
    e = (real - lip_plan) * 1000
    return [round(float(e @ d), 1), round(float(e @ u), 1), round(float(e[2]), 1)]


def run(
    seed: int,
    rate: float,
    water: str = "water0",
    blank: str = "blank0",
    hand: float | None = None,
    fast_until: float = MOVE_IN_TILT,
) -> dict:
    model, data, truth = test_scenes.build(seed, changed=False)
    arm = Arm(model, data)
    filler = Filler(arm)
    if hand is not None:
        filler.pour_hand = hand
    arm.wait(0.3)
    filler.balance.tare()
    supply = (filler.supply,)
    t0 = time.time()

    # The empty tube into the holder, weighed.
    _, cell = truth.cells[blank]
    xy = np.array(cell_centres(LAYOUT.supply_rack)[cell])
    filler.pick_top(xy, floor_height(), tube_at(arm, xy), supply)
    filler.place_top(filler.holder_xy, pan_top(), filler.on_balance, filler.holder_turn())
    empty, steady = filler.read_steady()

    # The water tube, from the side, to its pour place.
    _, cell = truth.cells[water]
    wxy = np.array(cell_centres(LAYOUT.supply_rack)[cell])
    count = counter(model, blank, water)
    stages = {"start": count(data)}
    filler.pick_side(wxy, tube_at(arm, wxy))
    stages["picked"] = count(data)
    filler.move_in()
    stages["moved in"] = count(data)
    filler.check_tilts(MOVE_IN_TILT, MAX_TILT)
    rows = []
    ticks = 0

    def tick(f, tilt):
        nonlocal ticks
        ticks += 1
        if ticks % 10 == 0:
            rec, src, air, spilled = count(data)
            rows.append(
                {
                    "t": round(arm.time, 2),
                    "tilt": round(tilt, 2),
                    "tube_tilt": round(tube_tilt(data, model.body(water).id), 2),
                    "lip_err_mm": lip_error(f, data, model.body(water).id, tilt) if arm.held else None,
                    "grams": f.balance.grams(),
                    "in_receiver": rec,
                    "in_source": src,
                    "air": air,
                    "spilled": spilled,
                }
            )

    filler.on_tick = tick
    filler.tip(MOVE_IN_TILT, fast_until, 4.0)
    filler.tip(fast_until, MAX_TILT, rate)
    filler.tip(MAX_TILT, MOVE_IN_TILT, 40.0)
    filler.on_tick = None
    arm.wait(1.0)
    rec, src, air, spilled = count(data)
    final, _ = filler.read_steady()
    return {
        "seed": seed,
        "rate_deg_s": rate,
        "pour_hand_deg": filler.pour_hand,
        "empty_g": empty,
        "empty_steady": steady,
        "end": {"in_receiver": rec, "in_source": src, "air": air, "spilled": spilled, "grams": final},
        "ball_g": round(LIQUID.ball_mass * 1000, 4),
        "wall_s": round(time.time() - t0, 1),
        "stages": stages,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rate", type=float, default=2.0)
    parser.add_argument("--hand", type=float, default=None, help="fingers' direction while pouring (deg)")
    parser.add_argument("--fast-until", type=float, default=MOVE_IN_TILT, help="tip at 4 deg/s up to here")
    args = parser.parse_args()
    out = run(args.seed, args.rate, hand=args.hand, fast_until=args.fast_until)
    OUT.mkdir(exist_ok=True)
    (OUT / f"tilt_study_seed{args.seed}_rate{args.rate:g}_hand{out['pour_hand_deg']:g}.json").write_text(
        json.dumps(out, indent=1)
    )
    rows = out["rows"]
    first = next((r for r in rows if r["in_receiver"] > 0), None)
    print(f"empty tube {out['empty_g']} g (steady {out['empty_steady']}); wall {out['wall_s']} s")
    print("stages (receiver, source, air, spilled):", out["stages"])
    print(f"first ball in at {first['tilt'] if first else None} deg; end: {out['end']}")
    for r in rows[::10]:
        print(r)


if __name__ == "__main__":
    main()
