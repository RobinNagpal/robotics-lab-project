"""Step 5's study: where can the centrifuge stand so the arm can both load every rotor hole and push the lid shut?

A first version of this study (2026-11-23) found "turn the centrifuge −90°" and was wrong. It put the pinch
point 8 mm off the lid's face, but the closed fingertips reach 17.7 mm past the pinch point, so the arm had to
reach about 1 cm less far than it really must. It also did not check loading: turned −90°, Step 4 could not
load holes 0, 4 and 5 at all (20 of 20 jobs failed). This version uses the real planners:

- **the lid**: `close.LidCloser.plan`: the closed fingertips on the lid's inside face, following its whole
  swing from its resting angle to shut, the fingers' turn free along the way, at least 3 mm from everything
  but the lid;
- **the holes**: `rotor.Loader.best_turn` for each of the six holes (rotor empty, a capped sample): a clear
  finger turn for the slide from the start (20 or 10 cm back along the hole's axis) to the bottom.

Options: the centrifuge turned (degrees from today's layout: 0 = the lid opens away from the arm), moved
towards the arm, the lid's resting angle (98.7 degrees now, AutoBio's stop; a smaller one would need a stop
added), and where on the lid to push. No physics: inverse kinematics and gaps only.

    cd step-5-close-and-report
    PYTHONPATH=$PWD:$PWD/../step-1-weigh:$PWD/../step-3-fill-balance-tube:$PWD/../step-4-load-rotor:$PWD/../simulation-environment \\
        pixi run --manifest-path ../simulation-environment/pixi.toml python lid_reach.py --workers 4
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from centrifuge_scene import centrifuge, scene
from centrifuge_scene.settings import LAYOUT, LID_CLOSED, ROTOR

OUT = Path(__file__).parent / "results"
YAWS = (0.0, 30.0, -30.0, 60.0, -60.0, 90.0, -90.0, 180.0)
CLOSER = (0.0, 0.08)
RESTS = (98.7, 85.0, 75.0)
ALONGS = (0.5, 0.7)


def variant_layout(closer: float, yaw_deg: float):
    """The layout with the centrifuge ``closer`` m nearer the arm's base and turned ``yaw_deg`` more."""
    x, y = LAYOUT.centrifuge
    r = math.hypot(x, y)
    k = (r - closer) / r
    return dataclasses.replace(
        LAYOUT, centrifuge=(x * k, y * k), centrifuge_yaw=LAYOUT.centrifuge_yaw + math.radians(yaw_deg)
    )


def study(job: tuple[float, float]) -> dict:
    yaw, closer = job
    scene.LAYOUT = variant_layout(closer, yaw)  # scene.build reads it
    import test_scenes
    from motion import Arm, Blocked
    from rotor import Grip, Loader, hole_frame

    from close import LidCloser

    model, data, truth = test_scenes.build(0, changed=False)
    arm = Arm(model, data)
    loader = Loader(arm)
    holes = {}
    for k in range(ROTOR.slots):
        seat, _ = hole_frame(data, k)
        try:
            _, gap = loader.best_turn(k, seat, Grip(True))
            holes[k] = round(gap * 1000, 1)
        except Blocked:
            holes[k] = None
    lid = {}
    adr = centrifuge.lid_qpos_adr(model)
    saved = data.qpos[adr]
    for rest in RESTS:
        for along in ALONGS:
            data.qpos[adr] = LID_CLOSED + math.radians(rest)
            try:
                _, _, gap = LidCloser(arm, along=along, rest=math.radians(rest)).plan()
                lid[f"{rest:g} deg, push at {along:g}"] = round(gap * 1000, 1)
            except Blocked:
                lid[f"{rest:g} deg, push at {along:g}"] = None
            data.qpos[adr] = saved
    return {
        "yaw_deg": yaw,
        "closer_cm": round(closer * 100),
        "holes_clear": sum(v is not None for v in holes.values()),
        "holes_gap_mm": holes,
        "lid_gap_mm": lid,
        "lid_closable": [k for k, v in lid.items() if v is not None],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=4, help="at most 4 on this Mac")
    a = p.parse_args()
    jobs = [(y, c) for y in YAWS for c in CLOSER]
    rows = []
    with ProcessPoolExecutor(a.workers) as pool:
        for r in pool.map(study, jobs):
            rows.append(r)
            print(
                f"yaw {r['yaw_deg']:+5.0f} closer {r['closer_cm']:2d} cm: holes {r['holes_clear']}/6 "
                f"lid closable: {r['lid_closable'] or 'never'}",
                flush=True,
            )
    OUT.mkdir(exist_ok=True)
    (OUT / "lid_reach.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
