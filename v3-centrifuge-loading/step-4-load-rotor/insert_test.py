"""Step 4's first physics test: one sample tube, one rotor hole.

The open question from docs/approach.md: does a tube held only by its 12 mm cap stay lined up with the
fingers when it is tilted to the hole's 35 degrees and slid into a hole with 0.9 mm to spare?

What one run does (the programmed arm of Steps 1 and 3; nothing here is learned):

    pick the sample from its rack cell (Step 1's cap grip, fingers down)
    ──► up to the travel height ──► joints turn the hand to the hole's slant, the tube 20 cm (or 10 cm)
        back along the hole's axis ──► slide in along the axis, slowly ──► stop at the bottom
        (the wrist force sensor feels the push back) ──► open ──► back out along the axis ──► up

Two ways to slide (``--mode``):
- ``plain``: straight down the axis; only the push back along the axis is watched (to stop).
- ``force``: also moves away from any sideways push, as a person feels a peg into a hole.

``--miss-mm`` moves the arm's idea of the hole sideways by that much (an error in where it thinks the hole
is, as a camera's would be), to see what the lead-in and the force sensing catch.

Measured from the simulator (the examiner's side; the arm never reads it): the tube's angle in the fingers
after tilting and during the slide, the tip's distance from the hole's bottom, the tube's axis against the
hole's, the largest force felt, and whether the grading code counts the tube as seated.

    cd step-4-load-rotor
    P="pixi run --manifest-path ../simulation-environment/pixi.toml"
    PYTHONPATH=$PWD:$PWD/../step-1-weigh:$PWD/../step-3-fill-balance-tube:$PWD/../simulation-environment \\
        $P python insert_test.py --seed 0 --hole 0 --mode plain
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from pathlib import Path

import mujoco
import numpy as np
from centrifuge_scene import grading, liquid, scene
from centrifuge_scene.settings import IMAGE_SIZE, PAD_TORSION
from motion import GRIP_SQUEEZE, HOLD_CAP, Arm, Blocked
from moves import hand_rot
from PIL import Image

from rotor import BACK, Grip, Loader, grip_at_hole, hole_frame

OUT = Path(__file__).parent / "results"
PADS = ("right_pad1", "right_pad2", "left_pad1", "left_pad2")


def set_pad_torsion(model, torsion: float) -> None:
    """Set the gripper's pads' torsional friction (0: none, the model as built, 3-direction contacts). The
    shared scene sets settings.PAD_TORSION; ``--pad-torsion 0`` is for comparison (docs/insert-test.md)."""
    ids = [model.geom(f"arm/gripper/{p}").id for p in PADS]
    model.geom_condim[ids] = 4 if torsion > 0 else 3
    model.geom_friction[ids, 1] = torsion


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(
        math.acos(float(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))
    )


def picture(model, data, path: Path, camera: str = "rotor") -> None:
    with mujoco.Renderer(model, height=IMAGE_SIZE[1], width=IMAGE_SIZE[0]) as r:
        r.update_scene(data, camera=camera)
        Image.fromarray(r.render()).save(path)


def run(
    seed: int,
    hole: int,
    mode: str,
    miss_mm: float,
    tube_name: str | None,
    pictures: bool,
    squeeze_mm: float,
    torsion: float = PAD_TORSION,
) -> dict:
    t_wall = time.time()
    model, truth = scene.build(seed, awake=True)
    set_pad_torsion(model, torsion)
    data = mujoco.MjData(model)
    scene.reset(model, data)
    scene.settle(model, data, 1.5)
    arm = Arm(model, data)
    name = tube_name or next(iter(truth.samples))
    rack, cell = truth.cells[name]
    body = model.body(name).id
    out: dict = {
        "seed": seed,
        "hole": hole,
        "mode": mode,
        "miss_mm": miss_mm,
        "squeeze_mm": squeeze_mm,
        "pad_torsion": torsion,
        "tube": name,
    }

    def in_fingers() -> float:
        return round(
            angle_deg(data.xmat[body].reshape(3, 3)[:, 2], -hand_rot(arm)[:, 2]), 2
        )

    seat, hole_rot = hole_frame(data, hole)
    axis = hole_rot[:, 2]
    log: list[dict] = []
    worst = [0.0]

    def tick(_loader, s, push, side, shift):
        worst[0] = max(worst[0], in_fingers())
        log.append(
            {
                "target_back_mm": round(s * 1000, 1),
                "tip_to_bottom_mm": round(
                    float(np.linalg.norm(data.xpos[body] - seat)) * 1000, 1
                ),
                "push_N": round(push, 2),
                "side_N": round(side, 2),
                "shift_mm": round(float(np.linalg.norm(shift)) * 1000, 2),
                "in_fingers_deg": in_fingers(),
            }
        )

    loader = Loader(arm, mode, on_tick=tick)
    grip = Grip(capped=True)
    loader.pick(name, rack, cell, grip)
    # Step 1 squeezes the cap only 2 mm (more spins the tube down into the rack's seat); out of the rack a
    # firmer squeeze can be tried.
    if squeeze_mm != GRIP_SQUEEZE * 1000:
        data.ctrl[arm.gripper] = scene.grip_command(HOLD_CAP, squeeze_mm / 1000)
        arm.wait(0.5)
    out["in_fingers_after_pick_deg"] = in_fingers()
    # Where the arm thinks the hole is: off by miss_mm, sideways to the axis (along the rim).
    believed = seat + miss_mm / 1000 * hole_rot[:, 0]
    turn, gap = loader.best_turn(hole, believed, grip)
    out["turn_deg"], out["planned_gap_mm"] = turn, round(gap * 1000, 1)
    start, rot = grip_at_hole(believed + BACK[hole] * axis, hole_rot, turn, grip)
    out["way_in_gap_mm"] = round(loader.go_to_start(start, rot) * 1000, 1)
    out["in_fingers_on_arrival_deg"] = in_fingers()
    arm.wait(1.0)
    out["in_fingers_tilted_deg"] = in_fingers()
    sl = loader.slide(hole, believed, turn, grip)
    out.update(
        slide_stop=sl.stop,
        slide_seconds=sl.seconds,
        pauses=sl.pauses,
        peak_axial_N=round(sl.peak_push_n, 2),
        peak_side_N=round(sl.peak_side_n, 2),
        worst_in_fingers_deg=round(worst[0], 2),
        final_shift_mm=sl.final_shift_mm,
    )
    try:
        loader.let_go(hole)
        out["backed_out"] = True
    except Blocked as e:
        out["backed_out"], out["back_out_error"] = False, str(e)
    arm.wait(1.0)

    # The examiner's look.
    tip = data.xpos[body].copy()
    out["tip_to_bottom_mm"] = round(float(np.linalg.norm(tip - seat)) * 1000, 2)
    out["tip_along_axis_mm"] = round(float((tip - seat) @ axis) * 1000, 2)
    out["axis_vs_hole_deg"] = round(
        angle_deg(data.xmat[body].reshape(3, 3)[:, 2], axis), 2
    )
    out["seated"] = grading.tubes_in_rotor(model, data, truth).get(hole) == name
    out["balls_spilled"] = int(liquid.spilled(model, data, truth.tubes))
    out["sim_blew_up"] = any(
        np.abs(data.xpos[model.body(n).id]).max() > 2.0 for n in truth.tubes
    )
    out["sim_seconds"] = round(arm.time, 1)
    out["wall_seconds"] = round(time.time() - t_wall, 1)
    out["slide_log"] = log
    if pictures:
        OUT.mkdir(exist_ok=True)
        tag = f"seed{seed}_hole{hole}_{mode}_miss{miss_mm:g}"
        with contextlib.suppress(
            Blocked
        ):  # out of the rotor camera's way (only for the picture)
            loader.rise()
        picture(model, data, OUT / f"{tag}_rotor.png")
        picture(model, data, OUT / f"{tag}_front.png", camera="front")
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hole", type=int, default=0)
    p.add_argument("--mode", choices=("plain", "force"), default="plain")
    p.add_argument("--miss-mm", type=float, default=0.0)
    p.add_argument(
        "--squeeze-mm", type=float, default=2.0, help="cap squeeze once out of the rack"
    )
    p.add_argument(
        "--pad-torsion",
        type=float,
        default=PAD_TORSION,
        help="0: the gripper model as built",
    )
    p.add_argument("--tube", default=None, help="tube name (default: the first sample)")
    p.add_argument("--pictures", action="store_true")
    p.add_argument("--verbose", action="store_true", help="print the slide log")
    a = p.parse_args()
    try:
        r = run(
            a.seed,
            a.hole,
            a.mode,
            a.miss_mm,
            a.tube,
            a.pictures,
            a.squeeze_mm,
            a.pad_torsion,
        )
    except Blocked as e:
        r = {
            "seed": a.seed,
            "hole": a.hole,
            "mode": a.mode,
            "miss_mm": a.miss_mm,
            "error": str(e),
        }
    log = r.pop("slide_log", [])
    print(json.dumps(r, indent=1))
    if a.verbose:
        for row in log:
            print(row)
    OUT.mkdir(exist_ok=True)
    name = f"insert_seed{a.seed}_hole{a.hole}_{a.mode}_miss{a.miss_mm:g}_sq{a.squeeze_mm:g}_tor{a.pad_torsion:g}.json"
    (OUT / name).write_text(json.dumps({**r, "slide_log": log}, indent=1))


if __name__ == "__main__":
    main()
