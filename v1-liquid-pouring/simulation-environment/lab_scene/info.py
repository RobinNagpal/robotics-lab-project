"""Print what is in the scene and how it settles: sizes, weights, balls, levels, speed.

python -m lab_scene.info --seed 0
"""

from __future__ import annotations

import argparse
import time

import mujoco

from . import liquid, scene
from .glassware import inner_volume
from .settings import RECEIVERS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--settle", type=float, default=2.0, help="seconds of simulation to settle")
    args = parser.parse_args()

    start = time.time()
    model, truth = scene.build(args.seed)
    print(f"built in {time.time() - start:.1f} s: {model.nbody} bodies, {model.ngeom} geoms, {model.nv} dofs")
    data = mujoco.MjData(model)
    scene.reset(model, data)
    start = time.time()
    scene.settle(model, data, args.settle)
    wall = time.time() - start
    print(f"settled {args.settle:.1f} s of simulation in {wall:.1f} s ({args.settle / wall:.2f}x real time)")
    print(f"{data.nbody_awake} of {model.nbody} bodies still awake\n")

    widths = {
        "container": -24,
        "size mm": 12,
        "mass g": 7,
        "inside ml": 9,
        "balls": 6,
        "asked": 6,
        "level": 6,
    }
    print(" ".join(f"{h:<{-w}}" if w < 0 else f"{h:>{w}}" for h, w in widths.items()))
    for name, c in truth.containers.items():
        size = f"{2 * c.radius * 1000:.0f} x {c.height * 1000:.0f}"
        ml = inner_volume(c, c.height) * 1e6
        balls = len(liquid.balls_in(model, data, c, name))
        asked = f"{truth.levels[name]:.0%}" if name in truth.levels else "-"
        now = liquid.level(model, data, c, name)
        print(f"{name:24} {size:>12} {c.mass * 1000:7.0f} {ml:9.1f} {balls:6d} {asked:>6} {now:6.0%}")
    print(f"\nballs outside every container: {liquid.spilled(model, data, truth.containers)}")
    print("receivers' 50 % mark, in balls:", {c.name: liquid.ball_count(c, 0.5) for c in RECEIVERS})


if __name__ == "__main__":
    main()
