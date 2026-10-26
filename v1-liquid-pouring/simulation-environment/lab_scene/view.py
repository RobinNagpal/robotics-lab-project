"""Open the scene in MuJoCo's window. The arm holds its home pose while the liquid settles.

    mjpython -m lab_scene.view --seed 0

On macOS MuJoCo's window needs ``mjpython`` rather than ``python``; `make view`
does that. Press Tab in the window to show the collision walls (group 3).
"""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer

from . import scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model, truth = scene.build(args.seed)
    data = mujoco.MjData(model)
    scene.reset(model, data)
    print(f"seed {args.seed}: {sum(truth.balls.values())} balls")
    for name, level in truth.levels.items():
        print(f"  {name}: filled to {level:.0%}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            # Never faster than real time; slower when there are many awake balls.
            time.sleep(max(0.0, model.opt.timestep - (time.time() - start)))


if __name__ == "__main__":
    main()
