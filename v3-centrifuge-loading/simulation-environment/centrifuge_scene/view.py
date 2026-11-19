"""Open the scene in MuJoCo's window. The arm holds its home pose.

    mjpython -m centrifuge_scene.view --seed 0

On macOS MuJoCo's window needs ``mjpython`` rather than ``python``; `make view` does that. Press Tab in the
window to show the collision walls (group 3) and the hole sites (group 4).
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
    print(f"seed {args.seed}: {truth.scenario}")
    for name, s in truth.samples.items():
        print(f"  {name}: {s.balls} balls, {s.mass * 1000:.2f} g")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            start = time.time()
            scene.step(model, data)
            viewer.sync()
            time.sleep(max(0.0, model.opt.timestep - (time.time() - start)))


if __name__ == "__main__":
    main()
