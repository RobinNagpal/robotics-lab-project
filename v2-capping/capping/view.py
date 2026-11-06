"""Open the capped scene in MuJoCo's window. The arms hold their home pose.

    mjpython -m capping.view --seed 150

On macOS MuJoCo's window needs ``mjpython`` rather than ``python``; `make view` does that. Press Tab in
the window to show the collision shapes (group 3).
"""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer

from . import caps, scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=150)
    args = parser.parse_args()

    model, _ = scene.build(args.seed)
    data = mujoco.MjData(model)
    scene.reset(model, data)
    print(f"seed {args.seed}: a screw cap on each of {', '.join(scene.CAPPED)}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            start = time.time()
            caps.step(model, data)
            viewer.sync()
            time.sleep(max(0.0, model.opt.timestep - (time.time() - start)))


if __name__ == "__main__":
    main()
