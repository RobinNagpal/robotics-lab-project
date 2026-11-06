"""Save a picture of the capped scene from every camera, after it has settled.

python -m capping.render --seed 150 --out figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
from lab_scene.render import CAMERAS
from lab_scene.settings import IMAGE_SIZE
from PIL import Image

from . import caps, scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=150)
    parser.add_argument("--out", type=Path, default=Path("figures"))
    parser.add_argument("--settle", type=float, default=0.5, help="seconds of simulation before the pictures")
    args = parser.parse_args()

    model, _ = scene.build(args.seed)
    data = mujoco.MjData(model)
    scene.reset(model, data)
    caps.step(model, data, round(args.settle / model.opt.timestep))

    args.out.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(model, height=IMAGE_SIZE[1], width=IMAGE_SIZE[0]) as renderer:
        for cam in CAMERAS:
            renderer.update_scene(data, camera=cam)
            path = args.out / f"seed{args.seed}_caps_{cam.replace('/', '_')}.png"
            Image.fromarray(renderer.render()).save(path)
            print(path)


if __name__ == "__main__":
    main()
