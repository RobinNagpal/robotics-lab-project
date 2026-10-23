"""Save a picture from every camera after the liquid has settled.

python -m lab_scene.render --seed 0 --out figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
from PIL import Image

from . import scene
from .settings import IMAGE_SIZE

CAMERAS = ("front", "overhead", "arm1/wrist", "arm2/wrist")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("figures"))
    parser.add_argument("--settle", type=float, default=0.5, help="seconds of simulation before the pictures")
    args = parser.parse_args()

    model, _ = scene.build(args.seed)
    data = mujoco.MjData(model)
    scene.reset(model, data)
    scene.settle(model, data, args.settle)

    args.out.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(model, height=IMAGE_SIZE[1], width=IMAGE_SIZE[0]) as renderer:
        for cam in CAMERAS:
            renderer.update_scene(data, camera=cam)
            path = args.out / f"seed{args.seed}_{cam.replace('/', '_')}.png"
            Image.fromarray(renderer.render()).save(path)
            print(path)


if __name__ == "__main__":
    main()
