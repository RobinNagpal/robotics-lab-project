"""Save a picture from every camera after the scene has settled.

python -m centrifuge_scene.render --seed 0 --out figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
from PIL import Image

from . import centrifuge, scene
from .settings import IMAGE_SIZE

CAMERAS = ("overhead", "racks_high", "racks_low", "front", "rotor", "arm/wrist")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("figures"))
    parser.add_argument("--settle", type=float, default=0.5, help="seconds of simulation before the pictures")
    parser.add_argument("--closed", action="store_true", help="also save the front view with the lid shut")
    parser.add_argument(
        "--loaded", action="store_true", help="put the samples and a blank in the rotor first"
    )
    args = parser.parse_args()

    model, truth = scene.build(args.seed, awake=args.loaded)
    data = mujoco.MjData(model)
    scene.reset(model, data)
    if args.loaded:
        # Moved by hand, for a picture: the arm does this in the task.
        for k, name in enumerate([*truth.samples, "blank0"][:6]):
            site = data.site(scene.slot_site(k))
            rot = site.xmat.reshape(3, 3).copy()
            scene.move_tube(model, data, name, site.xpos + rot[:, 2] * 0.0005, rot)
    scene.settle(model, data, args.settle)

    args.out.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(model, height=IMAGE_SIZE[1], width=IMAGE_SIZE[0]) as renderer:
        shots = [(cam, cam) for cam in CAMERAS]
        for cam, label in shots:
            renderer.update_scene(data, camera=cam)
            tag = "_loaded" if args.loaded else ""
            path = args.out / f"seed{args.seed}{tag}_{label.replace('/', '_')}.png"
            Image.fromarray(renderer.render()).save(path)
            print(path)
        if args.closed:
            centrifuge.close_lid(model, data)
            renderer.update_scene(data, camera="front")
            path = args.out / f"seed{args.seed}_front_lid_closed.png"
            Image.fromarray(renderer.render()).save(path)
            print(path)


if __name__ == "__main__":
    main()
