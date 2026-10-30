"""Draw the pictures used in the docs, into docs/images/.

    python -m carry.figures

One run (scene 0, medium tubes), seen from the front camera each time an arm
sends a message, plus the overhead picture with what perception found.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
from lab_scene.settings import FRONT
from PIL import Image, ImageDraw, ImageFont

from . import episode
from .perceive import Perception, overhead

OUT = Path("docs/images")
FONT = ImageFont.load_default(size=16)
WHITE = (255, 255, 255)
COLOURS = {"source_rack": (60, 200, 90), "empty_rack": (70, 140, 255), "destination_rack": (255, 210, 0)}


def _caption(img: Image.Image, text: str) -> Image.Image:
    out = Image.new("RGB", (img.width, img.height + 28), (20, 20, 20))
    out.paste(img, (0, 28))
    ImageDraw.Draw(out).text((8, 5), text, fill=WHITE, font=FONT)
    return out


def _grid(images: list[Image.Image], columns: int, gap: int = 8) -> Image.Image:
    w, h = images[0].size
    rows = math.ceil(len(images) / columns)
    out = Image.new("RGB", (columns * w + (columns - 1) * gap, rows * h + (rows - 1) * gap), WHITE)
    for i, img in enumerate(images):
        out.paste(img, ((i % columns) * (w + gap), (i // columns) * (h + gap)))
    return out


def perception_figure(perception: Perception) -> None:
    from lab_scene import scene as lab

    model, _ = lab.build(0)
    data = mujoco.MjData(model)
    lab.reset(model, data)
    lab.settle(model, data, 0.5)
    table = perception.table(model, data)
    rgb, _, calib = overhead(model, data)
    img = Image.fromarray(rgb)
    d = ImageDraw.Draw(img)
    small, big = ImageFont.load_default(size=18), ImageFont.load_default(size=24)
    points = []
    for role, xy in table.racks.items():
        u, v = calib.project(np.array([[xy[0], xy[1], 0.06]]))[0]
        points.append((u, v))
        text = role.replace("_", " ")
        below = role != "destination_rack"  # it stands right beside the empty rack
        d.text(
            (u - d.textlength(text, font=big) / 2, v + 150 if below else v - 180),
            text,
            fill=COLOURS[role],
            font=big,
        )
    for t in table.tubes:
        u, v = calib.project(np.array([[t.xy[0], t.xy[1], t.top]]))[0]
        colour = COLOURS[t.rack] if t.rack else WHITE
        d.ellipse([u - 18, v - 18, u + 18, v + 18], outline=colour, width=4)
        label = f"{int(round(t.size * 1000))} mm" + (" + liquid" if t.liquid else "")
        # Beside the circle, on the side away from the other racks.
        x = u - 26 - d.textlength(label, font=small) if t.rack == "source_rack" else u + 26
        d.text((x, v - 10), label, fill=colour, font=small)
    us, vs = zip(*points, strict=True)
    box = (int(min(us)) - 260, int(min(vs)) - 230, int(max(us)) + 200, int(max(vs)) + 230)
    img = img.crop(box)
    img.save(OUT / "perception.png")


def run_figure(perception: Perception) -> dict:
    frames: list[Image.Image] = []
    renderer = None
    a, b = episode.RECEIVER_ARM, episode.SOURCE_ARM
    words = {
        (b, "picked"): "Arm B: source picked",
        (b, "waiting"): "Arm B: parked, waiting",
        (a, "picked"): "Arm A: receiver picked",
        (a, "ready"): "Arm A: holding at the pouring point, ready",
        (b, "in place"): "Arm B: in place beside the receiver",
    }

    def heard(coordinator, arm, message):
        nonlocal renderer
        if renderer is None:
            renderer = mujoco.Renderer(coordinator.model, 480, 640)
        renderer.update_scene(coordinator.data, camera=FRONT.name)
        t = coordinator.data.time
        frames.append(_caption(Image.fromarray(renderer.render()), f"{t:4.1f} s  {words[(arm, message)]}"))

    result = episode.run_episode(0, [0.15, 0.2, 0.1], perception, episode.load_net(), heard=heard)
    _grid(frames, 3).save(OUT / "run.png")
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    perception = Perception()
    perception_figure(perception)
    result = run_figure(perception)
    print(f"pictures written to {OUT}/; the run: done={result['done']}")


if __name__ == "__main__":
    main()
