"""Pictures and short clips of the ported AutoBio uncapping, for docs/autobio-uncapping.md.

Writes into docs/images/autobio/:
- ``setup.png``: both arms holding the tube, with the parts named;
- ``task.gif``: one whole run (their expert, loose cap), about 3 times real speed, with the step named;
- ``turning.gif``: close up, the first two rounds of turning, with the hand's and the cap's turn counted;
- ``tight.gif``: the same with our tight cap: the hand turns, the cap does not.

    python -m autobio_uncap.media
"""

from __future__ import annotations

import contextlib
import math
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .expert import Expert, Options

OUT = Path(__file__).resolve().parents[1] / "docs" / "images" / "autobio"
W, H = 480, 360
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"


def _font(size: int):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default(size=size)


class Camera:
    """A free camera and a way to find where a 3D point lands in its picture."""

    def __init__(self, e: Expert, lookat, distance: float, azimuth: float = 150, elevation: float = -20):
        self.e = e
        self.r = mujoco.Renderer(e.model, H, W)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = lookat
        self.cam.distance, self.cam.azimuth, self.cam.elevation = distance, azimuth, elevation

    def image(self) -> Image.Image:
        self.r.update_scene(self.e.data, camera=self.cam)
        return Image.fromarray(self.r.render())

    def project(self, p) -> tuple[float, float]:
        # The scene keeps two eye cameras (for stereo); a plain picture is drawn from their midpoint.
        c0, c1 = self.r.scene.camera[0], self.r.scene.camera[1]
        fwd, up = np.array(c0.forward), np.array(c0.up)
        right = np.cross(fwd, up)
        v = np.asarray(p) - (np.array(c0.pos) + np.array(c1.pos)) / 2
        x, y, z = v @ right, v @ up, v @ fwd
        half_h = math.tan(math.radians(self.e.model.vis.global_.fovy) / 2)
        half_w = half_h * W / H
        return W / 2 * (1 + x / z / half_w), H / 2 * (1 - y / z / half_h)


def caption(img: Image.Image, text: str, lines: list[str] | None = None) -> Image.Image:
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([0, H - 34, W, H], fill=(0, 0, 0, 170))
    d.text((10, H - 28), text, font=_font(18), fill=(255, 255, 255))
    for i, line in enumerate(lines or []):
        d.rectangle([0, 6 + 26 * i, 250, 30 + 26 * i], fill=(0, 0, 0, 150))
        d.text((8, 9 + 26 * i), line, font=_font(17), fill=(255, 255, 255))
    return img


def hand_angle(e: Expert) -> float:
    """How far arm 1's hand has turned about the vertical (degrees, anticlockwise from above)."""
    y = e.data.site_xmat[e.arm1.site_id].reshape(3, 3)[:, 1]
    return math.degrees(math.atan2(y[1], y[0]))


def save_gif(frames: list[Image.Image], path: Path, fps: float) -> None:
    pal = [f.convert("P", palette=Image.Palette.ADAPTIVE, colors=128) for f in frames]
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=round(1000 / fps), loop=0, optimize=True)
    print(path, f"{path.stat().st_size / 1e6:.1f} MB, {len(frames)} frames")


def setup_picture() -> None:
    e = Expert(Options(turns=1.35))
    e.reset(0)
    e.hand_over()
    cam = Camera(e, e.data.xpos[e.tube] + [0, 0, 0.07], 0.42)
    img = cam.image()
    d = ImageDraw.Draw(img, "RGBA")
    f = _font(17)
    marks = [
        ("Arm 1 holds the cap, from above", e.data.xpos[e.thread.cap] + [0, 0, 0.03], (190, 12)),
        ("The cap (screwed on)", e.data.xpos[e.thread.cap] - [0, 0, 0.006], (300, 120)),
        ("Arm 2 holds the tube, from the side", e.data.site_xpos[e.arm2.site_id], (10, 300)),
        ("The tube (50 ml)", e.data.xpos[e.tube] + [0, 0, 0.03], (330, 250)),
    ]
    for text, p, (tx, ty) in marks:
        px, py = cam.project(p)
        tw = d.textlength(text, font=f)
        d.rectangle([tx - 4, ty - 3, tx + tw + 4, ty + 21], fill=(255, 255, 255, 220))
        d.text((tx, ty), text, font=f, fill=(0, 0, 0))
        ax, ay = (tx + tw / 2, ty + 21) if ty < py else (tx + tw / 2, ty - 3)
        d.line([ax, ay, px, py], fill=(255, 210, 0), width=2)
        d.ellipse([px - 4, py - 4, px + 4, py + 4], fill=(255, 210, 0))
    img.save(OUT / "setup.png")
    print(OUT / "setup.png")


def task_clip() -> None:
    e = Expert(Options(turns=1.35))
    cam = Camera(e, [0.0, 0.0, 0.97], 0.95)
    frames = []
    every = round(0.25 / e.dt)

    def cb(e):
        if round(e.data.time / e.dt) % every == 0:
            frames.append(caption(cam.image(), e.phase))

    e.on_step = cb
    e.run(0)
    for _ in range(8):
        frames.append(caption(cam.image(), "Done: the cap is off, the tube is still held"))
    save_gif(frames, OUT / "task.gif", fps=12)


class _Stop(Exception):
    pass


def turning_clip(tight: bool, name: str) -> None:
    """Close up from the first turn to the start of the third round."""
    e = Expert(Options(turns=1.35, tight=tight))
    frames = []
    every = round(0.1 / e.dt)
    state = {}

    def cb(e):
        if e.phase.startswith("Round 3"):
            raise _Stop
        if not (e.phase.startswith("Round") or e.phase.startswith("Arm 1 lets go")):
            return
        if not state:
            state["start"] = hand_angle(e)
            state["cam"] = Camera(e, e.data.xpos[e.thread.cap_body(e.data)] - [0, 0, 0.01], 0.3, 150, -15)
        if round(e.data.time / e.dt) % every == 0:
            hand = (hand_angle(e) - state["start"] + 180) % 360 - 180
            cap = e.thread.turned(e.data) * 360
            counts = [f"hand angle:  {hand:+4.0f} deg", f"cap turned:  {cap:4.0f} deg"]
            frames.append(caption(state["cam"].image(), e.phase, counts))

    e.reset(0)
    e.hand_over()
    e.on_step = cb
    with contextlib.suppress(_Stop):
        e.unscrew()
    save_gif(frames, OUT / name, fps=20)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    setup_picture()
    task_clip()
    turning_clip(False, "turning.gif")
    turning_clip(True, "tight.gif")


if __name__ == "__main__":
    main()
