"""Where the pour camera stands: the scene from above and from the side, with the camera and what it sees.

Both views are drawn with no perspective (orthographic), so distances on the picture are true to scale.

    python -m camera_study.placement --start data/starts/seed150_size200.npy --out OUT.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import mujoco
import numpy as np
from lab_scene.settings import IMAGE_SIZE, POUR_CAMERA
from PIL import Image, ImageDraw, ImageFont

from pour import start

W = 1500  # pixels, width of each view
SCALE = 800  # pixels per metre
POURING_POINT = np.array([0.45, 0.0, 0.28])  # the receiver's middle as Step 1b holds it

INK = (25, 25, 25)
CAM = (215, 40, 40)
VIEW = (255, 200, 40)
SOFT = (255, 255, 255)


def font(size: int):
    return ImageFont.load_default(size=size)


class Ortho:
    """An orthographic view: renders the scene and turns world points into pixels."""

    def __init__(self, model, data, lookat, azimuth, elevation, width, height):
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        model.vis.global_.orthographic = 1
        model.vis.global_.fovy = height / SCALE  # orthographic: the view's height in metres
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = lookat
        self.cam.distance = 4.0
        self.cam.azimuth = azimuth
        self.cam.elevation = elevation
        self.r = mujoco.Renderer(model, height, width)
        self.r.update_scene(data, camera=self.cam)
        self.image = Image.fromarray(self.r.render())
        gl = self.r.scene.camera[0]
        self.pos = np.array(gl.pos)
        self.up = np.array(gl.up)
        self.right = np.cross(np.array(gl.forward), self.up)
        self.w, self.h = width, height

    def px(self, p) -> tuple[float, float]:
        d = np.asarray(p, float) - self.pos
        return self.w / 2 + SCALE * d @ self.right, self.h / 2 - SCALE * d @ self.up


def label(draw, xy, text, size=26, anchor="mm", fill=INK):
    draw.text(xy, text, font=font(size), fill=fill, anchor=anchor, stroke_width=4, stroke_fill=SOFT)


def wedge(draw, view: Ortho, apex, axis, side, half_angle, reach):
    """The camera's field of view as a filled triangle, ``side`` being the across direction in this view."""
    a = math.tan(math.radians(half_angle)) * reach
    far = apex + axis * reach
    pts = [view.px(apex), view.px(far + side * a), view.px(far - side * a)]
    draw.polygon(pts, fill=VIEW + (90,), outline=VIEW + (255,))
    draw.line([view.px(apex), view.px(far)], fill=CAM + (255,), width=2)


def camera_mark(draw, view: Ortho, p, axis, side, size=0.05):
    """A small camera body with its lens pointing along ``axis``."""
    c = [
        p - axis * size + side * size * 0.5,
        p + side * size * 0.5,
        p - side * size * 0.5,
        p - axis * size - side * size * 0.5,
    ]
    draw.polygon([view.px(q) for q in c], fill=CAM + (255,))
    lens = [
        p - side * size * 0.3,
        p + axis * size * 0.5 - side * size * 0.45,
        p + axis * size * 0.5 + side * size * 0.45,
        p + side * size * 0.3,
    ]
    draw.polygon([view.px(q) for q in lens], fill=CAM + (255,))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=Path, default=Path("data/starts/seed150_size200.npy"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    model, data, containers, meta = start.load(a.start)
    cam_p = np.array(POUR_CAMERA.pos)
    xa, ya = np.array(POUR_CAMERA.xyaxes[:3]), np.array(POUR_CAMERA.xyaxes[3:])
    axis = -np.cross(xa, ya)  # where the camera looks
    dist = np.linalg.norm(POURING_POINT - cam_p)
    half_v = POUR_CAMERA.fovy / 2
    half_h = math.degrees(math.atan(math.tan(math.radians(half_v)) * IMAGE_SIZE[0] / IMAGE_SIZE[1]))
    reach = dist + 0.25

    # --- From above: x to the right, y up.
    top = Ortho(model, data, lookat=(0.28, -0.25, 0.0), azimuth=90, elevation=-90, width=W, height=1300)
    rgb = np.array(top.image)
    rgb[rgb.max(axis=2) < 4] = 255  # beyond the floor's edge
    img = Image.fromarray(rgb).convert("RGBA")
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    wedge(d, top, cam_p, axis, xa, half_h, reach)
    camera_mark(d, top, cam_p, axis, xa)
    img = Image.alpha_composite(img, over)
    d = ImageDraw.Draw(img)
    pp = top.px(POURING_POINT)
    d.ellipse([pp[0] - 9, pp[1] - 9, pp[0] + 9, pp[1] + 9], outline=CAM, width=4)
    mid = top.px((cam_p + POURING_POINT) / 2)
    label(d, (mid[0] - 40, mid[1] + 30), f"{dist:.1f} m", 30, fill=CAM)
    label(d, top.px(cam_p + [0.0, -0.09, 0]), "pour camera", 30, fill=CAM)
    label(d, top.px(cam_p + [0.0, -0.135, 0]), f"{2 * half_h:.0f} degrees wide, 0.28 m high", 22, fill=CAM)
    label(d, top.px(POURING_POINT + [0.0, 0.095, 0]), "pouring point", 26, fill=CAM)
    label(d, top.px([-0.10, -0.17, 0]), "Arm B (arm 1)", 28)
    label(d, top.px([-0.10, -0.21, 0]), "holds the source, pours", 20)
    label(d, top.px([1.00, 0.17, 0]), "Arm A (arm 2)", 28)
    label(d, top.px([1.00, 0.13, 0]), "holds the receiver still", 20)
    label(d, top.px(data.body("source_rack").xpos + [0.0, 0.085, 0]), "source rack", 24)
    for name, text in (("empty_rack", "empty rack"), ("destination_rack", "destination rack")):
        label(d, top.px(data.body(name).xpos + [0.13, 0.0, 0]), text, 24, anchor="lm")
    # Axes, in a corner.
    o = np.array([0.80, 0.32, 0.0])
    for v, t in (([0.12, 0, 0], "x"), ([0, 0.12, 0], "y")):
        e = top.px(o + v)
        d.line([top.px(o), e], fill=INK, width=4)
        label(d, (e[0] + (16 if t == "x" else 0), e[1] - (16 if t == "y" else 0)), t, 24)
    # A 50 cm scale bar.
    s0 = np.array([0.55, -0.95, 0.0])
    d.line([top.px(s0), top.px(s0 + [0.5, 0, 0])], fill=INK, width=6)
    label(d, top.px(s0 + [0.25, 0.04, 0]), "50 cm", 24)
    label(d, (24, 24), "From above", 34, anchor="la")
    top_img = img.convert("RGB")

    # --- From the side, across the camera's line of sight: the camera on the left, level with the receiver.
    across = np.cross(axis, [0, 0, 1])  # horizontal, square to the line of sight
    look = -across  # the side view looks along this
    az = math.degrees(math.atan2(look[1], look[0]))
    centre = cam_p + axis * (dist / 2 + 0.1)
    side = Ortho(
        model, data, lookat=(centre[0], centre[1], 0.25), azimuth=az, elevation=0, width=W, height=700
    )
    if side.px(cam_p)[0] > side.px(POURING_POINT)[0]:  # keep the camera on the left
        side = Ortho(
            model,
            data,
            lookat=(centre[0], centre[1], 0.25),
            azimuth=az + 180,
            elevation=0,
            width=W,
            height=700,
        )
    rgb = np.array(side.image)
    rgb[rgb.max(axis=2) < 4] = 255  # the empty sky behind the scene
    img = Image.fromarray(rgb).convert("RGBA")
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    wedge(d, side, cam_p, axis, np.array([0.0, 0.0, 1.0]), half_v, reach)
    camera_mark(d, side, cam_p, axis, np.array([0.0, 0.0, 1.0]))
    # The camera's post, down to the floor under the table's corner.
    d.line([side.px(cam_p - [0, 0, 0.03]), side.px([cam_p[0], cam_p[1], 0.0])], fill=CAM + (255,), width=8)
    img = Image.alpha_composite(img, over)
    d = ImageDraw.Draw(img)
    view_h = 2 * dist * math.tan(math.radians(half_v))
    q = POURING_POINT
    tp, bt = side.px(q + [0, 0, view_h / 2]), side.px(q - [0, 0, view_h / 2])
    d.line([(tp[0] + 70, tp[1]), (bt[0] + 70, bt[1])], fill=CAM, width=4)
    for y in (tp[1], bt[1]):
        d.line([(tp[0] + 58, y), (tp[0] + 82, y)], fill=CAM, width=4)
    label(d, (tp[0] + 92, (tp[1] + bt[1]) / 2), f"{view_h * 100:.0f} cm\nin view", 24, anchor="lm", fill=CAM)
    label(d, side.px(cam_p + [0, 0, 0.09]), f"pour camera, {2 * half_v:.1f} degrees tall", 26, fill=CAM)
    label(
        d,
        side.px((cam_p + q) / 2 + [0, 0, -0.06]),
        f"{dist:.1f} m, level with the receiver's middle",
        24,
        fill=CAM,
    )
    label(d, side.px([cam_p[0], cam_p[1], 0.06]) + np.array([14, 0]), "post", 22, anchor="lm", fill=CAM)
    label(d, (24, 24), "From the side", 34, anchor="la")
    side_img = img.convert("RGB")

    sheet = Image.new("RGB", (W, top_img.height + side_img.height + 10), (255, 255, 255))
    sheet.paste(top_img, (0, 0))
    sheet.paste(side_img, (0, top_img.height + 10))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(a.out)
    print(f"saved {a.out}: the receiver is {meta['receiver']}, scene {meta['seed']}")


if __name__ == "__main__":
    main()
