"""The pictures in the docs: what the arm knows before it looks, where the cameras are and what each sees,
and how an outline becomes a tube's pose. The camera pictures go to the main docs (v3's docs/images/).

    cd step-0-find
    PYTHONPATH=$PWD:$PWD/../simulation-environment \
      pixi run --manifest-path ../simulation-environment/pixi.toml -e sam python figures.py

Uses SAM 3's saved outlines (results/sam_raw/), so run evaluate_pose.py --outlines sam3 first.
"""

from __future__ import annotations

import io
from pathlib import Path

import mujoco
import numpy as np
from centrifuge_scene import scene
from centrifuge_scene.racks import cell_centres
from centrifuge_scene.settings import LAYOUT, RACK, SCALE
from PIL import Image, ImageDraw, ImageFont

import pose
from bench import take_pictures
from evaluate_pose import CAMERAS, knocked_scene, true_tubes

OUT = Path(__file__).parent / "docs" / "images"
MAIN = Path(__file__).resolve().parent.parent / "docs" / "images"
# One colour per camera, the same in every picture.
CAMERA_COLOURS = {
    "overhead": (255, 210, 60),
    "racks_high": (60, 200, 255),
    "racks_low": (255, 110, 200),
    "front": (190, 190, 190),
    "rotor": (140, 255, 120),
    "arm/wrist": (255, 150, 60),
}
GREEN = (40, 220, 90)
COLOURS = [
    (255, 80, 80),
    (255, 200, 40),
    (80, 200, 255),
    (200, 120, 255),
    (255, 140, 40),
    (120, 255, 160),
    (255, 120, 200),
    (160, 160, 255),
    (220, 255, 80),
]


def font(size: int):
    return ImageFont.load_default(size=size)


def known() -> None:
    """The overhead picture, with what the arm knows marked in green and a ? on every cell."""
    views, (model, data, truth) = take_pictures(0, ["overhead"])
    cam = views["overhead"].camera
    img = Image.open(io.BytesIO(views["overhead"].png)).convert("RGB").resize((1280, 960))
    s = 2  # the picture is drawn at twice its size
    d = ImageDraw.Draw(img)
    for rack in ("sample_rack", "supply_rack"):
        for x, y in cell_centres(getattr(LAYOUT, rack)):
            u, v, _ = cam.project(np.array([x, y, RACK.height]))
            d.ellipse([s * u - 15, s * v - 15, s * u + 15, s * v + 15], outline=GREEN, width=3)
            d.text(
                (s * u, s * v),
                "?",
                fill=(255, 255, 255),
                font=font(22),
                anchor="mm",
                stroke_width=2,
                stroke_fill=(0, 0, 0),
            )
    u, v, _ = cam.project(np.array([*LAYOUT.scale, 2 * SCALE.base[2] + SCALE.holder_height]))
    d.rectangle([s * u - 22, s * v - 22, s * u + 22, s * v + 22], outline=GREEN, width=3)
    for k in range(6):
        u, v, _ = cam.project(data.site_xpos[model.site(scene.entry_site(k)).id])
        d.ellipse([s * u - 16, s * v - 16, s * u + 16, s * v + 16], outline=GREEN, width=3)
    labels = [
        ((150, 330), "racks: 16 cells (known)\n? = tube or not, and what kind"),
        ((470, 300), "balance holder (known)"),
        ((790, 120), "rotor: 6 holes, 35° slant (known)"),
    ]
    for xy, text in labels:
        d.multiline_text(xy, text, fill=GREEN, font=font(24), stroke_width=3, stroke_fill=(0, 0, 0))
    OUT.mkdir(parents=True, exist_ok=True)
    img.save(OUT / "known.png")


def to_top(p: np.ndarray, box=((0.08, 0.66), (0.04, 0.66)), size=480) -> tuple[float, float]:
    """A world point (x, y) on a top-down map drawn like the overhead picture (x up, y to the left)."""
    (x0, x1), (y0, y1) = box
    scale = size / max(x1 - x0, y1 - y0)
    return (y1 - p[1]) * scale, (x1 - p[0]) * scale


def pose_pipeline(seed: int = 0) -> None:
    """Three panels: SAM 3's outlines on the high camera's picture; its depth picture; the points from all
    three cameras seen from above, with the fitted tubes (coloured) and the simulator's true axes (white)."""
    from sam import TUBE_WORD, Sam3

    views, (model, data, truth), _ = knocked_scene(seed)
    sam = Sam3()
    masks = {c: sam.outlines(views[c].png, TUBE_WORD)[0] for c in CAMERAS}
    per_camera = {c: pose.poses(views[c], masks[c], c) for c in CAMERAS}
    found = pose.merge(per_camera)

    high = views["racks_high"]
    a = Image.open(io.BytesIO(high.png)).convert("RGB")
    tint = np.array(a).astype(float)
    for i, m in enumerate(masks["racks_high"]):
        tint[m] = 0.45 * tint[m] + 0.55 * np.array(COLOURS[i % len(COLOURS)])
    a = Image.fromarray(tint.astype(np.uint8))

    depth = high.depth.copy()
    near, far = np.percentile(depth, 1), np.percentile(depth, 99)
    b = Image.fromarray((255 * (1 - np.clip((depth - near) / (far - near), 0, 1))).astype(np.uint8)).convert(
        "RGB"
    )

    c = Image.new("RGB", (480, 480), (25, 28, 34))
    dc = ImageDraw.Draw(c)
    for p in found:
        for q in p.points[:: max(1, len(p.points) // 1500)]:
            u, v = to_top(q)
            dc.point((u, v), fill=(120, 125, 135))
    for t in true_tubes(model, data, truth):
        dc.line([to_top(t.tip), to_top(t.top)], fill=(255, 255, 255), width=1)
    for i, p in enumerate(found):
        col = COLOURS[i % len(COLOURS)]
        dc.line([to_top(p.ends[0]), to_top(p.ends[1])], fill=col, width=3)
        for e in p.ends:
            u, v = to_top(e)
            dc.ellipse([u - 4, v - 4, u + 4, v + 4], fill=col)

    w, h = 640, 480
    sheet = Image.new("RGB", (w * 2 + 480 + 40, h + 50), (255, 255, 255))
    for x, img in ((0, a), (w + 20, b), (2 * w + 40, c)):
        sheet.paste(img, (x, 50))
    d = ImageDraw.Draw(sheet)
    titles = [
        (0, '1. SAM 3\'s outlines ("test tube"), high camera'),
        (w + 20, "2. Depth picture: brighter = nearer"),
        (2 * w + 40, "3. Points from 3 cameras, from above; fitted tubes"),
    ]
    for x, text in titles:
        d.text((x + 6, 14), text, fill=(0, 0, 0), font=font(20))
    dc2 = ImageDraw.Draw(sheet)
    dc2.text(
        (2 * w + 48, 56),
        "grey: points   colour: fitted tube   white: true axis",
        fill=(220, 220, 220),
        font=font(15),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT / "pose.png")


def camera_map() -> None:
    """The overhead picture with every camera's place (its spot on the bench below it), its height, and the
    way it looks (its field of view, flattened onto the bench)."""
    import math

    from bench import cameras

    model, _ = scene.build(0)
    data = mujoco.MjData(model)
    scene.reset(model, data)
    scene.settle(model, data, 0.3)
    cams = cameras(model, data, list(CAMERA_COLOURS))
    top = cams["overhead"]
    with mujoco.Renderer(model, height=top.height, width=top.width) as r:
        r.update_scene(data, camera="overhead")
        img = Image.fromarray(r.render()).resize((1280, 960))
    # Leave room round the picture: two cameras stand at its top edge, one just outside it.
    pad = 110
    sheet = Image.new("RGB", (1280 + 2 * pad, 960 + 2 * pad), (30, 33, 40))
    sheet.paste(img, (pad, pad))
    d = ImageDraw.Draw(sheet)

    def px(p):
        u, v, _ = top.project(np.array([p[0], p[1], 0.0]))
        return pad + 2 * u, pad + 2 * v

    for name, cam in cams.items():
        col = CAMERA_COLOURS[name]
        x, y = px(cam.pos)
        forward = -cam.rot[:, 2]
        if name == "front":
            pass  # for people watching; its view is the scene picture at the top of the docs
        elif name != "overhead":
            flat = np.array([forward[0], forward[1], 0.0])
            flat /= np.linalg.norm(flat)
            half = math.atan(math.tan(math.radians(cam.fovy) / 2) * cam.width / cam.height)
            # As far as the middle of the view reaches the bench, and a little more.
            reach = 1.15 * cam.pos[2] / max(-forward[2], 0.05) * float(np.hypot(forward[0], forward[1]))
            reach = min(reach, 0.9) if name != "arm/wrist" else 0.15
            for a in (-half, half):
                c, s_ = math.cos(a), math.sin(a)
                ray = np.array([c * flat[0] - s_ * flat[1], s_ * flat[0] + c * flat[1], 0.0])
                d.line([(x, y), px(cam.pos + reach * ray)], fill=col, width=3)
        else:
            d.ellipse([x - 70, y - 70, x + 70, y + 70], outline=col, width=3)
        d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=col, outline=(0, 0, 0), width=2)
        label = {
            "overhead": "overhead\n1.25 m up, straight down",
            "racks_high": f"racks_high\n{cam.pos[2]:.2f} m up",
            "racks_low": f"racks_low\n{cam.pos[2]:.2f} m up",
            "front": f"front (for people)\n{cam.pos[2]:.2f} m up",
            "rotor": f"rotor\n{cam.pos[2]:.2f} m up",
            "arm/wrist": "wrist\n(moves with the arm)",
        }[name]
        offset = {
            "racks_high": (14, -64),
            "racks_low": (14, 6),
            "front": (-250, -60),
            "overhead": (-120, 82),
            "rotor": (14, 14),
        }
        dx, dy = offset.get(name, (14, 10))
        d.multiline_text(
            (x + dx, y + dy), label, fill=col, font=font(24), stroke_width=3, stroke_fill=(0, 0, 0)
        )
    MAIN.mkdir(parents=True, exist_ok=True)
    sheet.save(MAIN / "camera-map.png")


def same_tube(seed: int = 0) -> None:
    """One sample tube as each Step 0 camera sees it: the same 120 x 120 pixel window round it in each one,
    enlarged 3 times, so the pixels are as big as the camera gives them."""
    views, (model, data, truth) = take_pictures(seed, CAMERAS)
    geom = int(mujoco.mjtObj.mjOBJ_GEOM)
    bodies = {}
    with mujoco.Renderer(model, height=480, width=640) as r:
        r.enable_segmentation_rendering()
        for cam in CAMERAS:
            r.update_scene(data, camera=cam)
            seg = r.render()
            ids = np.where(seg[..., 1] == geom, seg[..., 0], -1)
            bodies[cam] = np.where(ids >= 0, model.geom_bodyid[np.clip(ids, 0, None)], -1)
    # The sample every camera sees best (none in front of it), so the panels compare views, not hiding.
    samples = [t for t in truth.tubes if truth.capped[t]]
    name = max(samples, key=lambda t: min((bodies[c] == model.body(t).id).sum() for c in CAMERAS[1:]))
    tube = next(t for t in true_tubes(model, data, truth) if t.name == name)
    panels = []
    for cam in CAMERAS:
        view = views[cam]
        body = bodies[cam]
        pixels = int((body == model.body(name).id).sum())
        u, v, _ = view.camera.project(tube.middle)
        u = int(np.clip(u, 60, 580))
        v = int(np.clip(v, 60, 420))
        box = (u - 60, v - 60, u + 60, v + 60)
        crop = np.array(
            Image.open(io.BytesIO(view.png)).convert("RGB").crop(box).resize((360, 360), Image.NEAREST)
        )
        # The tube's true outline, in yellow, so it is clear which tube this is.
        mask = Image.fromarray((body == model.body(name).id).astype(np.uint8) * 255).crop(box)
        mask = np.array(mask.resize((360, 360), Image.NEAREST)) > 0
        inner = mask.copy()
        for _ in range(2):
            shrunk = inner.copy()
            shrunk[1:, :] &= inner[:-1, :]
            shrunk[:-1, :] &= inner[1:, :]
            shrunk[:, 1:] &= inner[:, :-1]
            shrunk[:, :-1] &= inner[:, 1:]
            inner = shrunk
        crop[mask & ~inner] = (255, 230, 0)
        panels.append((cam, Image.fromarray(crop), pixels))
    notes = {
        "overhead": "small, and the cap hides most of the liquid",
        "racks_high": "cap and tube, nothing in front",
        "racks_low": "the liquid, through the rack's side",
    }
    sheet = Image.new("RGB", (3 * 360 + 40, 360 + 90), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for i, (cam, img, pixels) in enumerate(panels):
        x = i * 380
        sheet.paste(img, (x, 50))
        d.text((x + 4, 6), cam, fill=(0, 0, 0), font=font(22))
        d.text((x + 4, 30), f"the outlined tube: {pixels} pixels", fill=(90, 90, 90), font=font(16))
        d.text((x + 4, 418), notes[cam], fill=(0, 0, 0), font=font(18))
    MAIN.mkdir(parents=True, exist_ok=True)
    sheet.save(MAIN / "same-tube.png")


if __name__ == "__main__":
    camera_map()
    same_tube()
    known()
    pose_pipeline()
