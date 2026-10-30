"""Draw the pictures used in the docs, into docs/top-down/images/.

    python -m grasp.figures --run runs/grconv

Needs the test data, the trained weights and the evaluation (runs/<name>/evaluation.json).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import torch
from detect_classify.camera import Calibration
from PIL import Image, ImageDraw, ImageFont

from . import pick, rules, trial
from .net import GraspNet
from .train import device

OUT = Path("docs/top-down/images")
SCALE = 3  # each 128-pixel picture is drawn 384 pixels wide
GREEN, RED, YELLOW, BLUE, WHITE = (60, 200, 90), (230, 60, 50), (255, 210, 0), (70, 140, 255), (255, 255, 255)
FONT = ImageFont.load_default(size=15)


def _grey(heights: np.ndarray) -> Image.Image:
    g = np.clip(heights / 0.25 * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(g).convert("RGB").resize((128 * SCALE, 128 * SCALE), Image.NEAREST)


def _xy(u: float, v: float) -> tuple[float, float]:
    return ((u + 0.5) * SCALE, (v + 0.5) * SCALE)


def _turn(
    draw: ImageDraw.ImageDraw, u: float, v: float, yaw: float, length: float, colour, width: int
) -> None:
    """A line through (u, v) along a turn; the picture's up is the table's +y."""
    cx, cy = _xy(u, v)
    dx, dy = math.cos(yaw) * length, -math.sin(yaw) * length
    draw.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], fill=colour, width=width)


def _fingers(draw: ImageDraw.ImageDraw, u: float, v: float, yaw: float, colour) -> None:
    """The two open fingers seen from above, roughly: 22 mm wide pads 36 to 58 mm out."""
    px = 1 / 0.0016 * SCALE  # pixels per metre near the rim's height
    along = np.array([math.cos(yaw), -math.sin(yaw)])
    across = np.array([along[1], -along[0]])
    centre = np.array(_xy(u, v))
    for side in (1, -1):
        corners = [
            centre + side * along * a * px + across * c * px
            for a, c in ((0.036, -0.011), (0.058, -0.011), (0.058, 0.011), (0.036, 0.011))
        ]
        draw.polygon([tuple(p) for p in corners], outline=colour, width=3)


def _caption(img: Image.Image, text: str) -> Image.Image:
    out = Image.new("RGB", (img.width, img.height + 26), (20, 20, 20))
    out.paste(img, (0, 26))
    ImageDraw.Draw(out).text((8, 5), text, fill=WHITE, font=FONT)
    return out


def _row(images: list[Image.Image], gap: int = 8) -> Image.Image:
    w = sum(i.width for i in images) + gap * (len(images) - 1)
    out = Image.new("RGB", (w, max(i.height for i in images)), (255, 255, 255))
    x = 0
    for i in images:
        out.paste(i, (x, 0))
        x += i.width + gap
    return out


def _target(sample: dict) -> dict:
    return next(t for t in sample["tubes"] if t["name"] == sample["target"])


def labels_figure(samples: list[dict], heights: np.ndarray) -> None:
    """How a picture is labelled: the fingers at a clear and a blocked turn, and all 18 turns."""
    mixed = next(i for i, s in enumerate(samples) if 5 <= sum(_target(s)["labels"]) <= 10)
    boxed = next(i for i, s in enumerate(samples) if not any(_target(s)["labels"]))
    free = next(i for i, s in enumerate(samples) if all(_target(s)["labels"]))
    t = _target(samples[mixed])
    lab = np.array(t["labels"])
    a = _grey(heights[mixed])
    d = ImageDraw.Draw(a)
    _fingers(d, t["u"], t["v"], rules.bin_yaw(int(np.argmax(lab))), GREEN)
    _fingers(d, t["u"], t["v"], rules.bin_yaw(int(np.argmin(lab))), RED)
    panels = [_caption(a, "open fingers: a clear turn, a blocked turn")]
    for i, text in (
        (mixed, "same tube: all 18 turns"),
        (boxed, "boxed in: no clear turn"),
        (free, "open: every turn clear"),
    ):
        t = _target(samples[i])
        img = _grey(heights[i])
        d = ImageDraw.Draw(img)
        for k, ok in enumerate(t["labels"]):
            _turn(d, t["u"], t["v"], rules.bin_yaw(k), 45 if ok else 25, GREEN if ok else RED, 3)
        panels.append(_caption(img, text))
    _row(panels).save(OUT / "labels.png")


def outputs_figure(samples, heights, scores, rows) -> None:
    """What the model says for a few test tubes, next to the truth."""
    picks = []
    for want in ("mixed", "boxed", "programmed_wrong"):
        for r in rows:
            n = r["clear_turns"]
            if want == "mixed" and 3 <= n <= 8 and r["model"] and r["model"]["good"]:
                picks.append(r["i"])
                break
            if want == "boxed" and n == 0 and r["model"] is None:
                picks.append(r["i"])
                break
            if (
                want == "programmed_wrong"
                and r["programmed"]
                and not r["programmed"]["good"]
                and r["model"]
                and r["model"]["good"]
            ):
                picks.append(r["i"])
                break
    top, bottom = [], []
    for i in picks:
        s, t = samples[i], _target(samples[i])
        truth = _grey(heights[i])
        d = ImageDraw.Draw(truth)
        for k, ok in enumerate(t["labels"]):
            _turn(d, t["u"], t["v"], rules.bin_yaw(k), 45 if ok else 25, GREEN if ok else RED, 3)
        top.append(_caption(truth, "truth: clear (green) and blocked (red) turns"))
        calib = Calibration.from_dict(s["camera"])
        grasp, (u, v, k) = pick.learned(scores[i], heights[i], calib)
        out = _grey(heights[i])
        d = ImageDraw.Draw(out)
        for b in range(rules.BINS):
            sc = float(scores[i][b, v, u])
            _turn(d, u, v, rules.bin_yaw(b), 10 + 40 * sc, (int(255 * (1 - sc)), int(255 * sc), 80), 3)
        if grasp is not None:
            _turn(d, u, v, grasp.yaw, 55, YELLOW, 5)
        prog = rows[i]["programmed"]
        if prog is not None:
            pu, pv = calib.project(np.array([[*prog["xy"], prog["top"]]]))[0]
            _turn(d, pu, pv, prog["yaw"], 55, BLUE, 3)
        bottom.append(_caption(out, "model: score per turn (longer = better)"))
    grid = _row(top)
    under = _row(bottom)
    out = Image.new("RGB", (grid.width, grid.height + under.height + 8), (255, 255, 255))
    out.paste(grid, (0, 0))
    out.paste(under, (0, grid.height + 8))
    out.save(OUT / "outputs.png")


def score_map_figure(samples, heights, scores) -> None:
    """The best score at each pixel, over all turns: high on the middle of tubes that have a clear turn."""
    i = next(i for i, s in enumerate(samples) if len(s["tubes"]) >= 5)
    best = scores[i].max(axis=0)
    pic = _grey(heights[i])
    # Yellow where the score is high, over a dimmed picture.
    base = np.asarray(pic, np.float32) * 0.35
    glow = np.asarray(
        Image.fromarray((255 * best).astype(np.uint8)).resize(pic.size, Image.BILINEAR), np.float32
    )
    glow = (glow / 255)[..., None]
    over = Image.fromarray(np.clip(base * (1 - glow) + glow * np.array(YELLOW), 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(over)
    for t in samples[i]["tubes"]:
        cx, cy = _xy(t["u"], t["v"])
        colour = GREEN if any(t["labels"]) else RED
        d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], outline=colour, width=2)
    _row(
        [
            _caption(pic, "input: height above the table"),
            _caption(over, "output: best score at each pixel"),
        ]
    ).save(OUT / "score_map.png")


def trial_figure(rows) -> None:
    """Four moments of one grasp with full physics."""
    r = next(r for r in rows if r["model"] and r["model"].get("physics", {}).get("success"))
    g = r["model"]
    frames = []
    renderer = None
    cam = mujoco.MjvCamera()

    def seen(stage, model, data):
        nonlocal renderer
        if renderer is None:
            renderer = mujoco.Renderer(model, 360, 480)
        cam.lookat[:] = [g["xy"][0], g["xy"][1], 0.12]
        cam.distance, cam.azimuth, cam.elevation = 0.55, 200, -20
        renderer.update_scene(data, camera=cam)
        frames.append(_caption(Image.fromarray(renderer.render()), stage))

    trial.run(r["seed"], r["target"], np.array(g["xy"]), g["yaw"], g["top"], seen=seen)
    _row(frames).save(OUT / "trial.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    samples = json.loads((args.data / "test" / "samples.json").read_text())
    heights = np.load(args.data / "test" / "heights.npy").astype(np.float32)
    rows = json.loads((args.run / "evaluation.json").read_text())["rows"]
    dev = device()
    net = GraspNet().to(dev)
    net.load_state_dict(torch.load(args.run / "best.pt", map_location=dev))
    net.eval()
    scores = np.concatenate(
        [pick.model_scores(net, heights[i : i + 64], dev) for i in range(0, len(heights), 64)]
    )
    labels_figure(samples, heights)
    outputs_figure(samples, heights, scores, rows)
    score_map_figure(samples, heights, scores)
    if not (OUT / "trial.png").exists():
        trial_figure(rows)
    print(f"pictures written to {OUT}/")


if __name__ == "__main__":
    main()
