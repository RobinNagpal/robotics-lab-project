"""Draw the pictures used in the side-grasp docs, into docs/side/images/.

    python -m grasp.figures_side --run runs/side_grconv
    python -m grasp.figures_side --version 2 --run runs/side_v2 --data data_side2   # into images/v2/

Needs the side test data, the trained weights and the evaluation (runs/<name>/evaluation.json).
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
from PIL import Image, ImageDraw

from . import pick, scenes, side
from .figures import GREEN, RED, YELLOW, _caption, _grey, _row, _target, _xy
from .net import GraspNet
from .train import device

OUT = Path("docs/side/images")
# How far closed the fingers are in the "nearly closed" picture of why_blocked.png.
NEARLY_CLOSED = 0.9


def _arrow(
    draw: ImageDraw.ImageDraw, u: float, v: float, a: float, length: float, colour, width: int
) -> None:
    """A line from the tube out towards where the gripper comes from; the picture's up is the table's +y."""
    cx, cy = _xy(u, v)
    draw.line([(cx, cy), (cx + math.cos(a) * length, cy - math.sin(a) * length)], fill=colour, width=width)


def _directions(draw: ImageDraw.ImageDraw, t: dict) -> None:
    for k, ok in enumerate(t["labels"]):
        _arrow(draw, t["u"], t["v"], float(side.direction(k)), 55 if ok else 22, GREEN if ok else RED, 3)


def labels_figure(samples: list[dict], heights: np.ndarray) -> None:
    """How a picture is labelled: all 24 directions, for a few tubes."""
    some = next(i for i, s in enumerate(samples) if 3 <= sum(_target(s)["labels"]) <= 8)
    boxed = next(i for i, s in enumerate(samples) if not any(_target(s)["labels"]))
    many = max(range(len(samples)), key=lambda i: sum(_target(samples[i])["labels"]))
    panels = []
    for i, text in (
        (some, "a few clear directions"),
        (boxed, "boxed in: none clear"),
        (many, "the most open tube"),
    ):
        img = _grey(heights[i])
        _directions(ImageDraw.Draw(img), _target(samples[i]))
        panels.append(_caption(img, text))
    _row(panels).save(OUT / "labels.png")


def outputs_figure(samples, heights, scores, rows) -> None:
    """What the model says for a few test tubes, next to the truth."""
    picks = []
    for want in ("good", "boxed"):
        for r in rows:
            if want == "good" and 2 <= r["clear_directions"] <= 8 and r["model"] and r["model"].get("good"):
                picks.append(r["i"])
                break
            if want == "boxed" and r["clear_directions"] == 0 and r["model"] is None:
                picks.append(r["i"])
                break
    top, bottom = [], []
    for i in picks:
        s, t = samples[i], _target(samples[i])
        truth = _grey(heights[i])
        _directions(ImageDraw.Draw(truth), t)
        top.append(_caption(truth, "truth: clear (green), blocked (red)"))
        calib = Calibration.from_dict(s["camera"])
        _, (u, v, k) = pick.learned(scores[i], heights[i], calib, threshold=0.0)
        out = _grey(heights[i])
        d = ImageDraw.Draw(out)
        for b in range(side.BINS):
            sc = float(scores[i][b, v, u])
            _arrow(
                d, u, v, float(side.direction(b)), 12 + 45 * sc, (int(255 * (1 - sc)), int(255 * sc), 80), 3
            )
        if rows[i]["model"] is not None:
            # In the picture's own directions (version 2 pictures are turned to the rack).
            _arrow(d, u, v, float(side.direction(k)), 70, YELLOW, 5)
        bottom.append(_caption(out, "model: score per direction"))
    grid, under = _row(top), _row(bottom)
    out = Image.new("RGB", (grid.width, grid.height + under.height + 8), (255, 255, 255))
    out.paste(grid, (0, 0))
    out.paste(under, (0, grid.height + 8))
    out.save(OUT / "outputs.png")


def score_map_figure(samples, heights, scores) -> None:
    """The best score at each pixel, over all directions."""
    i = next(
        i for i, s in enumerate(samples) if len(s["tubes"]) >= 5 and any(any(t["labels"]) for t in s["tubes"])
    )
    best = scores[i].max(axis=0)
    pic = _grey(heights[i])
    base = np.asarray(pic, np.float32) * 0.35
    glow = np.asarray(
        Image.fromarray((255 * best).astype(np.uint8)).resize(pic.size, Image.BILINEAR), np.float32
    )
    glow = (glow / 255)[..., None]
    over = Image.fromarray(np.clip(base * (1 - glow) + glow * np.array(YELLOW), 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(over)
    for t in samples[i]["tubes"]:
        cx, cy = _xy(t["u"], t["v"])
        d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], outline=GREEN if any(t["labels"]) else RED, width=2)
    _row(
        [_caption(pic, "input: height above the table"), _caption(over, "output: best score at each pixel")]
    ).save(OUT / "score_map.png")


def trial_figure(rows, preshape: bool = False, row: int | None = None) -> None:
    """Moments of one side grasp with full physics: test row ``row``, or the first success."""
    r = (
        rows[row]
        if row is not None
        else next(r for r in rows if r["model"] and r["model"].get("physics", {}).get("success"))
    )
    g = r["model"]
    frames = []
    renderer = None
    cam = mujoco.MjvCamera()
    # Look from the side the gripper comes from, a little round, so its body does not hide the tube.
    azimuth = math.degrees(g["direction"]) + 180 + 50

    def seen(stage, model, data):
        nonlocal renderer
        if renderer is None:
            renderer = mujoco.Renderer(model, 360, 480)
        cam.lookat[:] = [g["xy"][0], g["xy"][1], 0.12]
        cam.distance, cam.azimuth, cam.elevation = 0.5, azimuth, -20
        renderer.update_scene(data, camera=cam)
        frames.append(_caption(Image.fromarray(renderer.render()), stage))

    side.run(
        r["seed"], r["target"], np.array(g["xy"]), g["direction"], g["top"], seen=seen, preshape=preshape
    )
    _row(frames).save(OUT / "trial.png")


def _outward(data, scene, xy) -> float:
    """The direction straight out of the rack the tube stands in, across its two rows."""
    racks = [i.name for i in scene.instances if "rack" in i.name]
    rack = min(racks, key=lambda n: np.linalg.norm(data.body(n).xpos[:2] - xy))
    across = data.body(rack).xmat.reshape(3, 3)[:2, 1]
    if np.dot(xy - data.body(rack).xpos[:2], across) < 0:
        across = -across
    return float(math.atan2(across[1], across[0]))


def why_blocked_figure(seeds=range(300000, 300040), preshape: bool = False) -> None:
    """From above: a tube on the outside of its rack, the gripper coming straight in from outside.

    Fully open, the pads are 93 mm apart and land on the tubes beside the target in its row (36 mm
    apart); nearly closed, they fit in the 11 mm gaps. With ``preshape`` (version 2), "nearly
    closed" is the version 2 opening (side.PRESHAPE_GAP), and the example is one the full check
    calls clear.
    """
    for seed in seeds:
        model, data, scene = scenes.build(seed)
        checker = side.SideChecker(model, data, preshape=preshape)
        closing = checker.way_in_closing if preshape else NEARLY_CLOSED
        for t in scenes.tubes(scene):
            xy, top = scenes.rim(data, t)
            a = _outward(data, scene, xy)
            pinch = side.pinch_at(xy, a, top)
            open_hit = set(checker.touches_side(pinch, a, 0.0)) - {t.name}
            if not open_hit or not all(
                n.split("_")[0] in ("source", "receiver", "empty", "filled") for n in open_hit
            ):
                continue
            if preshape and not checker.clear(xy, a, top, t.name):
                continue
            if any(set(checker.touches_side(p, a, closing)) for p in side.way_in(xy, a, top)):
                continue
            renderer = mujoco.Renderer(model, 420, 420)
            cam = mujoco.MjvCamera()
            cam.lookat[:] = [*(xy + 0.06 * side.along(a)[:2]), top]
            cam.distance, cam.azimuth, cam.elevation = 0.42, math.degrees(a) + 180, -89
            panels = []
            shut = (
                f"opened {side.PRESHAPE_GAP * 1000:.0f} mm: pads fit"
                if preshape
                else "nearly closed: pads fit"
            )
            for f, text in ((0.0, "fully open: pads hit the row neighbours"), (closing, shut)):
                checker.touches_side(pinch, a, f)
                renderer.update_scene(data, camera=cam)
                panels.append(_caption(Image.fromarray(renderer.render()), text))
            _row(panels).save(OUT / ("fingers.png" if preshape else "why_blocked.png"))
            print(f"why_blocked: seed {seed}, {t.name}")
            return
    print("why_blocked: no example found")


def rack_frame_figure(samples: list[dict]) -> None:
    """Version 2: the close-up as before (unturned) and turned to the rack, with the 24 directions.

    Unturned, the straight-out directions of the rack (yellow) fall between two of the 24; turned,
    four of the 24 are exactly them.
    """
    i = next(i for i, s in enumerate(samples) if 20 < abs(math.degrees(s["frame_yaw"])) < 40)
    s = samples[i]
    model, data, _ = scenes.build(s["seed"])
    camera = scenes.LookCamera(model, data)
    xy, top = np.array(s["step0"]["xy"]), s["step0"]["top"]
    panels = []
    for yaw, text in (
        (0.0, "before: picture and directions follow the table"),
        (s["frame_yaw"], "now: turned to the rack"),
    ):
        heights, calib = camera.look(xy, top, side.LOOK_HEIGHT, yaw=yaw)
        u, v = calib.project(np.array([[*xy, top]]))[0]
        img = _grey(heights)
        d = ImageDraw.Draw(img)
        for k in range(side.BINS):
            _arrow(d, u, v, float(side.direction(k)), 60, (150, 150, 150), 1)
        for q in range(4):
            # The rack's sides, in this picture's directions.
            _arrow(d, u, v, s["frame_yaw"] - yaw + q * math.pi / 2, 70, YELLOW, 3)
        panels.append(_caption(img, text))
    camera.close()
    _row(panels).save(OUT / "rack_frame.png")


def smoothing_figure(samples, heights, scores, raw_rows, smooth_rows) -> None:
    """Version 2: one tube where the raw best direction is at the edge of the clear run, and blocked;
    smoothed over neighbours, the choice moves into the middle (version 1 network, version 1 data)."""
    i = next(
        r["i"]
        for r, q in zip(raw_rows, smooth_rows, strict=True)
        if r["model"]
        and q["model"]
        and not r["model"]["clear"]
        and q["model"]["good"]
        and r["clear_directions"] >= 3
    )
    s, t = samples[i], _target(samples[i])
    calib = Calibration.from_dict(s["camera"])
    panels = []
    truth = _grey(heights[i])
    _directions(ImageDraw.Draw(truth), t)
    panels.append(_caption(truth, "truth: clear (green), blocked (red)"))
    for sc, text in (
        (scores[i], "before: best single score"),
        (side.smoothed(scores[i]), "now: mixed with neighbours"),
    ):
        _, (u, v, k) = pick.learned(sc, heights[i], calib, threshold=0.0)
        img = _grey(heights[i])
        d = ImageDraw.Draw(img)
        for b in range(side.BINS):
            x = float(sc[b, v, u])
            _arrow(d, u, v, float(side.direction(b)), 12 + 45 * x, (int(255 * (1 - x)), int(255 * x), 80), 3)
        _arrow(d, u, v, float(side.direction(k)), 70, YELLOW, 5)
        panels.append(_caption(img, text))
    _row(panels).save(OUT / "smoothing.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=Path("data_side"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--version", type=int, choices=[1, 2], default=1)
    # Version 2: test row 24, a lift after which no neighbour has settled in its cell (see improved.md).
    parser.add_argument("--trial-row", type=int, default=None)
    args = parser.parse_args()
    global OUT
    if args.version == 2:
        OUT = OUT / "v2"
    OUT.mkdir(parents=True, exist_ok=True)
    samples = json.loads((args.data / "test" / "samples.json").read_text())
    heights = np.load(args.data / "test" / "heights.npy").astype(np.float32)
    rows = json.loads((args.run / "evaluation.json").read_text())["rows"]
    dev = device()
    net = GraspNet(side.BINS).to(dev)
    net.load_state_dict(torch.load(args.run / "best.pt", map_location=dev))
    net.eval()
    scores = np.concatenate(
        [pick.model_scores(net, heights[i : i + 64], dev) for i in range(0, len(heights), 64)]
    )
    if args.version == 2:
        scores = side.smoothed(scores)
    labels_figure(samples, heights)
    outputs_figure(samples, heights, scores, rows)
    score_map_figure(samples, heights, scores)
    why_blocked_figure(preshape=args.version == 2)
    if args.version == 2:
        rack_frame_figure(samples)
        v1 = Path("data_side") / "test"
        v1_heights = np.load(v1 / "heights.npy").astype(np.float32)
        v1_net = GraspNet(side.BINS).to(dev)
        v1_net.load_state_dict(torch.load("runs/side_grconv/best.pt", map_location=dev))
        v1_net.eval()
        v1_scores = np.concatenate(
            [pick.model_scores(v1_net, v1_heights[i : i + 64], dev) for i in range(0, len(v1_heights), 64)]
        )
        smoothing_figure(
            json.loads((v1 / "samples.json").read_text()),
            v1_heights,
            v1_scores,
            json.loads(Path("runs/side_grconv/evaluation.json").read_text())["rows"],
            json.loads(Path("runs/side_grconv_smoothed/evaluation.json").read_text())["rows"],
        )
    if not (OUT / "trial.png").exists():
        trial_figure(rows, preshape=args.version == 2, row=args.trial_row)
    print(f"pictures written to {OUT}/")


if __name__ == "__main__":
    main()
