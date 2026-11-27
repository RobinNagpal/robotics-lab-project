"""Pictures of the person reaching in, with the truth for each: what the detector will be trained on.

``python render.py --sheet out.jpg`` makes the go / no-go contact sheet: random frames from the safety camera,
each labelled with its zone and conditions, with the zones' edges drawn on the table. ``--cameras safety
overhead`` puts v3's overhead camera beside it.

The arm stands in a random working pose for each frame (the pinch point above a rack, the balance or the
centrifuge, solved with v3's own inverse kinematics), so the person is seen next to the arm as it would be
during a job. Phase 2 renders from real v3 jobs instead.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "v3-centrifuge-loading" / "step-1-weigh"))

from centrifuge_scene import scene  # noqa: E402
from centrifuge_scene.settings import LAYOUT  # noqa: E402

import cameras  # noqa: E402
import hand  # noqa: E402
import scene_plus  # noqa: E402
import zones  # noqa: E402

ZONE_COLOURS = {
    zones.Zone.CLEAR: (60, 200, 90),
    zones.Zone.WARNING: (250, 190, 40),
    zones.Zone.STOP: (235, 60, 50),
}
# Where the arm works: above the racks, the balance and the centrifuge.
WORK_SPOTS = (LAYOUT.sample_rack, LAYOUT.supply_rack, LAYOUT.scale, LAYOUT.centrifuge)


@dataclass(frozen=True)
class Frame:
    seed: int
    look: hand.Look
    lighting: str
    light_scale: float
    goal: str
    phase: str  # "in", "pause", "out"
    truth: zones.Truth
    gap: float  # m, the smallest gap between the person and the robot


def random_arm_pose(arm, rng: np.random.Generator) -> np.ndarray | None:
    """Joint angles with the pinch point 8 to 30 cm above a random work spot, fingers down; None if
    unreachable."""
    from motion import Blocked, fingers_down

    x, y = WORK_SPOTS[rng.integers(len(WORK_SPOTS))]
    target = np.array([x + rng.uniform(-0.06, 0.06), y + rng.uniform(-0.06, 0.06), rng.uniform(0.20, 0.42)])
    try:
        return arm.solve_any(target, fingers_down(rng.uniform(-math.pi / 2, math.pi / 2)))
    except Blocked:
        return None


def pose_frame(model, data, arm, rng: np.random.Generator, seed: int) -> Frame:
    """Set up one random frame: the person's look and place, the lighting, the arm's pose."""
    scene.reset(model, data)
    q = random_arm_pose(arm, rng)
    if q is not None:
        data.qpos[arm.joints] = q
    look = hand.Look.draw(rng)
    hand.set_look(model, look)
    lighting = str(rng.choice(list(scene_plus.LIGHTING), p=[0.25, 0.5, 0.25]))
    scale = float(rng.uniform(*scene_plus.LIGHTING[lighting]))
    scene_plus.set_lighting(model, scale)
    mujoco.mj_forward(model, data)
    goal, reach = hand.draw_reach(rng, model, data, 0.0, 0.0)
    # A moment in the reach: mostly while the hand is in (that is where the decisions are), some on the way.
    phase = str(rng.choice(["in", "pause", "out"], p=[0.4, 0.4, 0.2]))
    t = {
        "in": rng.uniform(reach.t0, reach.t_arrive),
        "pause": rng.uniform(reach.t_arrive, reach.t_leave),
        "out": rng.uniform(reach.t_leave, reach.t_end),
    }[phase]
    hand.place(model, data, reach.wrist(t), reach.quat())
    mujoco.mj_forward(model, data)
    truth = zones.truth(model, data)
    gap = zones.distance_to_robot(model, data)
    return Frame(seed, look, lighting, scale, goal, phase, truth, gap)


def draw_zones(img: Image.Image, model, data, camera: str) -> None:
    """The two zone edges, drawn on the table."""
    d = ImageDraw.Draw(img)
    for radius, colour in (
        (zones.STOP_RADIUS, ZONE_COLOURS[zones.Zone.STOP]),
        (zones.WARNING_RADIUS, ZONE_COLOURS[zones.Zone.WARNING]),
    ):
        a = np.linspace(-math.pi * 0.62, math.pi * 0.62, 120)
        pts = np.stack([radius * np.cos(a), radius * np.sin(a), np.zeros_like(a)], axis=1)
        px, ok = cameras.to_pixels(model, data, camera, pts, img.width, img.height)
        d.line([tuple(p) for p, k in zip(px, ok, strict=True) if k], fill=colour, width=2)


def label(img: Image.Image, f: Frame) -> None:
    d = ImageDraw.Draw(img)
    t = f.truth
    reach = "out of view" if not t.visible else f"{t.reach:.2f} m"
    lines = [
        f"{t.zone.name}  reach {reach}  gap {f.gap:.2f} m",
        f"{f.look.hand}, {f.look.sleeve}, size {f.look.size:.2f}",
        f"light {f.lighting} x{f.light_scale:.2f}  goal {f.goal}  {f.phase}",
    ]
    d.rectangle([0, 0, img.width, 14 * len(lines) + 4], fill=(0, 0, 0))
    for i, line in enumerate(lines):
        d.text((4, 2 + 14 * i), line, fill=ZONE_COLOURS[t.zone] if i == 0 else (230, 230, 230))


def contact_sheet(out: Path, n: int, seeds: list[int], rng_seed: int, camera_names: list[str]) -> list[Frame]:
    from motion import Arm

    rng = np.random.default_rng(rng_seed)
    tiles: list[list[Image.Image]] = [[] for _ in camera_names]
    frames = []
    per_seed = math.ceil(n / len(seeds))
    for seed in seeds:
        model, _ = scene_plus.build(seed)
        data = mujoco.MjData(model)
        scene.reset(model, data)
        arm = Arm(model, data)
        renderers = {
            c: mujoco.Renderer(model, height=cameras.SAFETY.height, width=cameras.SAFETY.width)
            for c in camera_names
        }
        for _ in range(per_seed):
            if len(frames) == n:
                break
            f = pose_frame(model, data, arm, rng, seed)
            frames.append(f)
            for i, c in enumerate(camera_names):
                r = renderers[c]
                r.update_scene(data, camera=c)
                img = Image.fromarray(r.render())
                draw_zones(img, model, data, c)
                label(img, f)
                tiles[i].append(img)
        for r in renderers.values():
            r.close()
    cols = 4
    w, h = tiles[0][0].size
    rows = math.ceil(n / cols)
    sheet = Image.new("RGB", (cols * w * len(camera_names), rows * h), (20, 20, 20))
    for k in range(n):
        for i in range(len(camera_names)):
            x = (k % cols) * len(camera_names) * w + i * w
            sheet.paste(tiles[i][k], (x, (k // cols) * h))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88) if out.suffix.lower() in (".jpg", ".jpeg") else sheet.save(out)
    return frames


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sheet", type=Path, required=True)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--rng", type=int, default=0)
    p.add_argument("--cameras", nargs="+", default=["safety"])
    args = p.parse_args()
    frames = contact_sheet(args.sheet, args.n, args.seeds, args.rng, args.cameras)
    counts = {z.name: sum(f.truth.zone == z for f in frames) for z in zones.Zone}
    print(f"{len(frames)} frames: {counts}; saved {args.sheet}")


if __name__ == "__main__":
    main()
