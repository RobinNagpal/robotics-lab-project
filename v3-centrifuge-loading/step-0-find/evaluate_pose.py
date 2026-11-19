"""Measure pose.py: each tube's place, direction and size from outlines and depth, against the simulator.

Two kinds of scene, on the 20 test seeds:
- **upright**: the scenes as built, every tube standing in its rack cell (the pictures Step 0 uses);
- **knocked over**: the same seeds with one sample and one empty tube taken out and laid on the bench at a
  random place and turn (seeded), then left to roll to rest. Only placements that both the overhead and the
  high camera see whole are used.

Two sources of outlines:
- ``perfect``: the simulator's own outline of each tube. It measures pose.py alone.
- ``sam3``: SAM 3's outlines for "test tube" (sam.py), with nothing else from the simulator.

The simulator's answers are read here only to grade, never by pose.py.

    cd step-0-find
    P="pixi run --manifest-path ../simulation-environment/pixi.toml"
    PYTHONPATH=$PWD:$PWD/../simulation-environment $P python evaluate_pose.py --outlines perfect
    PYTHONPATH=$PWD:$PWD/../simulation-environment $P -e sam python evaluate_pose.py --outlines sam3
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from centrifuge_scene import scene
from centrifuge_scene.settings import LAYOUT, RACK, SCALE, TUBE

import pose
from bench import take_pictures

CAMERAS = ("overhead", "racks_high", "racks_low")
CONFIGS = {
    "overhead only": ["overhead"],
    "high only": ["racks_high"],
    "low only": ["racks_low"],
    "high + low, joined": ["racks_high", "racks_low"],
    "all three, joined": ["overhead", "racks_high", "racks_low"],
}
OUT = Path(__file__).parent / "results"
# A pose counts for a tube if its centre is this close to the tube's axis (and between its ends, roughly).
MATCH = 0.025
# Knocked-over tubes: where their middles may land, and how far from the racks and the balance they must be.
LAY_X, LAY_Y = (0.28, 0.54), (0.15, 0.62)
CLEAR = 0.03
TRIES = 30


@dataclass
class TrueTube:
    name: str
    tip: np.ndarray
    axis: np.ndarray
    length: float
    lying: bool

    @property
    def top(self) -> np.ndarray:
        return self.tip + self.length * self.axis

    @property
    def middle(self) -> np.ndarray:
        return self.tip + self.length / 2 * self.axis


def true_tubes(model, data, truth) -> list[TrueTube]:
    out = []
    for name in truth.tubes:
        b = model.body(name).id
        axis = data.xmat[b].reshape(3, 3)[:, 2].copy()
        length = TUBE.capped_height if truth.capped[name] else TUBE.height
        out.append(
            TrueTube(name, data.xpos[b].copy(), axis, length, bool(abs(axis[2]) < math.cos(math.radians(45))))
        )
    return out


def tube_masks(model, data, truth, views) -> dict[str, dict[str, np.ndarray]]:
    """{camera: {tube: the simulator's outline of it}}, from the renderer's picture of what is where."""
    geom = int(mujoco.mjtObj.mjOBJ_GEOM)
    cam0 = next(iter(views.values())).camera
    out = {}
    with mujoco.Renderer(model, height=cam0.height, width=cam0.width) as r:
        r.enable_segmentation_rendering()
        for name in views:
            r.update_scene(data, camera=name)
            seg = r.render()
            ids = np.where(seg[..., 1] == geom, seg[..., 0], -1)
            body = np.where(ids >= 0, model.geom_bodyid[np.clip(ids, 0, None)], -1)
            out[name] = {t: body == model.body(t).id for t in truth.tubes}
    return out


def _boxes() -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """(centre, half size) of the racks' and the balance's footprints on the bench."""
    half = (RACK.rows * RACK.pitch / 2, RACK.columns * RACK.pitch / 2)
    return [(LAYOUT.sample_rack, half), (LAYOUT.supply_rack, half), (LAYOUT.scale, SCALE.base[:2])]


def _clear(points: np.ndarray, others: list[np.ndarray]) -> bool:
    for (cx, cy), (hx, hy) in _boxes():
        if np.any((np.abs(points[:, 0] - cx) < hx + CLEAR) & (np.abs(points[:, 1] - cy) < hy + CLEAR)):
            return False
    return all(np.min(np.linalg.norm(points[:, None, :2] - o[None, :, :2], axis=2)) >= 0.05 for o in others)


def knock_over(seed: int, attempt: int):
    """An ``arrange`` for take_pictures: lays one sample and one empty tube on the bench (seeded)."""
    rng = np.random.default_rng([seed, attempt, 7])

    def arrange(model, data, truth):
        sample = str(rng.choice([t for t in truth.tubes if truth.capped[t]]))
        empty = str(rng.choice([t for t in truth.tubes if t.startswith("blank")]))
        laid: list[np.ndarray] = []
        for name in (sample, empty):
            for _ in range(200):
                mid = np.array([rng.uniform(*LAY_X), rng.uniform(*LAY_Y), TUBE.cap_radius])
                yaw = rng.uniform(0, 2 * math.pi)
                axis = np.array([math.cos(yaw), math.sin(yaw), 0.0])
                along = mid + np.outer(np.linspace(-0.065, 0.065, 9), axis)
                if _clear(along, laid):
                    break
            laid.append(along)
            side = np.array([-axis[1], axis[0], 0.0])
            rot = np.column_stack([side, np.cross(axis, side), axis])
            scene.move_tube(model, data, name, mid - axis * TUBE.height / 2, rot)
        truth.knocked = (sample, empty)

    return arrange


def _whole_in(view, tube: TrueTube, margin: float = 8) -> bool:
    cam = view.camera
    for p in (tube.tip, tube.top):
        u, v, d = cam.project(p)
        if d <= 0 or not (margin < u < cam.width - margin and margin < v < cam.height - margin):
            return False
    return True


def knocked_scene(seed: int):
    """Pictures of ``seed`` with two tubes knocked over, both lying flat and seen whole by two cameras."""
    for attempt in range(TRIES):
        views, (model, data, truth) = take_pictures(
            seed, CAMERAS, settle=1.0, arrange=knock_over(seed, attempt)
        )
        tubes = {t.name: t for t in true_tubes(model, data, truth)}
        laid = [tubes[n] for n in truth.knocked]
        if all(t.lying and t.middle[2] < 0.03 for t in laid) and all(
            _whole_in(views[c], t) for t in laid for c in ("overhead", "racks_high")
        ):
            return views, (model, data, truth), attempt
    raise RuntimeError(f"seed {seed}: no good placement in {TRIES} tries")


def score(found: list[pose.Pose], tubes: list[TrueTube]) -> dict:
    """Each pose matched to the tube whose axis it lies on; then how far off each matched pose is."""
    pairs = []
    for i, p in enumerate(found):
        for j, t in enumerate(tubes):
            s = float((p.centre - t.tip) @ t.axis)
            if -0.03 <= s <= t.length + 0.03:
                d = pose.distance_to_line(p.centre, t.tip, t.axis)
                if d <= MATCH:
                    pairs.append((d, i, j))
    used_p, used_t, match = set(), set(), {}
    for _, i, j in sorted(pairs):
        if i not in used_p and j not in used_t:
            used_p.add(i)
            used_t.add(j)
            match[j] = found[i]
    rows = []
    for j, t in enumerate(tubes):
        p = match.get(j)
        row = {"tube": t.name, "lying": t.lying, "found": p is not None}
        if p is not None:
            row.update(
                stand_or_lie_right=bool((p.tilt_deg > 45) == t.lying),
                axis_deg=round(math.degrees(math.acos(min(1.0, abs(float(p.axis @ t.axis))))), 2),
                place_mm=round(pose.distance_to_line(p.centre, t.tip, t.axis) * 1000, 2),
                diameter_mm=round((2 * p.radius - 2 * TUBE.radius) * 1000, 2),
                cameras=list(p.cameras),
            )
            if t.lying:
                a, b = p.ends
                ends = min(
                    max(np.linalg.norm(a - t.tip), np.linalg.norm(b - t.top)),
                    max(np.linalg.norm(a - t.top), np.linalg.norm(b - t.tip)),
                )
                row.update(
                    ends_mm=round(float(ends) * 1000, 1), length_mm=round((p.length - t.length) * 1000, 1)
                )
            else:
                row.update(top_mm=round(float(np.linalg.norm(p.top - t.top)) * 1000, 1))
        rows.append(row)
    return {"tubes": rows, "made_up": len(found) - len(match)}


def _mid(values) -> str:
    if not values:
        return "-"
    return f"{statistics.median(values):.1f} / {np.percentile(values, 95):.1f}"


def print_tables(results: list[dict], title: str) -> None:
    groups = (
        ("upright scenes: tubes standing in the racks", "upright", False),
        ("knocked-over scenes: the two tubes lying on the bench", "knocked", True),
        ("knocked-over scenes: the tubes still standing", "knocked", False),
    )
    for heading, kind, lying in groups:
        by = defaultdict(list)
        made_up = defaultdict(int)
        for r in results:
            if r["scene"] != kind:
                continue
            by[r["config"]] += [t for t in r["tubes"] if t["lying"] == lying]
            made_up[r["config"]] += r["made_up"]
        end = "both ends off (mm)" if lying else "top off (mm)"
        print(f"\n{title}, {heading} (median / 95 % under)")
        print(
            "| Pictures | Found | Made up (whole scene) | Standing or lying right | Axis direction off (deg) "
            "| Axis over 10° off "
            f"| Axis place off (mm) | Diameter off (mm) | {end} |" + (" Length off (mm) |" if lying else "")
        )
        print("|---|---|---|---|---|---|---|---|---|" + ("---|" if lying else ""))
        for config, ts in by.items():
            f = [t for t in ts if t["found"]]
            line = (
                f"| {config} | {len(f)} of {len(ts)} | {made_up[config]} | "
                f"{sum(t['stand_or_lie_right'] for t in f)} of {len(f)} | "
                f"{_mid([t['axis_deg'] for t in f])} | "
                f"{sum(t['axis_deg'] > 10 for t in f)} | "
                f"{_mid([t['place_mm'] for t in f])} | {_mid([abs(t['diameter_mm']) for t in f])} | "
            )
            if lying:
                line += f"{_mid([t['ends_mm'] for t in f])} | {_mid([abs(t['length_mm']) for t in f])} |"
            else:
                line += f"{_mid([t['top_mm'] for t in f])} |"
            print(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outlines", choices=("perfect", "sam3"), default="perfect")
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()

    if args.outlines == "sam3":
        from sam import TUBE_WORD, Sam3

        sam_model = Sam3()
    results = []
    for seed in range(args.seeds):
        for kind in ("upright", "knocked"):
            if kind == "upright":
                views, (model, data, truth) = take_pictures(seed, CAMERAS)
                attempt = None
            else:
                views, (model, data, truth), attempt = knocked_scene(seed)
            tubes = true_tubes(model, data, truth)
            if args.outlines == "perfect":
                masks = {
                    c: [m for m in ms.values() if m.sum() >= 30]
                    for c, ms in tube_masks(model, data, truth, views).items()
                }
            else:
                masks = {c: sam_model.outlines(views[c].png, TUBE_WORD)[0] for c in CAMERAS}
            per_camera = {c: pose.poses(views[c], masks[c], c) for c in CAMERAS}
            for config, cams in CONFIGS.items():
                found = pose.merge({c: per_camera[c] for c in cams})
                s = score(found, tubes)
                results.append({"seed": seed, "scene": kind, "attempt": attempt, "config": config, **s})
        print(f"seed {seed} done", flush=True)

    OUT.mkdir(exist_ok=True)
    (OUT / f"evaluation_pose_{args.outlines}.json").write_text(json.dumps(results, indent=1))
    print_tables(results, f"{args.outlines} outlines, {args.seeds} seeds")


if __name__ == "__main__":
    main()
