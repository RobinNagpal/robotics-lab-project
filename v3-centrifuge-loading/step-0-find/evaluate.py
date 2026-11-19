"""Score Step 0 on 20 seeds (5 per job), for each set of pictures, against the simulator's answers.

    cd step-0-find
    PYTHONPATH=$PWD:$PWD/../simulation-environment \
      pixi run --manifest-path ../simulation-environment/pixi.toml python evaluate.py --model gemini

--model oracle answers from the simulator itself (every point exactly on its tube): no key, no cost. It checks
that the pictures, the geometry, the cell matching and the scoring are right, so that whatever Gemini gets
wrong is Gemini's.

The simulator's answers are read here only to grade, never by find.py.
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from centrifuge_scene.racks import floor_height
from centrifuge_scene.settings import SAMPLE_COLOURS, TUBE

import gemini
from bench import take_pictures
from find import FORMAT, find_separate, find_together  # noqa: F401  (FORMAT documents the oracle's output)
from sam import Sam3, describe

CAMERAS = ("overhead", "racks_high", "racks_low")
CONFIGS = {
    "overhead only": ("separate", ["overhead"]),
    "high only": ("separate", ["racks_high"]),
    "overhead + high": ("separate", ["overhead", "racks_high"]),
    "all three, our vote": ("separate", ["overhead", "racks_high", "racks_low"]),
    "all three, one question": ("together", ["racks_high", "racks_low", "overhead"]),
}
COLOUR_NAMES = ("red", "yellow", "green", "purple", "orange")
OUT = Path(__file__).parent / "results"


def true_table(model, data, truth) -> dict[str, dict]:
    """{cell: {kind, colour, top}} from the simulator. For grading only."""
    table = {}
    for name, (rack, idx) in truth.cells.items():
        b = model.body(name).id
        axis = data.xmat[b].reshape(3, 3)[:, 2]
        length = 0.1196 if truth.capped[name] else 0.1146
        if name in truth.samples:
            kind, colour = "sample", COLOUR_NAMES[SAMPLE_COLOURS.index(truth.samples[name].colour)]
        elif name.startswith("water"):
            kind, colour = "water", "blue"
        else:
            kind, colour = "empty", "none"
        table[f"{rack}/{idx}"] = {"kind": kind, "colour": colour, "top": data.xpos[b] + length * axis}
    return table


class Oracle:
    """Answers as a perfect model would: every tube's top, projected into the first picture."""

    def __init__(self, views, table):
        self.views, self.table = views, table

    def __call__(self, pngs, prompt):
        cam = next(v.camera for v in self.views.values() if v.png == pngs[0])
        items = []
        for t in self.table.values():
            u, v, _ = cam.project(t["top"])
            y, x = v / cam.height * 1000, u / cam.width * 1000
            items.append(
                {
                    "point": [y, x],
                    "label": f"{t['kind']} {t['colour']}",
                    "kind": t["kind"],
                    "colour": t["colour"],
                }
            )
        return gemini.Answer(json.dumps(items), 0.0, 0, 0)


class PerfectOutlines:
    """SAM 3's job done perfectly: each tube's and cap's outline from the simulator, then our rules (sam.py).

    It measures the rules alone, so that in SAM 3's results we can tell its outlining from our rules.
    """

    def __init__(self, views, model, data, truth):
        import mujoco
        from PIL import Image

        self.answers = {}
        geom = int(mujoco.mjtObj.mjOBJ_GEOM)
        with mujoco.Renderer(
            model, height=views[CAMERAS[0]].camera.height, width=views[CAMERAS[0]].camera.width
        ) as r:
            r.enable_segmentation_rendering()
            for name, view in views.items():
                r.update_scene(data, camera=name)
                seg = r.render()
                ids = np.where(seg[..., 1] == geom, seg[..., 0], -1)
                body = np.where(ids >= 0, model.geom_bodyid[np.clip(ids, 0, None)], -1)
                tubes = [body == model.body(t).id for t in truth.tubes]
                caps = [ids == model.geom(f"{t}_cap_visual").id for t in truth.tubes if truth.capped[t]]
                rgb = np.array(Image.open(io.BytesIO(view.png)).convert("RGB"))
                self.answers[view.png] = json.dumps(describe(rgb, [m for m in tubes if m.any()], caps, name))

    def __call__(self, pngs, prompt):
        return gemini.Answer(self.answers[pngs[0]], 0.0, 0, 0)


def score(found, table, per_camera, views) -> dict:
    """Right or wrong per cell, and how well each point is placed, against the simulator's answers.

    Positions, for every point a picture gave that lies on a real tube's cell:
    - px: the point in the picture against the true tube top drawn into the same picture (pixels);
    - depth: the 3D point from the depth picture against the true top (sideways and in height, m);
    - height: the camera's line through the point, met at the tube's known height (all tubes are the same
      50 ml tube), against the true top (sideways, m).
    found_within: tubes that some picture's 3D point (from depth) put within 10, 20, 30 mm of the true top
    sideways: would they be found without knowing where the cells are.
    """
    tp = [c for c in found if c in table]
    px, depth_xy, depth_z, height_xy = [], [], [], []
    nearest = {c: np.inf for c in table}
    for cam, dets in per_camera.items():
        camera = views[cam].camera
        for d in dets:
            if d.position is not None:
                for c, t in table.items():
                    nearest[c] = min(nearest[c], float(np.linalg.norm(d.position[:2] - t["top"][:2])))
            if d.cell not in table:
                continue
            top = table[d.cell]["top"]
            u, v, _ = camera.project(top)
            px.append(float(np.hypot(d.pixel[0] - u, d.pixel[1] - v)))
            if d.position is not None:
                depth_xy.append(float(np.linalg.norm(d.position[:2] - top[:2])))
                depth_z.append(float(d.position[2] - top[2]))
            origin, direction = d.ray
            z = floor_height() + (TUBE.capped_height if d.kind == "sample" else TUBE.height)
            if abs(direction[2]) > 1e-6:
                p = origin + direction * (z - origin[2]) / direction[2]
                height_xy.append(float(np.linalg.norm(p[:2] - top[:2])))
    return {
        "tubes": len(table),
        "found": len(tp),
        "missed": sorted(set(table) - set(found)),
        "made_up": sorted(set(found) - set(table)),
        "kind_right": sum(found[c].kind == table[c]["kind"] for c in tp),
        "colour_right": sum(found[c].colour == table[c]["colour"] for c in tp),
        "wrong": [
            f"{c}: said {found[c].kind}/{found[c].colour}, is {table[c]['kind']}/{table[c]['colour']}"
            for c in tp
            if (found[c].kind, found[c].colour) != (table[c]["kind"], table[c]["colour"])
        ],
        "px": [round(e, 1) for e in px],
        "depth_xy_mm": [round(e * 1000, 1) for e in depth_xy],
        "depth_z_mm": [round(e * 1000, 1) for e in depth_z],
        "height_xy_mm": [round(e * 1000, 1) for e in height_xy],
        "found_within": {str(r): sum(n <= r / 1000 for n in nearest.values()) for r in (10, 20, 30)},
    }


def _mid(values) -> str:
    """'median / 95 % under' of a list, or '-'."""
    if not values:
        return "-"
    return f"{statistics.median(values):.0f} / {np.percentile(values, 95):.0f}"


def print_tables(rows: list[dict], title: str) -> None:
    by = defaultdict(list)
    for r in rows:
        by[r["config"]].append(r)
    print(f"\n{title}: what each tube is")
    print(
        "| Pictures | Tubes found | Missed | Made up | Kind right | Colour right "
        "| Seconds per seed | Cost per seed |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for config, rs in by.items():
        tubes, found = sum(r["tubes"] for r in rs), sum(r["found"] for r in rs)
        print(
            f"| {config} | {found} of {tubes} | {sum(len(r['missed']) for r in rs)} | "
            f"{sum(len(r['made_up']) for r in rs)} | {sum(r['kind_right'] for r in rs)} of {found} | "
            f"{sum(r['colour_right'] for r in rs)} of {found} | "
            f"{statistics.mean(r['seconds'] for r in rs):.1f} | "
            f"${statistics.mean(r['dollars'] for r in rs):.4f} |"
        )
    skipped = sum(r.get("skipped", 0) for r in rows if r["config"] == "all three, our vote")
    skipped += sum(r.get("skipped", 0) for r in rows if r["config"] == "all three, one question")
    if skipped:
        print(
            f"({skipped} questions got no answer in time and were left out; "
            "their pictures count as seeing nothing)"
        )
    print(f"\n{title}: where each tube is (median / 95 % under)")
    print(
        "| Pictures | Point off in the picture (px) | 3D from depth: sideways (mm) "
        "| 3D from depth: height (mm) "
        "| At the tube's known height: sideways (mm) | Found within 10 / 20 / 30 mm, no layout |"
    )
    print("|---|---|---|---|---|---|")
    for config, rs in by.items():
        tubes = sum(r["tubes"] for r in rs)
        within = " / ".join(str(sum(r["found_within"][k] for r in rs)) for k in ("10", "20", "30"))
        print(
            f"| {config} | {_mid([e for r in rs for e in r['px']])} "
            f"| {_mid([e for r in rs for e in r['depth_xy_mm']])} "
            f"| {_mid([e for r in rs for e in r['depth_z_mm']])} "
            f"| {_mid([e for r in rs for e in r['height_xy_mm']])} | "
            f"{within} of {tubes} |"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("gemini", "sam3", "outlines", "oracle"), default="oracle")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--thinking", default=gemini.THINKING)
    args = ap.parse_args()

    sam_model = Sam3() if args.model == "sam3" else None
    rows = []
    for seed in range(args.seeds):
        views, (model, data, truth) = take_pictures(seed, CAMERAS)
        table = true_table(model, data, truth)
        if args.model == "oracle":
            ask = Oracle(views, table)
        elif args.model == "outlines":
            ask = PerfectOutlines(views, model, data, truth)
        elif args.model == "sam3":
            ask = sam_model
            sam_model.camera_of = {v.png: n for n, v in views.items()}
        else:

            def ask(pngs, prompt):
                return gemini.ask(pngs, prompt, thinking=args.thinking)

        for config, (mode, cams) in CONFIGS.items():
            if mode == "together" and args.model in ("sam3", "outlines"):
                continue  # SAM 3 looks at one picture at a time
            subset = {c: views[c] for c in cams}
            found, answers, per_camera = (find_separate if mode == "separate" else find_together)(subset, ask)
            s = score(found, table, per_camera, views)
            rows.append(
                {
                    "seed": seed,
                    "job": truth.scenario,
                    "config": config,
                    **s,
                    "seconds": round(sum(a.seconds for a in answers), 1),
                    "dollars": round(sum(a.dollars for a in answers), 5),
                    "tokens_in": sum(a.tokens_in for a in answers),
                    "tokens_out": sum(a.tokens_out for a in answers),
                    "skipped": sum(a.skipped for a in answers),
                }
            )
        print(f"seed {seed} ({truth.scenario}) done", flush=True)

    OUT.mkdir(exist_ok=True)
    (OUT / f"evaluation_{args.model}.json").write_text(json.dumps(rows, indent=1))
    by = defaultdict(list)
    for r in rows:
        by[r["config"]].append(r)
    print_tables(rows, f"{args.model}, {args.seeds} seeds")
    for r in rows:
        if r["missed"] or r["made_up"] or r["wrong"]:
            print(
                f"  seed {r['seed']} {r['config']}: missed {r['missed']} made up {r['made_up']} "
                f"wrong {r['wrong']}"
            )


if __name__ == "__main__":
    main()
