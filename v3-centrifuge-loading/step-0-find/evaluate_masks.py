"""How exact are SAM 3's outlines? Each against the simulator's own outline of the same tube, pixel by pixel.

The simulator draws, for every pixel, which object is seen there, so the true outline of each tube is exact:
the pixels where that tube (body or cap) is what the camera sees. Same scenes as evaluate_pose.py: the 20 test
seeds upright, and knocked over. Uses SAM 3's saved outlines (results/sam_raw/).

For each tube the camera sees (at least 30 pixels), the SAM 3 outline that overlaps it most:
- **IoU** (intersection over union): the pixels both outlines share, divided by the pixels either one covers.
  1.0 is a perfect match; 0.5 is the usual bar for "the same object".
- **Precision**: the share of SAM 3's outline that really is the tube (the rest spilled onto other things).
- **Recall**: the share of the tube that SAM 3's outline covers (the rest it missed).
- **Edge off**: wrong pixels (missed plus spilled) divided by the length of the tube's true edge, in pixels.
  Roughly how far, on average, SAM 3's edge sits from the true edge.
- **Cap covered**: for capped tubes, the share of the cap's pixels inside the tube outline.

    cd step-0-find
    PYTHONPATH=$PWD:$PWD/../simulation-environment \
      pixi run --manifest-path ../simulation-environment/pixi.toml -e sam python evaluate_masks.py
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import mujoco
import numpy as np

from bench import take_pictures
from evaluate_pose import CAMERAS, knocked_scene, true_tubes
from sam import TUBE_WORD, Sam3

OUT = Path(__file__).parent / "results"
MIN_PIXELS = 30
SAME = 0.5  # IoU at or above which a SAM 3 outline counts as that tube


def true_outlines(
    model, data, truth, views
) -> dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]]:
    """{camera: ({tube: its pixels}, {capped tube: its cap's pixels})} from the renderer."""
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
            tubes = {t: body == model.body(t).id for t in truth.tubes}
            caps = {t: ids == model.geom(f"{t}_cap_visual").id for t in truth.tubes if truth.capped[t]}
            out[name] = (tubes, caps)
    return out


def edge_length(mask: np.ndarray) -> int:
    """Pixels of ``mask`` that touch a pixel outside it."""
    inner = mask.copy()
    inner[1:, :] &= mask[:-1, :]
    inner[:-1, :] &= mask[1:, :]
    inner[:, 1:] &= mask[:, :-1]
    inner[:, :-1] &= mask[:, 1:]
    return int((mask & ~inner).sum())


def compare(
    sam: list[np.ndarray], tubes: dict[str, np.ndarray], caps: dict[str, np.ndarray], lying: set[str]
):
    """One row per tube seen, and how many SAM 3 outlines match no tube (and of those, how many cover two)."""
    rows, used = [], set()
    for name, true in tubes.items():
        n = int(true.sum())
        if n < MIN_PIXELS:
            continue
        best, best_iou = None, 0.0
        for i, m in enumerate(sam):
            iou = (m & true).sum() / (m | true).sum()
            if iou > best_iou:
                best, best_iou = i, float(iou)
        row = {
            "tube": name,
            "pixels": n,
            "lying": name in lying,
            "iou": round(best_iou, 3),
            "found": best_iou >= SAME,
        }
        if best is not None and best_iou > 0:
            m = sam[best]
            both = int((m & true).sum())
            row.update(
                precision=round(both / int(m.sum()), 3),
                recall=round(both / n, 3),
                edge_px=round(int((m ^ true).sum()) / max(edge_length(true), 1), 2),
            )
            if name in caps and caps[name].sum() >= 10:
                row["cap_covered"] = round(float((m & caps[name]).sum() / caps[name].sum()), 3)
            if best_iou >= SAME:
                used.add(best)
        rows.append(row)
    spare = [m for i, m in enumerate(sam) if i not in used]
    two = sum(
        sum((m & t).sum() >= 0.3 * t.sum() for t in tubes.values() if t.sum() >= MIN_PIXELS) >= 2
        for m in spare
    )
    return rows, len(spare), int(two)


def _mid(values, worst_low=True) -> str:
    """'median / worst 5 %' (the 5th percentile when low is bad, the 95th when high is bad)."""
    if not values:
        return "-"
    return f"{statistics.median(values):.2f} / {np.percentile(values, 5 if worst_low else 95):.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()
    sam = Sam3()
    rows, extra = [], defaultdict(lambda: [0, 0])
    for seed in range(args.seeds):
        for kind in ("upright", "knocked"):
            if kind == "upright":
                views, (model, data, truth) = take_pictures(seed, CAMERAS)
            else:
                views, (model, data, truth), _ = knocked_scene(seed)
            lying = {t.name for t in true_tubes(model, data, truth) if t.lying}
            outlines = true_outlines(model, data, truth, views)
            for cam in CAMERAS:
                tubes, caps = outlines[cam]
                r, spare, two = compare(sam.outlines(views[cam].png, TUBE_WORD)[0], tubes, caps, lying)
                rows += [{"seed": seed, "scene": kind, "camera": cam, **x} for x in r]
                extra[cam][0] += spare
                extra[cam][1] += two
        print(f"seed {seed} done", flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / "evaluation_masks_sam3.json").write_text(json.dumps(rows, indent=1))

    print(
        "\nSAM 3's outlines against the simulator's (median / worst 5 %), outlines that matched (IoU >= 0.5)"
    )
    print(
        "| Camera | Tubes | Tubes seen | Matched | IoU | Precision | Recall | Edge off (px) | Cap covered |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for cam in CAMERAS:
        for label, pick in (("standing", False), ("lying", True)):
            rs = [r for r in rows if r["camera"] == cam and r["lying"] == pick]
            f = [r for r in rs if r["found"]]
            print(
                f"| {cam} | {label} | {len(rs)} | {len(f)} | {_mid([r['iou'] for r in f])} "
                f"| {_mid([r['precision'] for r in f])} | {_mid([r['recall'] for r in f])} "
                f"| {_mid([r['edge_px'] for r in f], worst_low=False)} "
                f"| {_mid([r['cap_covered'] for r in f if 'cap_covered' in r])} |"
            )
    print("\nOutlines matching no tube (IoU < 0.5 with every tube), and of those, how many cover two tubes:")
    for cam in CAMERAS:
        print(f"  {cam}: {extra[cam][0]} spare, {extra[cam][1]} covering two tubes")
    print("\nTubes seen but missed (best IoU < 0.5), by size in pixels:")
    for cam in CAMERAS:
        missed = sorted((r["pixels"], r["iou"]) for r in rows if r["camera"] == cam and not r["found"])
        print(
            f"  {cam}: {len(missed)} missed; (pixels, best IoU): {missed[:12]}"
            + (" ..." if len(missed) > 12 else "")
        )


if __name__ == "__main__":
    main()
