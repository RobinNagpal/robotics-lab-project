"""How well the hand detector and the box rule pick the right zone, on pictures it never trained on.

For every picture: YOLO's hand boxes, then ``box_zone`` turns them into a zone (clear, warning, stop) with the
depth picture, compared with the true zone from the simulation. Two kinds of error, always reported apart:

- **missed**: the true zone is more serious than the one chosen (a hand in the stop zone, and the arm only
  slows or carries on). This is the one that matters for safety.
- **false alarm**: the chosen zone is more serious than the true one (the arm slows or stops for nothing).

Each rate comes with its 95% upper confidence bound (exact binomial, Clopper-Pearson), and is broken down by
condition: lighting, hand colour, sleeve, hand height, hand size, how much of the hand shows.

References: **perfect boxes** through the same rule (the true hand boxes from the labels: what the rule alone
costs); YOLO's boxes through the **worst-case height rule** (no depth); and the pictures where the hand is in
a zone but hidden from the camera (counted apart: the camera's limit, not the detector's).

The detector's confidence threshold is chosen on the validation pictures (the lowest-miss threshold whose
false alarms stay under ``MAX_FALSE_ALARMS``), never on the test pictures.

    python evaluate.py --weights runs/v1/weights/best.pt --name v1
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import mujoco

import box_zone
import dataset
import scene_plus
import zones

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "hands"
RESULTS = HERE / "results"
THRESHOLDS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
MAX_FALSE_ALARMS = 0.10  # share of pictures, on validation, when choosing the threshold
ZONE = {z.name: int(z) for z in zones.Zone}


def upper_bound(k: int, n: int, confidence: float = 0.95) -> float:
    """The exact (Clopper-Pearson) one-sided upper bound on a rate seen as k out of n."""
    if n == 0:
        return float("nan")
    if k >= n:
        return 1.0
    alpha = 1 - confidence

    def cdf(p: float) -> float:  # P(X <= k) for X ~ Binomial(n, p), in logs (n is in the thousands)
        if p <= 0.0:
            return 1.0
        if p >= 1.0:
            return 0.0
        lp, lq, ln = math.log(p), math.log1p(-p), math.lgamma(n + 1)
        return sum(
            math.exp(ln - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq)
            for i in range(k + 1)
        )

    lo, hi = k / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if cdf(mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def load_meta(split: str) -> list[dict]:
    return [json.loads(line) for line in (DATA / "meta" / f"{split}.jsonl").read_text().splitlines()]


def detect(weights: str, split: str, rows: list[dict], imgsz: int) -> dict[str, list[tuple[float, tuple]]]:
    """YOLO's boxes for every picture of a split, at a low threshold: {name: [(confidence, box), ...]}."""
    from ultralytics import YOLO

    model = YOLO(weights)
    out = {}
    names = [r["name"] for r in rows]
    for i in range(0, len(names), 64):
        batch = [str(DATA / "images" / split / f"{n}.jpg") for n in names[i : i + 64]]
        for name, res in zip(
            names[i : i + 64],
            model.predict(batch, imgsz=imgsz, conf=min(THRESHOLDS), device="mps", verbose=False),
            strict=True,
        ):
            out[name] = [
                (float(c), tuple(float(v) for v in b))
                for c, b in zip(res.boxes.conf.tolist(), res.boxes.xyxy.tolist(), strict=True)
            ]
    return out


def choose_zones(world, rows: list[dict], boxes_of, split: str | None = None) -> list[int]:
    """The zone for each picture from its boxes: with the depth picture if ``split`` is given (the safety
    feature's rule), else the worst-case height rule (no depth).
    """
    model, data = world
    out = []
    for r in rows:
        boxes = boxes_of(r)
        if split is None:
            out.append(int(box_zone.zone_of_boxes(model, data, boxes)))
        elif not boxes:
            out.append(0)
        else:
            depth = dataset.load_depth(DATA / "depth" / split / f"{r['name']}.png")
            out.append(int(box_zone.zone_of_boxes_depth(model, data, boxes, depth)))
    return out


def tally(rows: list[dict], chosen: list[int]) -> dict:
    """Missed and false alarms, over all pictures and per true zone."""
    true = [ZONE[r["zone"]] for r in rows]
    n = len(rows)
    in_zone = [i for i in range(n) if true[i] > 0]
    missed = [i for i in in_zone if chosen[i] < true[i]]
    missed_stop = [i for i in range(n) if true[i] == 2 and chosen[i] < 2]
    false = [i for i in range(n) if chosen[i] > true[i]]
    clear = [i for i in range(n) if true[i] == 0]
    false_on_clear = [i for i in clear if chosen[i] > 0]

    def rate(k, total):
        return {
            "k": len(k),
            "n": len(total),
            "rate": len(k) / len(total) if total else None,
            "upper95": upper_bound(len(k), len(total)),
        }

    return {
        "pictures": n,
        "missed (hand in a zone, less serious zone chosen)": rate(missed, in_zone),
        "missed stop (hand in stop zone, arm not stopped)": rate(
            missed_stop, [i for i in range(n) if true[i] == 2]
        ),
        "false alarm (more serious zone than true)": rate(false, list(range(n))),
        "false alarm when clear": rate(false_on_clear, clear),
    }


def conditions(r: dict) -> dict[str, str]:
    """The conditions a picture is counted under, for the breakdown."""
    out = {"lighting": r["lighting"], "goal": r["goal"]}
    if r.get("hand"):
        out["hand"] = r["hand"]
        out["sleeve"] = r["sleeve"]
        out["hand size"] = (
            "small (<0.95)" if r["size"] < 0.95 else "large (>1.05)" if r["size"] > 1.05 else "medium"
        )
        h = r["hand_height"]
        out["hand height"] = "low (<20 cm)" if h < 0.2 else "high (>35 cm)" if h > 0.35 else "middle"
        px = r["hand_pixels"]
        out["hand showing"] = "hidden (<12 px)" if px < 12 else "partly (<300 px)" if px < 300 else "fully"
    return out


def breakdown(rows: list[dict], chosen: list[int]) -> dict:
    groups: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(rows):
        for key, value in conditions(r).items():
            groups[key][value].append(i)
    out = {}
    for key, values in groups.items():
        out[key] = {}
        for value, idx in sorted(values.items()):
            t = tally([rows[i] for i in idx], [chosen[i] for i in idx])
            out[key][value] = {
                "pictures": len(idx),
                "missed": t["missed (hand in a zone, less serious zone chosen)"],
                "false alarm": t["false alarm (more serious zone than true)"],
            }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--weights", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    args = p.parse_args()

    model, _ = scene_plus.build(0)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)  # the safety camera is fixed: any scene gives its pose
    world = (model, data)

    val, test = load_meta("val"), load_meta("test")
    found_val = detect(args.weights, "val", val, args.imgsz)
    by_threshold = {}
    for th in THRESHOLDS:
        chosen = choose_zones(
            world, val, lambda r, th=th: [b for c, b in found_val[r["name"]] if c >= th], "val"
        )
        by_threshold[th] = tally(val, chosen)
    ok = [
        th
        for th in THRESHOLDS
        if by_threshold[th]["false alarm (more serious zone than true)"]["rate"] <= MAX_FALSE_ALARMS
    ]
    key = "missed (hand in a zone, less serious zone chosen)"
    threshold = min(ok or THRESHOLDS, key=lambda th: (by_threshold[th][key]["k"], -th))

    found = detect(args.weights, "test", test, args.imgsz)

    def yolo(r):
        return [b for c, b in found[r["name"]] if c >= threshold]

    def perfect(r):
        return [r["box"]] if r["box"] else []

    chosen = choose_zones(world, test, yolo, "test")
    hidden = [i for i, r in enumerate(test) if r["zone"] != "CLEAR" and r["box"] is None]
    result = {
        "weights": args.weights,
        "threshold (chosen on validation)": threshold,
        "validation by threshold": {str(k): v for k, v in by_threshold.items()},
        "test": tally(test, chosen),
        "test, perfect boxes (the rule alone)": tally(test, choose_zones(world, test, perfect, "test")),
        "test, no depth (worst-case height rule)": tally(test, choose_zones(world, test, yolo)),
        "test, hand in a zone but hidden from the camera": len(hidden),
        "test by condition": breakdown(test, chosen),
    }
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"evaluation_{args.name}.json"
    out.write_text(json.dumps(result, indent=1))
    for name in ("test", "test, perfect boxes (the rule alone)", "test, no depth (worst-case height rule)"):
        print(name)
        for k, v in result[name].items():
            if isinstance(v, dict):
                print(f"  {k}: {v['k']} of {v['n']} ({v['rate']:.2%}, 95% upper bound {v['upper95']:.2%})")
    print(f"threshold {threshold}; hidden hands in a zone: {len(hidden)}; saved {out}")


if __name__ == "__main__":
    main()
