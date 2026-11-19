"""Gemini against SAM 3 on the seeds both have answered, from saved answers only (never calls Google).

    cd step-0-find
    PYTHONPATH=$PWD:$PWD/../simulation-environment \
      pixi run --manifest-path ../simulation-environment/pixi.toml python compare.py --seeds 4
"""

from __future__ import annotations

import argparse

import evaluate
import gemini
import sam
from bench import take_pictures
from find import find_separate, find_together


def saved_gemini(pngs, prompt):
    answer = gemini.ask(pngs, prompt)
    if not answer.cached:
        raise SystemExit("compare.py only uses saved answers, and this one is not saved")
    return answer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    args = ap.parse_args()
    s3 = sam.Sam3()
    rows = {"gemini": [], "sam3": []}
    for seed in range(args.seeds):
        views, (model, data, truth) = take_pictures(seed, evaluate.CAMERAS)
        table = evaluate.true_table(model, data, truth)
        s3.camera_of = {v.png: n for n, v in views.items()}
        for name, ask in (("gemini", saved_gemini), ("sam3", s3)):
            for config, (mode, cams) in evaluate.CONFIGS.items():
                if mode == "together" and name == "sam3":
                    continue
                find = find_separate if mode == "separate" else find_together
                found, answers, per_camera = find({c: views[c] for c in cams}, ask)
                r = evaluate.score(found, table, per_camera, views)
                r.update(
                    config=config,
                    seed=seed,
                    seconds=sum(a.seconds for a in answers),
                    dollars=sum(a.dollars for a in answers),
                )
                rows[name].append(r)
    for name, rs in rows.items():
        evaluate.print_tables(rs, f"{name}, seeds 0-{args.seeds - 1}")


if __name__ == "__main__":
    main()
