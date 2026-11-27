"""Check a scripted-hand run (``run_guarded.py --hand scripted``) visit by visit, from its safety logs alone.

For every visit of the hand (4 s in the warning zone, then 2 s in the stop zone):

- **slowed**: the first frame in the warning visit is logged as the warning zone, and the arm goes to slow;
- **held**: the first frame in the stop visit is logged as the stop zone, and the arm holds;
- **carried on**: the arm goes back to full speed at the first frame 1 s after the last stop-zone frame (not
  earlier, and not more than one frame later).

    $PY check_scripted.py results/guarded_scripted [results/calibrate_no_hand]

"v3" in the output is the same seed with no hand (the guard on, which is identical to v3 as it is today).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from run_guarded import VISIT_EVERY, VISIT_FIRST  # first: it puts v3's folders on the import path
from guard import CLEAR_FOR, HOLD, RUN, SLOWED  # noqa: I001
from safety_log import read
from standin import repeating_visits

FRAME = 0.1


def check_seed(log_path: Path) -> dict:
    entries = [e for e in read(log_path) if e.get("kind") == "detector"]
    end = entries[-1]["t"] if entries else 0.0
    visits = repeating_visits(VISIT_FIRST, VISIT_EVERY, end)
    out = {"visits": 0, "slowed": 0, "held": 0, "carried_on": 0, "problems": []}
    for warn, stop in zip(visits[::2], visits[1::2], strict=True):
        if stop.end > end:
            break
        out["visits"] += 1
        # The visits start on a frame's time, where rounding decides whether that frame sees the hand: take the
        # first frame clearly inside each visit.
        first_w = next(e for e in entries if e["t"] > warn.start + 1e-6)
        first_s = next(e for e in entries if e["t"] > stop.start + 1e-6)
        if first_w["zone"] == "warning" and first_w["action"] == SLOWED:
            out["slowed"] += 1
        else:
            out["problems"].append(f"t={first_w['t']:.1f}: warning visit logged {first_w['zone']}/{first_w['action']}")
        if first_s["zone"] == "stop" and first_s["action"] == HOLD:
            out["held"] += 1
        else:
            out["problems"].append(f"t={first_s['t']:.1f}: stop visit logged {first_s['zone']}/{first_s['action']}")
        last_stop = max(e["t"] for e in entries if e["zone"] == "stop" and e["t"] < stop.end + 1e-9)
        back = next((e for e in entries if e["t"] > stop.start and e["action"] == RUN), None)
        if back is not None and last_stop + CLEAR_FOR - 1e-6 <= back["t"] <= last_stop + CLEAR_FOR + FRAME + 1e-6:
            out["carried_on"] += 1
        else:
            out["problems"].append(
                f"t={stop.start:.1f}: back to run at {back and round(back['t'], 2)}, last stop frame {last_stop:.2f}"
            )
    return out


def main() -> None:
    folder = Path(sys.argv[1])
    rows = {r["seed"]: r for r in json.loads((folder / "results.json").read_text())}
    # The same jobs with the guard on and no hand: identical to v3 as it is today (docs/safety-layer.md).
    base = Path(sys.argv[2]) if len(sys.argv) > 2 else folder.parent / "calibrate_no_hand"
    v3 = {r["seed"]: r for r in json.loads((base / "results.json").read_text())}
    total = {"visits": 0, "slowed": 0, "held": 0, "carried_on": 0}
    summary = []
    for log in sorted(folder.glob("log_seed*.jsonl")):
        seed = int(log.stem[-2:])
        c = check_seed(log)
        r, ref = rows[seed], v3[seed]
        g = r.get("guard", {})
        stops = [s for s in g.get("stops", []) if s["seconds"] is not None]
        row = {
            "seed": seed,
            "scenario": r.get("scenario"),
            **{k: c[k] for k in total},
            "problems": c["problems"],
            "loaded_and_balanced": bool(r.get("all_samples_loaded") and r.get("balanced") and not r.get("error")),
            "error": r.get("error", ""),
            "spilled": r.get("balls_spilled"),
            "spilled_v3": ref["balls_spilled"],
            "imbalance_g": r.get("examiner_g"),
            "imbalance_g_v3": ref["examiner_g"],
            "sim_seconds": g.get("sim_seconds"),
            "sim_seconds_v3": ref["guard"]["sim_seconds"],
            "held_seconds": g.get("held_seconds"),
            "slow_seconds": g.get("slow_seconds"),
            "stop_seconds_max": max((s["seconds"] for s in stops), default=None),
            "stop_mm_max": round(1000 * max((s["distance"] for s in stops), default=0.0), 1),
            "log_ok": not g.get("log_problems"),
            "replay_ok": not g.get("replay_disagrees"),
        }
        for k in total:
            total[k] += c[k]
        summary.append(row)
        print(
            f"seed {seed:2d} {row['scenario']:8s} visits {c['visits']:2d}: slowed {c['slowed']}, held {c['held']}, "
            f"carried on {c['carried_on']} | loaded+balanced {row['loaded_and_balanced']} spilled {row['spilled']} "
            f"(v3 {row['spilled_v3']}) | job {row['sim_seconds']} s (v3 {row['sim_seconds_v3']}) | worst stop "
            f"{row['stop_seconds_max']} s {row['stop_mm_max']} mm | log {row['log_ok']} replay {row['replay_ok']}"
            + (f"\n   {c['problems'][:3]}" if c["problems"] else "")
        )
    print(f"\nall seeds: {total}")
    (folder / "check.json").write_text(json.dumps({"total": total, "seeds": summary}, indent=1))


if __name__ == "__main__":
    main()
