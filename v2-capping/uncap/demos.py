"""The demonstrations recorded by record.py: what is in them, summed up.

    python -m uncap.demos runs/demos

Prints (and writes to <folder>/summary.json): how many runs, how many were right, the samples and hours,
the caps' tightness and quarter turns, the time each part of the loop takes, and how clearly the wrist
force tells "on the thread" from "off" in the lift test (zeroed with each run's tare).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: Path) -> tuple[dict, dict]:
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    return meta, {k: z[k] for k in z.files if k != "meta"}


def _range(values) -> list:
    v = np.asarray(values, dtype=float)
    return [round(float(v.min()), 3), round(float(v.max()), 3)] if v.size else []


def quarter_counts(runs: list) -> dict:
    """How many runs took each number of quarter turns."""
    counts: dict[str, int] = {}
    for m, _ in runs:
        counts[str(m["quarters"])] = counts.get(str(m["quarters"]), 0) + 1
    return dict(sorted(counts.items()))


def summary(folder: Path) -> dict:
    files = sorted(folder.glob("seed*.npz"))
    runs = [load(f) for f in files]
    right = [(m, a) for m, a in runs if m.get("done")]
    phases = runs[0][0]["phases"] if runs else []
    samples = sum(m["samples"] for m, _ in runs)
    seconds = {p: [] for p in phases}
    on_pull, off_pull, seal_twist, later_twist = [], [], [], []
    for m, a in right:
        dt = 1.0 / m["rate"]
        for i, p in enumerate(phases):
            n = int((a["phase"] == i).sum())
            if n:
                seconds[p].append(n * dt)
        fz = a["wrist_force"][:, 2] - m["tare"]["force"][2]
        tz = a["wrist_torque"][:, 2] - m["tare"]["torque"][2]
        test = a["phase"] == phases.index("test")
        # the lift test's reading: the end of its hold (the last 0.1 s, as the arm reads it), each quarter
        for q in np.unique(a["quarter"][test]):
            sel = np.flatnonzero(test & (a["quarter"] == q))[-max(1, int(0.1 * m["rate"])) :]
            (on_pull if a["on_thread"][sel].all() else off_pull).append(float(-fz[sel].mean()))
        turn = a["phase"] == phases.index("turn")
        for q in np.unique(a["quarter"][turn]):
            peak = float(np.abs(tz[turn & (a["quarter"] == q)]).max())
            (seal_twist if q == 1 else later_twist).append(peak)
    return {
        "runs": len(runs),
        "right": len(right),
        "not_right": [(m["seed"], m.get("failure")) for m, _ in runs if not m.get("done")],
        "samples": samples,
        "hours": round(samples / (runs[0][0]["rate"] if runs else 1) / 3600, 2),
        "tightness_Nm": _range([m["variation"]["breakaway_Nm"] for m, _ in runs]),
        "quarters": quarter_counts(runs),
        "seconds_per_run_in": {p: round(float(np.mean(v)), 2) for p, v in seconds.items() if v},
        "lift_test_pull_N": {
            "on_thread": _range(on_pull),
            "off": _range(off_pull),
            "tests": len(on_pull) + len(off_pull),
        },
        "wrist_twist_peak_Nm": {
            "first_quarter_seal": _range(seal_twist),
            "later_quarters": _range(later_twist),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    s = summary(args.folder)
    (args.folder / "summary.json").write_text(json.dumps(s, indent=1, default=str))
    print(json.dumps(s, indent=1, default=str))


if __name__ == "__main__":
    main()
