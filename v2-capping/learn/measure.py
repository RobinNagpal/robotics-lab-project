"""Measure an unscrewing from its recording, the same way for the teacher and for a learned policy.

From the first quarter turn (the cap gripped) to the end of the recording (the cap lifted away):

- **seconds**;
- **grips**: how many times Arm B's fingers were closed on the cap (one per quarter turn);
- **lift tests**: grips during which the hand rose more than RISE (the test lifts 6 mm; turning raises
  it only a quarter of the thread's 3.2 mm pitch). The last one is the test that found the cap off and
  the lift away that followed.

Hand heights come from the commanded joint targets, through the arm's kinematics.

    python -m learn.measure runs/teacher_A_rec runs/learn/act_A/eval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from lab_scene import scene as lab

from uncap.episode import CAP_ARM, build

from .data import TURN

CLOSED = 170.0  # gripper command: about 199 closed on the cap, 138 open
RISE = 0.003  # m


class Heights:
    """The pinch point's height for Arm B's joint values (one model, built once)."""

    def __init__(self):
        self.model, _ = build(300)
        self.data = mujoco.MjData(self.model)
        self.q = lab.arm_joint_ids(self.model, CAP_ARM)
        self.site = self.model.site(lab.pinch_site(CAP_ARM)).id

    def __call__(self, qs: np.ndarray) -> np.ndarray:
        out = np.empty(len(qs))
        for i, q in enumerate(qs):
            self.data.qpos[self.q] = q
            mujoco.mj_kinematics(self.model, self.data)
            out[i] = self.data.site_xpos[self.site][2]
        return out


def measure(path: Path, heights: Heights) -> dict:
    f = np.load(path)
    meta = json.loads(str(f["meta"]))
    first = int(np.argmax(f["phase"] == TURN))
    t = f["time"][first:]
    closed = f["action_grip"][first:] > CLOSED
    z = heights(f["action_q"][first:])
    # The closed stretches: [start, end) sample ranges.
    edges = np.flatnonzero(np.diff(np.r_[0, closed.astype(int), 0]))
    spans = list(zip(edges[::2], edges[1::2], strict=True))
    tests = sum(z[a:b].max() - z[a] > RISE for a, b in spans)
    return {
        "run": path.stem,
        "seed": meta.get("seed"),
        "done": meta.get("done"),
        "seconds": round(float(t[-1] - t[0]), 2),
        "grips": len(spans),
        "lift_tests": int(tests),
        "cap_off_at_end": not bool(f["on_thread"][-1]),
    }


def folder(path: Path, heights: Heights | None = None) -> list[dict]:
    heights = heights or Heights()
    return [measure(p, heights) for p in sorted(path.glob("seed*_size*.npz"))]


def summary(rows: list[dict]) -> dict:
    def stats(key):
        v = [r[key] for r in rows if r["done"]]
        return {"mean": round(float(np.mean(v)), 2), "min": min(v), "max": max(v)} if v else {}

    return {
        "runs": len(rows),
        "done": sum(bool(r["done"]) for r in rows),
        "seconds": stats("seconds"),
        "grips": stats("grips"),
        "lift_tests": stats("lift_tests"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("folders", type=Path, nargs="+")
    args = parser.parse_args()
    heights = Heights()
    for f in args.folders:
        rows = folder(f, heights)
        print(f, json.dumps(summary(rows), indent=1))


if __name__ == "__main__":
    main()
