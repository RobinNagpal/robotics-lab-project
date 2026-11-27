"""The test scenes for Step 1, where every arm (programmed or learned) is measured. Never recorded.

- **A, normal:** seeds 0 to 19 as built (60 sample tubes).
- **B, changed:** the same 20 seeds, with the sample rack moved 1 to 3 cm (its tubes with it) and the
  balance moved 1 to 2 cm, each in a random direction. Nobody tells the arm: the programmed arm still goes to
  the cells and the holder where the layout says they are, so it fails by design; a model that steers by its
  cameras might not. Moves that bring the rack within 1.5 cm of the balance or the other rack are redrawn.

The changes are drawn once from a fixed seed and saved in ``test_scenes.json``, so every model faces exactly
the same test.

    P="pixi run --manifest-path ../simulation-environment/pixi.toml"
    PYTHONPATH=$PWD:$PWD/../simulation-environment $P python test_scenes.py   # (re)writes test_scenes.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np
from centrifuge_scene import liquid, scene
from centrifuge_scene.settings import LAYOUT, RACK, SCALE

SEEDS = range(20)
FILE = Path(__file__).parent / "test_scenes.json"
# The scene's tubes start asleep about 1 mm into the rack's floor and tip seats; the first touch wakes the
# whole rack, and every tube in it jumps out of the overlap at once. So every test scene starts awake and
# settles first, as a real bench is at rest before the arm starts (bodies fall asleep again once still).
SETTLE = 1.5  # s
GAP = 0.015  # m kept between a moved rack and anything else on the bench


def _footprint(xy, half) -> tuple[float, float, float, float]:
    return (xy[0] - half[0], xy[0] + half[0], xy[1] - half[1], xy[1] + half[1])


def _apart(a, b, gap: float) -> bool:
    """Two (x0, x1, y0, y1) boxes at least ``gap`` apart."""
    return a[1] + gap <= b[0] or b[1] + gap <= a[0] or a[3] + gap <= b[2] or b[3] + gap <= a[2]


def draw_changes(seed: int = 2026) -> list[dict]:
    rng = np.random.default_rng(seed)
    rack_half = (RACK.rows * RACK.pitch / 2, RACK.columns * RACK.pitch / 2)
    supply = _footprint(LAYOUT.supply_rack, rack_half)
    out = []
    for s in SEEDS:
        while True:
            a, r = rng.uniform(0, 2 * math.pi), rng.uniform(0.01, 0.03)
            rack = np.array([r * math.cos(a), r * math.sin(a)])
            a, r = rng.uniform(0, 2 * math.pi), rng.uniform(0.01, 0.02)
            balance = np.array([r * math.cos(a), r * math.sin(a)])
            moved_rack = _footprint(np.array(LAYOUT.sample_rack) + rack, rack_half)
            moved_balance = _footprint(np.array(LAYOUT.scale) + balance, SCALE.base[:2])
            if _apart(moved_rack, moved_balance, GAP) and _apart(moved_rack, supply, GAP):
                break
        out.append(
            {
                "seed": s,
                "rack_shift_m": np.round(rack, 4).tolist(),
                "balance_shift_m": np.round(balance, 4).tolist(),
            }
        )
    return out


def load_changes() -> dict[int, dict]:
    return {c["seed"]: c for c in json.loads(FILE.read_text())}


def shift_rack(model: mujoco.MjModel, data: mujoco.MjData, truth, rack: str, shift: np.ndarray) -> None:
    """Move a rack and the tubes standing in it (with the balls inside them) sideways by ``shift``."""
    tubes = [n for n, (r, _) in truth.cells.items() if r == rack]
    bodies = [model.body(rack).id] + [model.body(n).id for n in tubes]
    tube_xy = [data.xpos[model.body(n).id][:2].copy() for n in tubes]
    for b in liquid.ball_body_ids(model):
        # A ball goes with the tube it is in (within a tube's radius of its axis).
        if any(np.linalg.norm(data.xpos[b][:2] - xy) < 0.02 for xy in tube_xy):
            bodies.append(b)
    for b in bodies:
        adr = model.jnt_qposadr[model.body_jntadr[b]]
        data.qpos[adr : adr + 2] += shift
    mujoco.mj_forward(model, data)


def build(seed: int, changed: bool) -> tuple[mujoco.MjModel, mujoco.MjData, object]:
    """Test scene A (``changed`` False) or B for ``seed``, settled and ready for the arm."""
    model, truth = scene.build(seed, awake=True)
    data = mujoco.MjData(model)
    if changed:
        c = load_changes()[seed]
        for name in ("scale", "scale/pan"):
            model.body_pos[model.body(name).id][:2] += c["balance_shift_m"]
    scene.reset(model, data)
    if changed:
        shift_rack(model, data, truth, "sample_rack", np.array(c["rack_shift_m"]))
    scene.settle(model, data, SETTLE)
    return model, data, truth


def main() -> None:
    changes = draw_changes()
    FILE.write_text(json.dumps(changes, indent=1))
    for c in changes:
        rack = np.linalg.norm(c["rack_shift_m"]) * 100
        bal = np.linalg.norm(c["balance_shift_m"]) * 100
        print(f"seed {c['seed']:2d}: sample rack moved {rack:.1f} cm, balance {bal:.1f} cm")


if __name__ == "__main__":
    main()
