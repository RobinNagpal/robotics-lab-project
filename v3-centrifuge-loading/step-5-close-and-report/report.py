"""Step 5, the report: how far out of balance the loaded rotor is, as the robot works it out and as the examiner
measures it.

- **The robot's number**: from what the robot itself read on the balance (Step 1's sample weights, Step 3's
  balance tube readings) and the holes it loaded (Step 4). This is all a real robot would have.
- **The examiner's number**: from the simulator's true weights and where the tubes really are
  (`grading.grade`). The two should agree within a few hundredths of a gram.

The lid stays open: as the centrifuge stands, the arm cannot reach to push it shut
(docs/lid-reach.md), so the report says so.

Run on its own, it reports Step 4's 20 seeds from their saved results (no simulation):

    cd step-5-close-and-report
    PYTHONPATH=$PWD/../step-2-plan:$PWD/../simulation-environment \\
        pixi run --manifest-path ../simulation-environment/pixi.toml python report.py
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import solver
from centrifuge_scene.settings import BALANCE_TOLERANCE

HERE = Path(__file__).parent
STEP4 = HERE.parent / "step-4-load-rotor" / "results"
LIMIT_G = BALANCE_TOLERANCE * 1000


@dataclass
class Report:
    holes: dict[int, float]  # the robot's own weights (g) in the holes it loaded
    robot_g: float
    examiner_g: float | None = None
    lid_shut: bool = False
    not_seated: dict = field(
        default_factory=dict
    )  # {hole: mm high, or None if nothing seen there}

    @property
    def balanced(self) -> bool:
        return self.robot_g <= LIMIT_G and not self.not_seated

    def text(self) -> str:
        if self.not_seated:
            bad = ", ".join(
                f"hole {k} " + ("empty" if v is None else f"{v:.0f} mm high")
                for k, v in sorted(self.not_seated.items())
            )
            mark = f"NOT done: tube not seated ({bad}); of the rest"
        else:
            mark = "Balanced" if self.balanced else "NOT balanced"
        line = f"{mark}: {self.robot_g:.2f} g out (limit {LIMIT_G:.1f} g), {len(self.holes)} tubes in holes "
        line += ", ".join(str(k) for k in sorted(self.holes))
        if self.examiner_g is not None:
            line += f"; examiner {self.examiner_g:.2f} g"
        return line + (
            "; lid shut"
            if self.lid_shut
            else "; lid left open (out of the arm's reach)"
        )


def robot_report(
    holes: dict[int, float],
    examiner_g: float | None = None,
    lid_shut: bool = False,
    seen: dict[int, float | None] | None = None,
) -> Report:
    """``holes``: {hole: the robot's own reading of the tube in it, g}. ``seen``: the rotor camera's check
    (``camera_check``); tubes it does not see seated are left out of the number and named in the report."""
    not_seated = {k: v for k, v in (seen or {}).items() if not seen_seated(v)}
    kept = {k: g for k, g in holes.items() if k not in not_seated}
    return Report(
        kept, round(solver.imbalance(kept), 3), examiner_g, lid_shut, not_seated
    )


# The rotor camera's check: a tube is seen seated if the top of what stands on its hole's axis (from the depth
# picture) is within SEEN_MM of where a seated tube's top would be. Higher: stuck short of the bottom (a tube
# seated tells -1 to +1 mm; one held 4 mm high, 4 mm). Nothing there: no top found.
SEEN_MM = 3.0
WINDOW = 45  # pixels either side of where the tube's top should be


def camera_check(model, data, holes: dict[int, bool]) -> dict[int, float | None]:
    """For each hole in ``holes`` ({hole: capped}), how far (mm, along the hole's axis) the top of whatever stands
    on the hole's axis is above where a seated tube's top would be: about 0 seated, positive stuck high, None
    if nothing stands there. The arm must be out of the view. Uses the bench layout (the holes, the camera) and
    the rotor camera's depth picture only."""
    import mujoco
    from centrifuge_scene import scene
    from centrifuge_scene.settings import IMAGE_SIZE, TUBE

    cam = model.camera("rotor").id
    pos, rot = data.cam_xpos[cam].copy(), data.cam_xmat[cam].reshape(3, 3).copy()
    w, h = IMAGE_SIZE
    focal = (h / 2) / np.tan(np.radians(model.cam_fovy[cam]) / 2)
    with mujoco.Renderer(model, height=h, width=w) as r:
        r.enable_depth_rendering()
        r.update_scene(data, camera="rotor")
        depth = r.render().copy()
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    # Every pixel as a point in the world.
    cam_pts = np.stack(
        [(us - w / 2) / focal * depth, -(vs - h / 2) / focal * depth, -depth], axis=-1
    )
    world = pos + cam_pts @ rot.T
    out = {}
    for k, capped in holes.items():
        site = data.site(scene.slot_site(k))
        axis = site.xmat.reshape(3, 3)[:, 2]
        length = TUBE.capped_height if capped else TUBE.height
        top = site.xpos + axis * length
        c = rot.T @ (top - pos)
        u, v = round(w / 2 + focal * c[0] / -c[2]), round(h / 2 - focal * c[1] / -c[2])
        win = world[
            max(0, v - WINDOW) : v + WINDOW, max(0, u - WINDOW) : u + WINDOW
        ].reshape(-1, 3)
        rel = win - site.xpos
        along = rel @ axis
        radial = np.linalg.norm(rel - np.outer(along, axis), axis=1)
        near = (radial < (TUBE.cap_radius if capped else TUBE.radius) + 0.001) & (
            along > 0.05
        )
        out[k] = (
            round(float(along[near].max() - length) * 1000, 1) if near.any() else None
        )
    return out


def seen_seated(offset: float | None) -> bool:
    return offset is not None and abs(offset) <= SEEN_MM


def from_plan(meta: dict, loaded: set[int] | None = None) -> dict[int, float]:
    """The robot's weights per hole from a saved plan: samples as Step 1 read them, balance tubes as Step 3 read
    them after filling. ``loaded``: only these holes (default: every planned hole)."""
    read = {f["balance_tube"]: f["read_g"] for f in meta["fills"]}
    out = {}
    for t in meta["plan"]:
        if loaded is not None and t["hole"] not in loaded:
            continue
        out[t["hole"]] = read[t["name"]] if t["balance"] else t["grams"]
    return out


def main() -> None:
    runs = json.loads((STEP4 / "evaluation_plain.json").read_text())
    rows = []
    for r in runs:
        meta = json.loads((STEP4 / "filled" / f"seed{r['seed']}.json").read_text())
        loaded = {t["hole"] for t in r["tubes"] if not t.get("error")}
        rep = robot_report(from_plan(meta, loaded), r["imbalance_g"])
        rows.append(
            {
                "seed": r["seed"],
                "scenario": r["scenario"],
                **asdict(rep),
                "text": rep.text(),
            }
        )
        print(f"seed {r['seed']:2d} {r['scenario']:8s} {rep.text()}")
    diff = [abs(x["robot_g"] - x["examiner_g"]) for x in rows]
    agree = sum((x["robot_g"] <= LIMIT_G) == (x["examiner_g"] <= LIMIT_G) for x in rows)
    print(
        f"\nrobot vs examiner: median {statistics.median(diff):.3f} g apart, worst {max(diff):.3f} g; "
        f"same verdict on {agree} of {len(rows)}"
    )
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "report_step4_seeds.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
