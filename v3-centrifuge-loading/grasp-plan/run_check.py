"""Run every grip check and print the tables for results.md.

    pixi run --manifest-path ../simulation-environment/pixi.toml python run_check.py [--workers 4]

Parts:
  A. picking from (and putting back in) a rack cell, every tube on 8 seeds (axial grip on the cap/rim)
  B. the balance holder (put in, let go, pick up again)
  C. every rotor hole, all other holes full: slide in 6 cm along the hole's axis, let go, back out
  D. the pour: the water tube held from the side, tipped from 40 to 130 degrees over a balance tube
  E. picking the water tube from its rack cell from the side
  F. pushing the lid shut
  G. the carries between places (joint-space moves with the tube in the fingers)
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np
from centrifuge_scene import centrifuge, scene
from centrifuge_scene.settings import LID_CLOSED, TUBE

from grasp_check import (
    HOLD_CAP,
    HOLD_RIM,
    MARGIN,
    OPEN,
    SAFE,
    UP,
    Check,
    Pose,
    TubePose,
    World,
    axial_grip,
    hold_width,
    line,
    rot_about,
    side_grip,
    stand_in_and_let_go,
    walk,
)

OUT = Path(__file__).parent / "results"
SEEDS = range(8)
CAP_DEPTH = 0.004  # pinch point 4 mm under the cap's top: the pads cover the whole 12 mm cap
RIM_DEPTH = 0.006  # open tubes: 6 mm under the rim
UNDER_CAP = 0.035  # the grip planned in Step 1's approach.md: pads on the body, just under the cap
RACK_TURNS = [math.radians(t) for t in range(0, 180, 15)]
ROTOR_TURNS = [math.radians(t) for t in range(0, 180, 5)]


def depth_for(capped: bool) -> float:
    return CAP_DEPTH if capped else RIM_DEPTH


def as_dict(c: Check) -> dict:
    return {"ok": c.ok, "gap_mm": round(c.gap * 1000, 1), "what": c.what, "note": c.note}


# --- A. racks --------------------------------------------------------------------------------------------


def part_a(seed: int) -> list[dict]:
    w = World(seed)
    rows = []
    for name in w.truth.tubes:
        seat = w.tube_pose(name)
        rack = w.model.body(w.truth.cells[name][0]).id
        checks = [
            stand_in_and_let_go(w, name, seat, 0.10, depth_for(seat.capped), t, None, (rack,))
            for t in RACK_TURNS
        ]
        rows.append(
            {
                "seed": seed,
                "tube": name,
                "cell": w.truth.cells[name],
                "turns_ok": [
                    round(math.degrees(t)) for t, c in zip(RACK_TURNS, checks, strict=False) if c.ok
                ],
                "best": as_dict(max(checks, key=lambda c: c.gap)),
            }
        )
    return rows


# --- B. the balance holder -----------------------------------------------------------------------------


def part_b() -> list[dict]:
    w = World(0)
    rows = []
    for name in ("sample0", "blank0"):
        seat = w.holder_pose(w.truth.capped[name])
        ignore = (w.model.body("scale/pan").id, w.model.body("scale").id)
        checks = [
            stand_in_and_let_go(w, name, seat, 0.10, depth_for(seat.capped), t, None, ignore)
            for t in RACK_TURNS
        ]
        rows.append(
            {
                "tube": name,
                "turns_ok": [
                    round(math.degrees(t)) for t, c in zip(RACK_TURNS, checks, strict=False) if c.ok
                ],
                "best": as_dict(max(checks, key=lambda c: c.gap)),
            }
        )
    return rows


# --- C. the rotor ------------------------------------------------------------------------------------------


def part_c(job: tuple[int, str, float]) -> dict:
    """One hole, one held tube, one grip depth; seed 1 has five capped samples to fill the other holes."""
    k, held, depth = job
    w = World(1)
    w.fill_rotor(k, held)
    w.base_qpos = w.data.qpos.copy()
    seat = w.seated_pose(k, w.truth.capped[held])
    ref = w.data.site(scene.slot_site(k)).xmat.reshape(3, 3)[:, 0]  # along the rim: 0 = fingers tangential
    ignore = (w.hole_body[k], w.rotor)
    checks = [stand_in_and_let_go(w, held, seat, 0.06, depth, t, ref, ignore) for t in ROTOR_TURNS]
    capped = w.truth.capped[held]
    return {
        "hole": k,
        "capped": capped,
        "depth_mm": depth * 1000,
        "turns_ok": [round(math.degrees(t)) for t, c in zip(ROTOR_TURNS, checks, strict=False) if c.ok],
        "gaps_mm": [round(c.gap * 1000, 1) for c in checks],
        "best": as_dict(max(checks, key=lambda c: c.gap)),
    }


# --- D. the pour --------------------------------------------------------------------------------------------


def pour_poses(
    w: World, receiving: TubePose, pour_dir: np.ndarray, approach_deg: float, below_rim: float, angles
) -> list[Pose]:
    r_rim = TUBE.outline[-1][1]
    u = np.cross(UP, pour_dir)
    a0 = math.cos(math.radians(approach_deg)) * pour_dir + math.sin(math.radians(approach_deg)) * u
    lip = receiving.top + np.array([0, 0, 0.015]) - 0.004 * pour_dir
    poses = []
    for th in angles:
        R = rot_about(u, math.radians(th))
        rot = R
        z_t = R @ UP
        tip = lip - TUBE.height * z_t - r_rim * (R @ pour_dir)
        tube = TubePose(tip, rot, False)
        pos, prot = side_grip(tube, below_rim, R @ a0)
        poses.append(Pose(pos, prot, 2 * TUBE.radius, "water", tube, ()))
    return poses


def part_d(job: tuple[float, float]) -> list[dict]:
    yaw_deg, below_rim = job
    w = World(0)
    receiving = w.holder_pose(False)
    w.place("blank0", receiving)
    w.base_qpos = w.data.qpos.copy()
    pour_dir = np.array([math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg)), 0.0])
    rows = []
    for approach in (0, 45, 90, -45, -90):
        w.reset()
        c = walk(w, pour_poses(w, receiving, pour_dir, approach, below_rim, range(50, 135, 5)))
        rows.append(
            {"pour_yaw": yaw_deg, "below_rim_mm": below_rim * 1000, "approach": approach, **as_dict(c)}
        )
    return rows


# --- E. the water tube from the side ---------------------------------------------------------------------


def part_e(job: tuple[int, float]) -> dict:
    seed, below_rim = job
    w = World(seed)
    seat = w.tube_pose("water")
    rack = w.model.body(w.truth.cells["water"][0]).id
    ok = []
    best = None
    for yaw in range(0, 360, 15):
        a = np.array([math.cos(math.radians(yaw)), math.sin(math.radians(yaw)), 0.0])
        pos, rot = side_grip(seat, below_rim, a)
        poses = [Pose(p, rot, OPEN) for p in line(pos - 0.10 * a, pos)]
        for tip in line(seat.tip, seat.tip + 0.10 * UP):
            tp = TubePose(tip, seat.rot, False)
            poses.append(Pose(side_grip(tp, below_rim, a)[0], rot, 2 * TUBE.radius, "water", tp, (rack,)))
        w.reset()
        c = walk(w, poses)
        if c.ok:
            ok.append(yaw)
        if best is None or c.gap > best.gap:
            best = c
    return {
        "seed": seed,
        "below_rim_mm": below_rim * 1000,
        "cell": w.truth.cells["water"],
        "approaches_ok": ok,
        "best": as_dict(best),
    }


# --- F. the lid --------------------------------------------------------------------------------------------


def part_f() -> dict:
    w = World(0)
    m, d = w.model, w.data
    adr = centrifuge.lid_qpos_adr(m)
    hinge = m.joint("centrifuge/lid").id
    rows = []
    for turn in range(0, 180, 30):
        q = None
        best, what, ok = MARGIN, "", True
        for frac in np.linspace(0, 1, 13):
            w.reset()
            d.qpos[adr] = LID_CLOSED * frac
            mujoco.mj_forward(m, d)
            point = d.site("centrifuge/lid").xpos.copy()
            lever = point - d.xanchor[hinge]
            push = np.cross(d.xaxis[hinge], lever)
            push /= np.linalg.norm(push)
            if np.cross(lever, push) @ d.xaxis[hinge] > 0:
                push = -push
            pos = point - 0.008 * push  # the fingertips 8 mm short of the lid: the push itself is fine
            y = np.cross(push, d.xaxis[hinge])
            y = rot_about(push, math.radians(turn)) @ (y / np.linalg.norm(y))
            rot = np.column_stack([np.cross(y, push), y, push])
            base = d.qpos.copy()
            w.base_qpos = base
            q2 = w.ik(pos, rot, q)
            if q2 is None or (q is not None and np.abs(q2 - q).max() > 0.5):
                ok = False
                what = "out of reach" if q2 is None else "flip"
                break
            q = q2
            g, wh = w.gap(q, 0.0)
            if g < best:
                best, what = g, wh
        rows.append({"turn": turn, "ok": ok and best >= SAFE, "gap_mm": round(best * 1000, 1), "what": what})
    return {"rows": rows}


# --- G. carries ---------------------------------------------------------------------------------------------


def tube_from_pinch(pos, rot, depth, capped) -> TubePose:
    z_t = -rot[:, 2]
    length = TUBE.capped_height if capped else TUBE.height
    top = pos + depth * z_t
    x = rot[:, 1]
    return TubePose(top - length * z_t, np.column_stack([x, np.cross(z_t, x), z_t]), capped)


def best_turn(w, name, seat, back, ref, ignore, turns) -> float | None:
    checks = [
        (stand_in_and_let_go(w, name, seat, back, depth_for(seat.capped), t, ref, ignore), t) for t in turns
    ]
    good = [(c.gap, t) for c, t in checks if c.ok]
    return max(good)[1] if good else None


def carry(w: World, name: str, a: tuple, b: tuple, capped: bool, via_height: float = 0.0) -> Check:
    """Joint-space move from pinch pose a to b holding ``name``; optionally up through a raised middle."""
    depth = depth_for(capped)
    qa, qb = w.ik(*a), None
    if qa is None:
        return Check(False, -1, "", "start out of reach")
    qb = w.ik(*b, qa)
    if qb is None:
        qb = w.ik(*b)
    if qb is None:
        return Check(False, -1, "", "end out of reach")
    knots = [qa, qb]
    if via_height:
        mid_pos = (a[0] + b[0]) / 2 + np.array([0, 0, via_height])
        qm = w.ik(mid_pos, a[1], qa)
        if qm is not None:
            knots = [qa, qm, qb]
    best, what = MARGIN, ""
    m = w.model
    for q0, q1 in zip(knots, knots[1:], strict=False):
        for s in np.linspace(0, 1, 40):
            q = q0 + (q1 - q0) * s
            w.data.qpos[w.arm_q] = q
            mujoco.mj_kinematics(m, w.data)
            site = w.data.site(scene.pinch_site())
            tp = tube_from_pinch(site.xpos.copy(), site.xmat.reshape(3, 3).copy(), depth, capped)
            w.place(name, tp)
            g, wh = w.gap(q, hold_width(tp, depth), name)
            if g < best:
                best, what = g, wh
    return Check(best >= SAFE, best, what)


def part_g(seed: int) -> list[dict]:
    w = World(seed)
    rows = []
    holder_ignore = (w.model.body("scale/pan").id, w.model.body("scale").id)
    for name in [n for n in w.truth.tubes if n != "water"]:
        capped = w.truth.capped[name]
        depth = depth_for(capped)
        seat = w.tube_pose(name)
        rack = w.model.body(w.truth.cells[name][0]).id
        t_rack = best_turn(w, name, seat, 0.10, None, (rack,), RACK_TURNS)
        holder = w.holder_pose(capped)
        t_hold = best_turn(w, name, holder, 0.10, None, holder_ignore, RACK_TURNS)
        if t_rack is None or t_hold is None:
            rows.append(
                {"seed": seed, "tube": name, "move": "-", "ok": False, "note": "no turn at rack/holder"}
            )
            continue
        lifted = TubePose(seat.tip + 0.10 * UP, seat.rot, capped)
        above_holder = TubePose(holder.tip + 0.10 * UP, holder.rot, capped)
        a = axial_grip(lifted, depth, t_rack)
        b = axial_grip(above_holder, depth, t_hold)
        w.reset()
        for via in (0.0, 0.10):
            c = carry(w, name, a, b, capped, via)
            if c.ok:
                break
        rows.append({"seed": seed, "tube": name, "move": "rack -> holder", "via_cm": via * 100, **as_dict(c)})
        # From the holder (balance tubes) or the rack (samples) to the first free hole, 6 cm back.
        for k in (0, 3):
            w.reset()
            hole_seat = w.seated_pose(k, capped)
            ref = w.data.site(scene.slot_site(k)).xmat.reshape(3, 3)[:, 0]
            t_hole = best_turn(w, name, hole_seat, 0.06, ref, (w.hole_body[k], w.rotor), ROTOR_TURNS[::2])
            if t_hole is None:
                rows.append(
                    {"seed": seed, "tube": name, "move": f"-> hole {k}", "ok": False, "note": "no turn"}
                )
                continue
            back = TubePose(hole_seat.tip + 0.06 * hole_seat.axis, hole_seat.rot, capped)
            start = b if not capped else a
            w.reset()
            for via in (0.0, 0.10, 0.20):
                c = carry(w, name, start, axial_grip(back, depth, t_hole, ref), capped, via)
                if c.ok:
                    break
            src = "holder" if not capped else "rack"
            rows.append(
                {"seed": seed, "tube": name, "move": f"{src} -> hole {k}", "via_cm": via * 100, **as_dict(c)}
            )
    return rows


# --- H. rotor holes by which neighbours are full ------------------------------------------------------------

NEIGHBOURS = {"both empty": (), "left full": (-1,), "right full": (1,), "both full": (-1, 1)}


def part_h(job: tuple[int, str, str, float]) -> dict:
    """Hole k with the given neighbours full, and the opposite hole full; seed 5 has 5 samples."""
    k, config, held_kind, lid_deg = job
    w = World(5)
    held = "sample0" if held_kind == "capped" else "blank0"
    w.set_lid(lid_deg)
    holes = [(k + o) % 6 for o in NEIGHBOURS[config]] + [(k + 3) % 6]
    w.fill_holes(holes, held)
    w.base_qpos = w.data.qpos.copy()
    seat = w.seated_pose(k, w.truth.capped[held])
    ref = w.data.site(scene.slot_site(k)).xmat.reshape(3, 3)[:, 0]
    depth = depth_for(seat.capped)
    release = (HOLD_CAP if seat.capped else HOLD_RIM) + 0.004
    checks = [
        stand_in_and_let_go(w, held, seat, 0.06, depth, t, ref, (w.hole_body[k], w.rotor), release)
        for t in ROTOR_TURNS
    ]
    return {
        "hole": k,
        "neighbours": config,
        "held": held_kind,
        "lid_deg": lid_deg,
        "turns_ok": [round(math.degrees(t)) for t, c in zip(ROTOR_TURNS, checks, strict=False) if c.ok],
        "best": as_dict(max(checks, key=lambda c: c.gap)),
    }


# --- L. the lid, pinched at its front edge and swung shut -------------------------------------------------


def part_l(open_deg: float) -> dict:
    w = World(0)
    m, d = w.model, w.data
    lid = m.body("centrifuge/lid").id
    q, best, what = None, MARGIN, ""
    for ang in np.linspace(open_deg, 0.0, 15):
        w.set_lid(float(ang))
        w.base_qpos = d.qpos.copy()
        R = d.xmat[lid].reshape(3, 3)
        edge = d.site("centrifuge/lid").xpos.copy()
        z = -R[:, 2]  # fingers point from the front edge towards the hinge, in the lid's plane
        y = R[:, 1]  # and close across its thickness
        pos = edge + 0.012 * z  # pads 12 mm onto the lid
        rot = np.column_stack([np.cross(y, z), y, z])
        q2 = w.ik(pos, rot, q)
        if q2 is None:
            return {"open_deg": open_deg, "ok": False, "at_deg": round(float(ang), 1), "what": "out of reach"}
        if q is not None and np.abs(q2 - q).max() > 0.5:
            return {"open_deg": open_deg, "ok": False, "at_deg": round(float(ang), 1), "what": "flip"}
        q = q2
        g, wh = w.gap(q, 0.0272, gripper_may_touch=(lid,))
        if g < best:
            best, what = g, wh
    return {"open_deg": open_deg, "ok": best >= SAFE, "gap_mm": round(best * 1000, 1), "what": what}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--parts", default="abcdefg")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    results = {}
    with ProcessPoolExecutor(args.workers) as pool:
        if "a" in args.parts:
            results["a"] = [r for rows in pool.map(part_a, SEEDS) for r in rows]
        if "b" in args.parts:
            results["b"] = part_b()
        if "c" in args.parts:
            jobs = [(k, "blank0", d) for k in range(6) for d in (RIM_DEPTH,)]
            jobs += [(k, "sample0", d) for k in range(6) for d in (CAP_DEPTH, UNDER_CAP)]
            results["c"] = list(pool.map(part_c, jobs))
        if "d" in args.parts:
            jobs = [(y, h) for y in range(0, 360, 30) for h in (0.025, 0.035, 0.045)]
            results["d"] = [r for rows in pool.map(part_d, jobs) for r in rows]
        if "e" in args.parts:
            jobs = [(s, h) for s in SEEDS for h in (0.025, 0.035, 0.045)]
            results["e"] = list(pool.map(part_e, jobs))
        if "f" in args.parts:
            results["f"] = part_f()
        if "h" in args.parts:
            jobs = [
                (k, cfg, kind, 98.7) for k in range(6) for cfg in NEIGHBOURS for kind in ("capped", "open")
            ]
            results["h"] = list(pool.map(part_h, jobs))
        if "l" in args.parts:
            results["l"] = list(pool.map(part_l, [98.7, 90.0, 85.0, 80.0, 75.0, 70.0]))
        if "g" in args.parts:
            results["g"] = [r for rows in pool.map(part_g, SEEDS) for r in rows]
    for part, res in results.items():
        (OUT / f"part_{part}.json").write_text(json.dumps(res, indent=1, default=str))
    print("saved", list(results))


if __name__ == "__main__":
    main()
