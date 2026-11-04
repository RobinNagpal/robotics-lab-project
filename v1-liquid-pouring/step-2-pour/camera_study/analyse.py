"""Which camera can see the receiver's level while the source pours?

For every recorded Step 1b run (record.py) that reached the pouring point:
  * the pose at the end of Step 1b, and the source tipped to 45, 78, 85, 90 and 95 degrees with its lip
    15 mm from the receiver's axis and 20 mm above its rim (the approaches doc's pour position), set by
    inverse kinematics on Arm B (no physics: the balls are moved out of the way);
  * rays from each candidate camera to points inside the receiver, from 5 to 65 % of its height:
    a point counts as seen if nothing but the receiver's own glass is in between.
Also where both arms went during the whole run (2 cm boxes), so a fixed camera that the arms would hit
is marked.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
from lab_scene import liquid
from lab_scene import scene as lab

POUR = np.array([0.45, 0.0, 0.35])
CENTRE = np.array([0.45, 0.0, 0.28])  # about the 45 % mark of a medium or large receiver
TILTS = (78, 85, 90, 95)
LIP_OUT, LIP_UP = 0.015, 0.020
HEIGHTS = np.round(np.arange(0.05, 0.651, 0.05), 2)
BAND = (0.35, 0.60)  # where the level has to be read well: the run-up to 50 %
AZ = np.arange(0, 360, 15)
EL = (-15, 0, 15, 30, 45, 60, 75, 89)
DIST = (0.5, 0.8, 1.2)
VOXEL = 0.02
GROUPS = np.array([1, 0, 1, 0, 0, 0], np.uint8)  # the table, and everything's visual shape


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def ik6(m, d, arm, site, R_goal, p_goal, iters=400):
    """Arm joints that put ``site`` at p_goal with orientation R_goal (damped least squares)."""
    joints = lab.arm_joint_ids(m, arm)
    dofs = [m.joint(f"{arm}/{n}").dofadr[0] for n in lab.ARM_JOINTS]
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    sid = m.site(site).id
    for _ in range(iters):
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        R = d.site_xmat[sid].reshape(3, 3)
        dR = R_goal @ R.T
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, dR.flatten())
        rv = np.zeros(3)
        mujoco.mju_quat2Vel(rv, q, 1.0)
        err = np.concatenate([p_goal - d.site_xpos[sid], rv])
        if np.linalg.norm(err[:3]) < 1e-5 and np.linalg.norm(err[3:]) < 1e-4:
            break
        mujoco.mj_jacSite(m, d, jacp, jacr, sid)
        J = np.vstack([jacp, jacr])[:, dofs]
        step = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
        d.qpos[joints] += step * min(1.0, 0.2 / np.abs(step).max())
    mujoco.mj_kinematics(m, d)
    R = d.site_xmat[sid].reshape(3, 3)
    perr = np.linalg.norm(p_goal - d.site_xpos[sid])
    aerr = math.degrees(math.acos(np.clip((np.trace(R_goal @ R.T) - 1) / 2, -1, 1)))
    lim = np.array([m.jnt_range[m.joint(f"{arm}/{n}").id] for n in lab.ARM_JOINTS])
    q = d.qpos[joints]
    ok = perr < 1e-3 and aerr < 0.5 and np.all(q >= lim[:, 0]) and np.all(q <= lim[:, 1])
    return ok, perr, aerr


def free_qadr(m, body):
    return m.jnt_qposadr[m.body(body).jntadr[0]]


def set_body_pose(m, d, body, p, R):
    a = free_qadr(m, body)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    d.qpos[a : a + 3] = p
    d.qpos[a + 3 : a + 7] = q


def tube_names(m, d, containers):
    rec = min(
        (n for n in containers if n.startswith("receiver")),
        key=lambda n: np.linalg.norm(
            d.body(n).xpos + d.body(n).xmat.reshape(3, 3)[:, 2] * containers[n].height - POUR
        ),
    )
    src = min((n for n in containers if n.startswith("source")), key=lambda n: abs(d.body(n).xpos[1]))
    return rec, src


def pour_poses(m, d0, containers, rec, src):
    """The end-of-Step-1b state, then the source tipped. Returns [(label, data, note)]."""
    out = [("in place", d0, "")]
    h_s = containers[src].height
    r_s = containers[src].radius
    R0 = d0.body(src).xmat.reshape(3, 3).copy()
    p0 = d0.body(src).xpos.copy()
    site = lab.pinch_site("arm1")
    Ts = (d0.site(site).xpos.copy(), d0.site(site).xmat.reshape(3, 3).copy())
    # site pose in the tube's frame
    site_in_tube = (R0.T @ (Ts[0] - p0), R0.T @ Ts[1])
    rec_rim = d0.body(rec).xpos + d0.body(rec).xmat.reshape(3, 3)[:, 2] * containers[rec].height
    rec_axis_xy = d0.body(rec).xpos[:2]
    target = np.array([rec_axis_xy[0] + LIP_OUT, rec_axis_xy[1], rec_rim[2] + LIP_UP])
    d = mujoco.MjData(m)
    d.qpos[:] = d0.qpos
    phis = np.linspace(0, 2 * math.pi, 144, endpoint=False)
    rim_local = np.stack([r_s * np.cos(phis), r_s * np.sin(phis), np.full_like(phis, h_s)], 1)
    for t in TILTS:
        Rd = rot_y(-math.radians(t)) @ R0
        lip = rim_local[np.argmin((rim_local @ Rd.T)[:, 2])]
        pd = target - Rd @ lip
        R_site = Rd @ site_in_tube[1]
        p_site = pd + Rd @ site_in_tube[0]
        ok, pe, ae = ik6(m, d, "arm1", site, R_site, p_site)
        set_body_pose(m, d, src, pd, Rd)
        dd = mujoco.MjData(m)
        dd.qpos[:] = d.qpos
        mujoco.mj_forward(m, dd)
        out.append((f"{t} deg", dd, "" if ok else f"IK miss {pe * 1000:.1f} mm {ae:.1f} deg"))
    return out


def hits(m, d, cam, pts, exclude):
    n = len(pts)
    vec = pts - cam
    L = np.linalg.norm(vec, axis=1)
    vec = (vec / L[:, None]).flatten()
    gid = np.zeros(n, np.int32)
    dist = np.zeros(n)
    mujoco.mj_multiRay(m, d, cam, vec, GROUPS, 1, exclude, gid, dist, None, n, 10.0)
    seen = (dist < 0) | (dist >= L - 1e-4)
    return seen, gid


def target_points(d, rec, cont, cam):
    R = d.body(rec).xmat.reshape(3, 3)
    base = d.body(rec).xpos
    axis = R[:, 2]
    side = np.cross(cam - CENTRE, axis)
    side /= np.linalg.norm(side) + 1e-12
    r = 0.6 * (cont.radius - 0.002)
    pts, hs = [], []
    for h in HEIGHTS:
        c = base + axis * h * cont.height
        for o in (-r, 0.0, r):
            pts.append(c + side * o)
            hs.append(h)
    return np.array(pts), np.array(hs)


def blocker(m, g):
    if g < 0:
        return "?"
    b = m.body(m.geom_bodyid[g]).name
    if b.startswith("arm1/gripper"):
        return "Arm B gripper"
    if b.startswith("arm2/gripper"):
        return "Arm A gripper"
    if b.startswith("arm1"):
        return "Arm B links"
    if b.startswith("arm2"):
        return "Arm A links"
    if b.startswith("source_tube"):
        return "source tube"
    if "rack" in b or b.startswith("receiver"):
        return "racks/other tubes"
    return b or "table"


def candidates():
    out = []
    for D in DIST:
        for el in EL:
            for az in AZ:
                e, a = math.radians(el), math.radians(az)
                p = CENTRE + D * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
                out.append((D, el, int(az), p))
    return out


def arm_points(m, rng):
    """Sample points of every arm geom (visual meshes) and of the six tubes, in each geom's frame."""
    geoms, pts = [], []
    for g in range(m.ngeom):
        b = m.body(m.geom_bodyid[g]).name
        if m.geom_group[g] != 2 or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if not (b.startswith("arm") or b.startswith("source_tube") or b.startswith("receiver")):
            continue
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        if len(v) > 400:
            v = v[rng.choice(len(v), 400, replace=False)]
        geoms.append(g)
        pts.append(v.astype(np.float64))
    return geoms, pts


def voxels_of(m, d, geoms, pts):
    allp = [d.geom_xpos[g] + pts[i] @ d.geom_xmat[g].reshape(3, 3).T for i, g in enumerate(geoms)]
    return np.unique(np.floor(np.concatenate(allp) / VOXEL).astype(np.int32), axis=0)


def wrist_view(m, d, cam, pts, exclude):
    cid = m.camera(cam).id
    pos = d.cam_xpos[cid]
    R = d.cam_xmat[cid].reshape(3, 3)
    local = (pts - pos) @ R
    z = -local[:, 2]
    ty = math.tan(math.radians(m.cam_fovy[cid]) / 2)
    tx = ty * 640 / 480
    inside = (z > 0.02) & (np.abs(local[:, 0]) <= tx * z) & (np.abs(local[:, 1]) <= ty * z)
    seen, gid = hits(m, d, pos, pts, exclude)
    dist = float(np.linalg.norm(pts.mean(0) - pos))
    mm_px = 2 * dist * ty / 480 * 1000
    return inside & seen, inside, gid, dist, mm_px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    cands = candidates()
    band_vis = {i: [] for i in range(len(cands))}
    all_vis = {i: [] for i in range(len(cands))}
    block = {i: Counter() for i in range(len(cands))}
    named = {
        k: {"band": [], "all": [], "in_view": [], "dist": [], "mm_px": [], "block": Counter()}
        for k in ("front", "arm1/wrist", "arm2/wrist")
    }
    vox = set()
    poses_log = []
    files = sorted(a.runs.glob("*.npz"))
    for f in files:
        meta = json.loads(f.with_suffix(".json").read_text())
        seed = meta["seed"]
        m, truth = lab.build(seed)
        z = np.load(f)
        d = mujoco.MjData(m)
        geoms, pts = arm_points(m, rng)
        idx = z["idx"]
        # where the arms went
        mujoco.mj_resetData(m, d)
        for fr in z["frames"][::2]:
            d.qpos[idx] = fr
            mujoco.mj_kinematics(m, d)
            for v in map(tuple, voxels_of(m, d, geoms, pts)):
                vox.add(v)
        if not meta["done"]:
            poses_log.append({"run": f.stem, "done": False, "failure": meta["failure"]})
            continue
        d0 = mujoco.MjData(m)
        d0.qpos[:] = z["qpos"]
        balls = liquid.ball_body_ids(m)
        for b in balls:
            a_ = free_qadr(m, m.body(b).name)
            d0.qpos[a_ : a_ + 3] = [5.0 + 0.01 * b, 5.0, -5.0]
        mujoco.mj_forward(m, d0)
        rec, src = tube_names(m, d0, truth.containers)
        cont = truth.containers[rec]
        rec_id = m.body(rec).id
        for label, dp, note in pour_poses(m, d0, truth.containers, rec, src):
            for v in map(tuple, voxels_of(m, dp, geoms, pts)):
                vox.add(v)
            # contacts of Arm B or the source with Arm A or the receiver, as a check on the set pose
            clash = set()
            for c in dp.contact[: dp.ncon]:
                n1 = m.body(m.geom_bodyid[c.geom1]).name
                n2 = m.body(m.geom_bodyid[c.geom2]).name
                for x, y in ((n1, n2), (n2, n1)):
                    if (
                        (x.startswith("arm1") or x == src)
                        and (y.startswith("arm2") or y == rec)
                        and c.dist < -0.001
                    ):
                        clash.add((x.split("/")[-1], y.split("/")[-1]))
            poses_log.append(
                {"run": f.stem, "size": cont.height, "pose": label, "note": note, "clash": sorted(clash)[:4]}
            )
            for i, (*_, cam) in enumerate(cands):
                p, hs = target_points(dp, rec, cont, cam)
                seen, gid = hits(m, dp, cam, p, rec_id)
                band = (hs >= BAND[0]) & (hs <= BAND[1])
                band_vis[i].append(float(seen[band].mean()))
                all_vis[i].append(float(seen.mean()))
                for g in gid[~seen]:
                    block[i][blocker(m, g)] += 1
            for cam in named:
                p, hs = target_points(dp, rec, cont, dp.cam_xpos[m.camera(cam).id])
                band = (hs >= BAND[0]) & (hs <= BAND[1])
                ok, inside, gid, dist, mmpx = wrist_view(m, dp, cam, p, rec_id)
                s = named[cam]
                s["band"].append(float(ok[band].mean()))
                s["all"].append(float(ok.mean()))
                s["in_view"].append(float(inside.mean()))
                s["dist"].append(dist)
                s["mm_px"].append(mmpx)
                for g in gid[inside & ~ok]:
                    s["block"][blocker(m, g)] += 1
        print(f.stem, "done", flush=True)
    vox = np.array(sorted(vox), np.float64) * VOXEL + VOXEL / 2
    res = []
    for i, (D, el, az, cam) in enumerate(cands):
        clear = float(np.min(np.linalg.norm(vox - cam, axis=1)) - VOXEL * 0.87)
        res.append(
            {
                "D": D,
                "el": el,
                "az": az,
                "pos": np.round(cam, 3).tolist(),
                "band_min": round(min(band_vis[i]), 3),
                "band_mean": round(float(np.mean(band_vis[i])), 3),
                "all_min": round(min(all_vis[i]), 3),
                "all_mean": round(float(np.mean(all_vis[i])), 3),
                "clearance_m": round(clear, 3),
                "below_table": bool(cam[2] < 0.02),
                "blockers": dict(block[i].most_common(4)),
            }
        )
    nm = {
        k: {
            "band_min": round(min(v["band"]), 3),
            "band_mean": round(float(np.mean(v["band"])), 3),
            "all_mean": round(float(np.mean(v["all"])), 3),
            "in_view_mean": round(float(np.mean(v["in_view"])), 3),
            "dist_m": round(float(np.median(v["dist"])), 3),
            "mm_per_px": round(float(np.median(v["mm_px"])), 2),
            "blockers": dict(v["block"].most_common(4)),
        }
        for k, v in named.items()
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "candidates.json").write_text(json.dumps(res, indent=1))
    (a.out / "named.json").write_text(json.dumps(nm, indent=1))
    (a.out / "poses.json").write_text(json.dumps(poses_log, indent=1))
    np.save(a.out / "voxels.npy", vox)
    print(json.dumps(nm, indent=1))


if __name__ == "__main__":
    main()
