"""Pictures of the receiver at the pouring point from chosen cameras, to check the ray results by eye.

python render.py RUN.npz OUT.png  [--cams "az,el,D;az,el,D"]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import analyse as A
import mujoco
import numpy as np
from lab_scene import liquid
from lab_scene import scene as lab
from PIL import Image, ImageDraw

FIELD = 0.30  # m of view, top to bottom, at the receiver: a lens chosen to frame a large tube


def free_cam(p, fovy_field=FIELD):
    f = A.CENTRE - p
    D = np.linalg.norm(f)
    f /= D
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:] = A.CENTRE
    c.distance = D
    c.azimuth = math.degrees(math.atan2(f[1], f[0]))
    c.elevation = math.degrees(math.asin(f[2]))
    fovy = math.degrees(2 * math.atan(fovy_field / 2 / D))
    return c, fovy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--cams", default="240,0,1.2;270,15,1.2;90,15,1.2;0,15,1.2;180,15,1.2")
    ap.add_argument("--poses", default="in place,90 deg")
    ap.add_argument("--fill", type=float, default=0.45, help="receiver level shown in the tipped poses")
    ap.add_argument("--full", type=Path, default=None, help="also save each picture full size here")
    a = ap.parse_args()
    meta = json.loads(a.run.with_suffix(".json").read_text())
    m, truth = lab.build(meta["seed"])
    z = np.load(a.run)
    d0 = mujoco.MjData(m)
    d0.qpos[:] = z["qpos"]
    mujoco.mj_forward(m, d0)
    rec, src = A.tube_names(m, d0, truth.containers)
    poses = dict((lab_, dp) for lab_, dp, _ in A.pour_poses(m, d0, truth.containers, rec, src))
    # balls in the source are left behind by the set (not simulated) tilt: hide them
    ids = liquid.ball_body_ids(m)
    sb = d0.body(src)
    local = (d0.xpos[ids] - sb.xpos) @ sb.xmat.reshape(3, 3)
    src_balls = ids[
        (np.hypot(local[:, 0], local[:, 1]) < 0.02) & (local[:, 2] > -0.01) & (local[:, 2] < 0.21)
    ]
    W, H = 640, 480
    r = mujoco.Renderer(m, H, W)
    cams = [("front", None), ("arm2/wrist", None), ("arm1/wrist", None)]
    for s in a.cams.split(";"):
        az, el, D = map(float, s.split(","))
        e, b = math.radians(el), math.radians(az)
        p = A.CENTRE + D * np.array([math.cos(e) * math.cos(b), math.cos(e) * math.sin(b), math.sin(e)])
        cams.append((f"az {az:.0f} el {el:.0f} D {D}", p))
    rows = []
    for pname in a.poses.split(","):
        dp = poses[pname]
        if pname != "in place":
            for bid in src_balls:
                adr = A.free_qadr(m, m.body(bid).name)
                dp.qpos[adr : adr + 3] = [5.0, 5.0, -5.0]
            cont = truth.containers[rec]
            c = liquid.settled_liquid(cont, a.fill, np.random.default_rng(0))[: len(src_balls)]
            rb = dp.body(rec)
            for bid, cc in zip(src_balls, c, strict=False):
                adr = A.free_qadr(m, m.body(bid).name)
                dp.qpos[adr : adr + 3] = rb.xpos + rb.xmat.reshape(3, 3) @ cc
            mujoco.mj_forward(m, dp)
        row = []
        for name, p in cams:
            if p is None:
                r.update_scene(dp, camera=name)
            else:
                c, fovy = free_cam(p)
                m.vis.global_.fovy = fovy
                r.update_scene(dp, camera=c)
            img = Image.fromarray(r.render())
            if a.full:
                a.full.mkdir(parents=True, exist_ok=True)
                img.save(a.full / f"{pname.replace(' ', '')}_{name.replace('/', '-').replace(' ', '_')}.png")
            ImageDraw.Draw(img).text((8, 8), f"{pname} | {name}", fill=(255, 255, 0))
            row.append(img)
        rows.append(row)
    flat = [img for row in rows for img in row]
    cols = 4
    sheet = Image.new("RGB", (W * cols // 2, H * math.ceil(len(flat) / cols) // 2))
    for k, img in enumerate(flat):
        sheet.paste(img.resize((W // 2, H // 2)), ((k % cols) * W // 2, (k // cols) * H // 2))
    sheet.save(a.out)


if __name__ == "__main__":
    main()
