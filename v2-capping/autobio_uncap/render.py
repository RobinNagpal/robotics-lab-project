"""Pictures of one run of the ported expert, from AutoBio's front camera and a close view, into
runs/autobio/ (at the hand-over, mid-unscrewing and the end).

    python -m autobio_uncap.render --seed 0 --variant end135
"""

from __future__ import annotations

import argparse

import mujoco
import numpy as np
from PIL import Image

from .evaluate import OUT, VARIANTS
from .expert import Expert


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--variant", default="end135", choices=list(VARIANTS))
    a = p.parse_args()
    e = Expert(VARIANTS[a.variant])
    r = mujoco.Renderer(e.model, 480, 640)
    close = mujoco.MjvCamera()
    close.distance, close.azimuth, close.elevation = 0.45, 150, -20
    opt = mujoco.MjvOption()
    shots = {"handover": None, "unscrewing": None}
    frames = []

    def snap(tag):
        close.lookat[:] = e.data.xpos[e.tube] + [0, 0, 0.08]
        row = []
        for cam in ("table_cam_front", close):
            r.update_scene(e.data, camera=cam, scene_option=opt)
            row.append(r.render())
        frames.append(np.concatenate(row, axis=1))
        shots[tag] = True

    def cb(e):
        n = len(e.rec.rounds)
        if shots["handover"] is None and e.rec.hold_squeeze_N:
            snap("handover")
        if shots["unscrewing"] is None and n == 2:
            snap("unscrewing")

    e.on_step = cb
    rec = e.run(a.seed)
    snap("end")
    out = OUT / f"render_{a.variant}_seed{a.seed}.png"
    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(frames, axis=0)).save(out)
    print(out, "their check:", rec.ok_their_check)


if __name__ == "__main__":
    main()
