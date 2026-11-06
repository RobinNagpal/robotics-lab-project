"""AutoBio's uncapping scene (``lab_screw_all.xml``), loaded on a Mac.

Everything is AutoBio's own file, unchanged: the table, the two ALOHA arms (ViperX 300s with the ALOHA
gripper), the 10-slot rack, the cameras, the light and the physics options (implicit-fast integrator,
``impratio`` 10, elliptic friction cone, 2 no-slip iterations, multi-point convex collision, 2 ms step).
Only the 50 ml screw-cap tube is taken out and put back by code, because its thread needs the Linux-only
plugin: the tube's own file is read with the two thread shapes removed and the thread made of joints
instead (``thread.py``). Its contact pairs are the scene's, plus the same four for the free cap.

What compiling their file shows (``python -m autobio_uncap.scene``):
- only the scene's contact pairs take effect; the tube file's own pairs (friction 0.1) are dropped when
  it is attached, so the thread's friction is the scene's 1;
- the pairs have ``condim`` 3 (a pair has its own, default 3); only the fingers' other contacts (with the
  tube's and cap's collision boxes) get the finger geoms' ``condim`` 6;
- the first keyframe, which AutoBio's reset uses, puts arm 1 in its neutral pose and leaves arm 2 at all
  zeros (the second arm's neutral pose is the second keyframe, never used).
"""

from __future__ import annotations

import argparse
import re
import time
import warnings
from pathlib import Path

import mujoco
import numpy as np

from . import fetch, thread

# Attaching the tube's model notes that it asks for impratio 1; the scene's 10 is kept, as in AutoBio.
warnings.filterwarnings("ignore", message="Attach conflict")

FINGERS = [f"{a}/aloha:{f}" for a in (1, 2) for f in ("leftfinger", "rightfinger")]
THREAD_MASS = 0.0036945  # kg: each thread shape's, from AutoBio's compiled model (tube 14.7 g, cap 20.8 g)
PAIR = dict(friction=[1, 1, 0.005, 0.0001, 0.0001], solref=[0.002, 1], solimp=[0.9, 0.95, 0.001, 0.5, 2])


def _scene_xml(model_dir: Path) -> str:
    """lab_screw_all.xml with absolute paths, and without the tube (bodies, model and its pairs)."""
    t = (model_dir / "scene" / "lab_screw_all.xml").read_text()
    t = re.sub(r"<!--.*?-->", "", t, flags=re.S)
    t = t.replace('file="../', f'file="{model_dir}/')
    t = re.sub(r'<model name="centrifuge_50ml_screw"[^>]*/>', "", t)
    t = re.sub(r'<body name="centrifuge_50ml_screw_(cap|body)_wrap".*?</body>', "", t, flags=re.S)
    t = re.sub(r'<pair geom1="[67]/[^>]*/>', "", t)
    return t


def _tube_xml(model_dir: Path) -> str:
    """centrifuge_50ml_screw.xml without its thread shapes (and its own pairs, dropped by AutoBio too)."""
    t = (model_dir / "object" / "centrifuge_50ml_screw.xml").read_text()
    t = t.replace('meshdir="../../assets"', f'meshdir="{model_dir.parent / "assets"}"')
    t = re.sub(r"<extension>.*?</extension>", "", t, flags=re.S)
    t = re.sub(r'<mesh name="(bolt|nut)">.*?</mesh>', "", t, flags=re.S)
    # The thread shapes go, but their mass stays (3.7 g each in AutoBio's compiled model): a cylinder
    # in their place that touches nothing.
    t = re.sub(
        r'<geom type="sdf" name="(\w+)" mesh="\w+" rgba="[^"]*" (pos="[^"]*").*?</geom>',
        rf'<geom type="cylinder" name="\1" size="0.014 0.0033" \2 mass="{THREAD_MASS}" '
        r'contype="0" conaffinity="0" group="5"/>',
        t,
        flags=re.S,
    )
    t = re.sub(r"<contact>.*?</contact>", "", t, flags=re.S)
    return t


def load_spec(turns: float = thread.TURNS) -> mujoco.MjSpec:
    model_dir = fetch.ensure() / "autobio" / "model"
    spec = mujoco.MjSpec.from_string(_scene_xml(model_dir))
    tube_xml = _tube_xml(model_dir)

    def tube_model() -> mujoco.MjSpec:
        return mujoco.MjSpec.from_string(tube_xml)

    # The tube, as the scene had it: a free body at (-0.072, -0.018, 0.83), the first rack hole.
    wrap = spec.worldbody.add_body(name="centrifuge_50ml_screw_body_wrap", pos=[-0.072, -0.018, 0.83])
    wrap.add_freejoint(name="centrifuge_50ml_screw_body")
    wrap.add_frame().attach_body(tube_model().body("centrifuge_50ml_screw_body"), "7/", "")
    thread.add(spec, tube_model, turns)
    for cap in ("6", "6f"):
        for f in FINGERS:
            spec.add_pair(geomname1=f"{cap}/cap_cyl", geomname2=f, **PAIR)
    for f in FINGERS:
        spec.add_pair(geomname1="7/body_cyl", geomname2=f, **PAIR)
    # AutoBio's ScrewLoose.prepare: both arms gravity-compensated (their weight carried, as by motors
    # that hold position without effort).
    for arm in (1, 2):
        _gravcomp(spec.body(f"{arm}/aloha:left/base_link"))
    # Keyframes are replaced by ``reset`` (the tube's joints are added after they were made).
    for key in list(spec.keys):
        spec.delete(key)
    return spec


def _gravcomp(body: mujoco.MjsBody) -> None:
    body.gravcomp = 1
    for child in body.bodies:
        _gravcomp(child)


NEUTRAL = np.array([0, -0.96, 1.16, 0, -0.3, 0])
NEUTRAL_GRIP = 0.0084


def reset(model: mujoco.MjModel, data: mujoco.MjData, th: thread.Thread) -> None:
    """AutoBio's ``manager.reset(keyframe=0)``: arm 1 neutral, arm 2 at zeros, the tube in its hole."""
    mujoco.mj_resetData(model, data)
    th.reset(data)
    a1 = model.joint("1/aloha:left/waist").qposadr[0]
    data.qpos[a1 : a1 + 6] = NEUTRAL
    for f in ("left_finger", "right_finger"):
        data.qpos[model.joint(f"1/aloha:left/{f}").qposadr[0]] = NEUTRAL_GRIP
    c1 = model.actuator("1/aloha:left/waist").id
    data.ctrl[c1 : c1 + 6] = NEUTRAL
    data.ctrl[model.actuator("1/aloha:left/gripper").id] = NEUTRAL_GRIP
    mujoco.mj_forward(model, data)
    mujoco.mj_rnePostConstraint(model, data)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.parse_args()
    spec = load_spec()
    m = spec.compile()
    d = mujoco.MjData(m)
    th = thread.Thread(m)
    reset(m, d, th)
    o = m.opt
    print(f"nq {m.nq}, nu {m.nu}, step {o.timestep} s, impratio {o.impratio}, noslip {o.noslip_iterations}")
    print("tube", m.body(thread.TUBE).mass, "kg; cap", m.body(thread.CAP).mass, "kg")
    t0 = time.perf_counter()
    for _ in range(1000):
        thread.step(m, d, th)
    z = d.xpos[m.body(thread.TUBE).id][2]
    print(f"1000 steps (2 s) in {time.perf_counter() - t0:.2f} s; tube z {z:.4f}, contacts {d.ncon}")


if __name__ == "__main__":
    main()
