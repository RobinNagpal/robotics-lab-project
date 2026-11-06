"""AutoBio's four screw-cap lab tubes with their matching caps, unscrewed on our plain-joint thread.

A look at an option for later; the capped lab scene does not use these tubes yet. AutoBio's own thread is
in a Linux-only binary (libmjlab.so), so only its meshes are used here: each cap gets a turn joint and a
rise joint tied by the pitch, as in caps.py, and is turned 1.5 turns by a motor. Two pictures are saved:
caps on, and caps unscrewed. `make assets` downloads the meshes first.

python -m capping.autobio_tubes --root assets/autobio/autobio/assets/container --out figures
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import mujoco
from PIL import Image

# AutoBio gives every tube the same 3 mm pitch: a stand-in, not each tube's real thread.
PITCH = 0.003
TURNS = 1.5
CAP_MASS = 0.004

# name, tube mesh, cap mesh, height of the cap's top above the tube's base (from AutoBio's MJCF), x in the row
TUBES = [
    (
        "tube_1_5ml",
        "centrifuge_1-5ml_screw_vis/centrifuge_1-5ml_screw_body.STL",
        "centrifuge_1-5ml_screw_vis/centrifuge_1-5ml_screw_cap.STL",
        0.04455,
        -0.075,
    ),
    (
        "tube_15ml",
        "centrifuge_15ml_screw_vis/centrifuge_tube_15ml_body.STL",
        "centrifuge_15ml_screw_vis/centrifuge_tube_15ml_cap.STL",
        0.1226,
        -0.025,
    ),
    (
        "tube_50ml",
        "centrifuge_50ml_screw_vis/centrifuge_tube_50ml_body.STL",
        "centrifuge_50ml_screw_vis/centrifuge_tube_50ml_cap.STL",
        0.11965,
        0.035,
    ),
    (
        "cryovial_1_8ml",
        "cryovial_1-8ml_vis/cryovial_1-8ml_body.STL",
        "cryovial_1-8ml_vis/cryovial_1-8ml_cap.STL",
        0.03644,
        0.09,
    ),
]


def build(root: Path) -> mujoco.MjModel:
    """The four tubes in a row, each cap on a thread, driven by a velocity motor on its turn joint."""
    assets, bodies, ties, motors = [], [], [], []
    for name, tube_mesh, cap_mesh, cap_top, x in TUBES:
        assets.append(
            f'<mesh name="{name}" file="{root / tube_mesh}" scale="0.001 0.001 0.001"/>'
            f'<mesh name="{name}_cap" file="{root / cap_mesh}" scale="0.001 0.001 0.001"/>'
        )
        # The cap mesh is upside down in its file (top at z = 0), so the cap body is flipped, as in AutoBio.
        # Both joint axes then point down the flipped body, i.e. up in the world: positive = anticlockwise
        # seen from above = coming off.
        bodies.append(f"""
        <body name="{name}" pos="{x} 0 0">
          <geom type="mesh" mesh="{name}" rgba="0.92 0.95 1 0.45" contype="0" conaffinity="0"/>
          <body name="{name}_cap" pos="0 0 {cap_top}" euler="{math.pi} 0 0">
            <joint name="{name}_turn" type="hinge" axis="0 0 -1" range="0 {2 * math.pi * TURNS}"
                   frictionloss="0.002" damping="1e-4" armature="1e-4"/>
            <joint name="{name}_rise" type="slide" axis="0 0 -1" range="0 {PITCH * TURNS}" armature="0.004"/>
            <geom type="mesh" mesh="{name}_cap" rgba="0.93 0.27 0.06 1" mass="{CAP_MASS}"
                  contype="0" conaffinity="0"/>
          </body>
        </body>""")
        ties.append(
            f'<joint joint1="{name}_rise" joint2="{name}_turn" polycoef="0 {PITCH / (2 * math.pi)} 0 0 0" '
            f'solref="0.002 1"/>'
        )
        motors.append(f'<velocity joint="{name}_turn" kv="0.01" ctrlrange="-10 10"/>')

    xml = f"""
    <mujoco model="autobio tubes with caps">
      <compiler angle="radian" autolimits="true"/>
      <option timestep="0.001" gravity="0 0 0" integrator="implicitfast"/>
      <visual><global offwidth="1600" offheight="900"/></visual>
      <asset>{"".join(assets)}</asset>
      <worldbody>
        <light pos="0 -0.4 0.6" dir="0 0.5 -1" diffuse="0.9 0.9 0.9"/>
        <geom type="plane" size="0.3 0.3 0.01" rgba="0.8 0.8 0.78 1"/>
        {"".join(bodies)}
      </worldbody>
      <equality>{"".join(ties)}</equality>
      <actuator>{"".join(motors)}</actuator>
    </mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def picture(renderer: mujoco.Renderer, data: mujoco.MjData, path: Path) -> None:
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (0.005, 0, 0.07)
    cam.distance, cam.azimuth, cam.elevation = 0.36, 90, -12
    renderer.update_scene(data, cam)
    Image.fromarray(renderer.render()).save(path)
    print(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("assets/autobio/autobio/assets/container"))
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()

    model = build(args.root.resolve())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    args.out.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(model, height=900, width=1600) as renderer:
        picture(renderer, data, args.out / "autobio_tubes_caps_on.png")
        data.ctrl[:] = 2 * math.pi  # one turn a second
        mujoco.mj_step(model, data, 2000)
        for name, *_ in TUBES:
            turns = data.joint(f"{name}_turn").qpos[0] / (2 * math.pi)
            rise = data.joint(f"{name}_rise").qpos[0] * 1000
            expected = turns * PITCH * 1000
            print(f"{name:15s} turned {turns:.2f} turns, risen {rise:.2f} mm (thread says {expected:.2f})")
        picture(renderer, data, args.out / "autobio_tubes_caps_off.png")


if __name__ == "__main__":
    main()
