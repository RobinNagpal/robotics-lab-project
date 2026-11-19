"""The lab balance: a base bolted to the bench, a round pan, and a tube holder standing on the pan.

The pan rides on a very stiff spring (a slide joint, 20 kN/m: a 50 g tube moves it 25 micrometres), as a
real balance's pan rides on its load cell, and a force sensor under it feels everything resting on it. The
pan never sleeps: MuJoCo stops computing the contacts of bodies at rest ("asleep") to save time, and a tube
asleep on a sleeping pan would weigh nothing. A tube touching the awake pan stays awake, and so do the balls
in it. The reading is in grams, zeroed with the empty holder on the pan (tared) and
rounded to the balance's 0.01 g. Unlike the simulator's own answers, **the arm may read it**: a real
balance shows it on its display.

A 50 ml tube has a cone for a bottom and cannot stand on a flat pan, so the pan carries a square sleeve the
tube stands in, tip in a small seat on the pan, the way people weigh conical tubes in a beaker or holder.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .racks import tip_seat
from .settings import GRAVITY, SCALE, SCALE_BODY, STEEL

SENSOR = "scale/load"
PAN = "scale/pan"
# The load cell's stiffness, N/m, and the gap between the base and the pan, m.
SPRING = 20000.0
PAN_GAP = 0.001


def add_scale(spec: mujoco.MjSpec, xy: tuple[float, float]) -> None:
    """The balance, fixed to the table at ``xy``."""
    bx, by, bz = SCALE.base
    base = spec.worldbody.add_body(name="scale", pos=[xy[0], xy[1], 0])
    base.add_geom(
        name="scale_base",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[bx, by, bz],
        pos=[0, 0, bz],
        rgba=list(SCALE_BODY.rgba),
        friction=[SCALE_BODY.friction, 0.005, 0.0001],
        mass=SCALE.mass,
    )
    # The display, on the side facing the arm (for people watching; the reading itself is ``read``).
    base.add_geom(
        name="scale_display",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.004, 0.045, 0.012],
        pos=[-bx - 0.002, 0, bz],
        rgba=[0.10, 0.12, 0.14, 1],
        contype=0,
        conaffinity=0,
    )
    top = 2 * bz + PAN_GAP
    pan = spec.worldbody.add_body(name=PAN, pos=[xy[0], xy[1], top])
    pan.add_joint(
        name=f"{PAN}_spring",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[0, 0, 1],
        stiffness=SPRING,
        damping=2 * (SPRING * 0.16) ** 0.5,
    )
    pan.sleep = mujoco.mjtSleepPolicy.mjSLEEP_NEVER
    pan.add_site(name=f"{PAN}_site", size=[0.003, 0, 0], group=4)
    spec.add_exclude(bodyname1="scale", bodyname2=PAN)
    t = SCALE.pan_thickness
    pan.add_geom(
        name="scale_pan",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[SCALE.pan_radius, t / 2, 0],
        pos=[0, 0, t / 2],
        rgba=list(STEEL.rgba),
        friction=[STEEL.friction, 0.005, 0.0001],
        mass=0.12,
    )
    # The holder: four walls round a square hole, and a thin foot under them.
    h, w, hole = SCALE.holder_height, SCALE.holder_wall, SCALE.holder_hole
    half = hole / 2 + w / 2
    for i, (x, y, sx, sy) in enumerate(
        [
            (half, 0, w / 2, half + w / 2),
            (-half, 0, w / 2, half + w / 2),
            (0, half, half + w / 2, w / 2),
            (0, -half, half + w / 2, w / 2),
        ]
    ):
        pan.add_geom(
            name=f"scale_holder{i}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[sx, sy, h / 2],
            pos=[x, y, t + h / 2],
            rgba=[0.85, 0.87, 0.90, 1],
            friction=[0.5, 0.005, 0.0001],
            mass=0.01,
        )
    for i, (half, pos) in enumerate(tip_seat(0.0, 0.0, t)):
        pan.add_geom(
            name=f"scale_seat{i}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=half,
            pos=pos,
            rgba=list(STEEL.rgba),
            mass=0.002,
        )
    spec.add_sensor(
        name=SENSOR,
        type=mujoco.mjtSensor.mjSENS_FORCE,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname=f"{PAN}_site",
    )


def pan_top() -> float:
    """Height of the pan's surface above the table: where a tube's tip rests."""
    return 2 * SCALE.base[2] + PAN_GAP + SCALE.pan_thickness


def raw(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """What the load cell feels, in kg, including the pan and holder."""
    adr = model.sensor(SENSOR).adr[0]
    # The sensor gives the force the pan's spring and joint pass to it, in the pan's frame (z up).
    return float(data.sensordata[adr + 2]) / GRAVITY


class Balance:
    """The balance's display: tare with the empty holder, then read in grams to 0.01 g."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model, self.data = model, data
        self.zero = 0.0

    def tare(self) -> None:
        """Zero the reading (with nothing in the holder)."""
        mujoco.mj_forward(self.model, self.data)
        self.zero = raw(self.model, self.data)

    def grams(self) -> float:
        """The reading now, rounded to 0.01 g. Only steady once whatever was put down has stopped moving."""
        kg = raw(self.model, self.data) - self.zero
        return float(np.round(kg / SCALE.readability) * SCALE.readability * 1000)
