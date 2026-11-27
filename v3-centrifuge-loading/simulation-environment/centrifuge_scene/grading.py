"""Grading a finished run: which tube is in which hole, and how far out of balance the rotor is.

For tests and the examiner only: it reads the simulator's answers (where each tube is, how many balls are
in it), which the arm must not.

The balance number. Spinning, each tube pulls outwards with a force proportional to its mass (all holes are
at the same radius). The rotor is balanced when those pulls cancel. Adding them as arrows, one per loaded
hole, each as long as its tube's mass and pointing out through its hole, the length of the sum is the
imbalance, in grams: 0 for a perfect load, the mass of one tube for a single tube on its own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from . import centrifuge, liquid
from .scene import Truth, slot_site
from .settings import BALANCE_TOLERANCE, LIQUID, ROTOR, TUBE
from .tubes import cap_on

# A tube counts as in a hole when its tip is this close to the hole's bottom and its axis this close to the
# hole's.
SEATED_DISTANCE = 0.006
SEATED_ANGLE_DEG = 3.0


def tube_mass(model: mujoco.MjModel, data: mujoco.MjData, truth: Truth, name: str) -> float:
    """The tube's mass now, kg: its own (with cap if it has one) and the balls inside it."""
    if name in truth.samples:
        own = truth.samples[name].body_mass
    elif name in truth.blanks:
        own = truth.blanks[name] + (TUBE.cap_mass if cap_on(model, name) else 0.0)
    else:
        own = TUBE.body_mass
    return own + len(liquid.balls_in(model, data, TUBE, name)) * LIQUID.ball_mass


def tubes_in_rotor(model: mujoco.MjModel, data: mujoco.MjData, truth: Truth) -> dict[int, str]:
    """{hole: tube} for every tube seated in a hole."""
    seated = {}
    for k in range(ROTOR.slots):
        site = data.site(slot_site(k))
        axis = site.xmat.reshape(3, 3)[:, 2]
        for name in truth.tubes:
            b = model.body(name).id
            close = np.linalg.norm(data.xpos[b] - site.xpos) < SEATED_DISTANCE
            aligned = data.xmat[b].reshape(3, 3)[:, 2] @ axis > math.cos(math.radians(SEATED_ANGLE_DEG))
            if close and aligned:
                seated[k] = name
    return seated


def imbalance(masses: dict[int, float]) -> float:
    """The length of the sum of the tubes' outward pulls, in the units of ``masses``, from {hole: mass}."""
    x = sum(m * math.cos(2 * math.pi * k / ROTOR.slots) for k, m in masses.items())
    y = sum(m * math.sin(2 * math.pi * k / ROTOR.slots) for k, m in masses.items())
    return math.hypot(x, y)


@dataclass
class Result:
    seated: dict[int, str]
    masses: dict[int, float]  # kg
    imbalance: float  # kg
    samples_loaded: bool  # every sample tube is in the rotor
    lid_latched: bool
    spilled: int

    @property
    def balanced(self) -> bool:
        return self.imbalance <= BALANCE_TOLERANCE

    @property
    def success(self) -> bool:
        return self.samples_loaded and self.balanced and self.lid_latched and self.spilled == 0

    def summary(self) -> str:
        mark = "balanced" if self.balanced else "NOT balanced"
        return (
            f"{mark}: {self.imbalance * 1000:.2f} g out (limit {BALANCE_TOLERANCE * 1000:.1f} g); "
            f"{len(self.seated)} tubes loaded; lid {'latched' if self.lid_latched else 'open'}; "
            f"{self.spilled} balls spilled"
        )


def grade(model: mujoco.MjModel, data: mujoco.MjData, truth: Truth) -> Result:
    seated = tubes_in_rotor(model, data, truth)
    masses = {k: tube_mass(model, data, truth, name) for k, name in seated.items()}
    return Result(
        seated=seated,
        masses=masses,
        imbalance=imbalance(masses),
        samples_loaded=set(truth.samples) <= set(seated.values()),
        lid_latched=bool(data.eq_active[centrifuge.latch_id(model)]),
        spilled=liquid.spilled(model, data, truth.tubes),
    )
