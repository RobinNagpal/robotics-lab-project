"""A person's arm and hand reaching into the robot's cell, made of simple shapes.

The whole arm is one mocap body ("person"): the simulation moves it exactly where we put it, and nothing
pushes it. Its shapes don't collide with anything (contype and conaffinity 0), so adding it changes nothing in
v3's physics. Touching the robot is measured instead, as the smallest distance between the person's shapes and
the robot's (``zones.py``).

The body's frame sits at the wrist. Its +x points along the hand towards the fingertips, +z is the back of the
hand, so the palm faces -z (down, when the hand is level). The forearm runs back along -x, and the upper arm
carries on behind it, rising towards the shoulder.

Each episode draws a ``Look`` (size and colours) and a ``Reach`` (where, when and how fast). The look is set
on a compiled model with ``set_look``, so one scene can show many different people without being rebuilt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

BODY = "person"
PREFIX = "person/"
# Where the person waits between reaches: far away and far below the floor (0.9 m below the bench), out of
# every camera's view whichever way the arm is turned. (At 3 m away and 1 m down, the upper arm poked up
# through the floor at the safety camera's edge, and the detector saw it.)
PARKED = np.array([50.0, 0.0, -10.0])

# The shapes at size 1 (an average adult hand), in the body's frame, metres.
# name: (type, size, pos, quat, part). Part is "skin" (bare or gloved hand), "sleeve" (forearm, upper arm).


def _along(direction: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """The turn that points a capsule's own axis (z) along ``direction``."""
    d = np.asarray(direction, dtype=float)
    d /= np.linalg.norm(d)
    q = np.zeros(4)
    mujoco.mju_quatZ2Vec(q, d)
    return tuple(float(c) for c in q)


_X = _along((1.0, 0.0, 0.0))
_UPPER_TILT = 0.45  # rad: the upper arm rises behind the forearm, towards the shoulder
_UPPER_DIR = np.array([-math.cos(_UPPER_TILT), 0.0, math.sin(_UPPER_TILT)])
_UPPER_HALF = 0.15
_UPPER_CENTRE = np.array([-0.27, 0.0, 0.0]) + _UPPER_HALF * _UPPER_DIR
_THUMB_DIR = (math.cos(0.6), math.sin(0.6), -0.25)

SHAPES: dict[str, tuple[int, tuple[float, ...], tuple[float, ...], tuple[float, ...], str]] = {
    "palm": (mujoco.mjtGeom.mjGEOM_BOX, (0.048, 0.042, 0.014), (0.050, 0.0, 0.0), (1, 0, 0, 0), "skin"),
    "finger0": (mujoco.mjtGeom.mjGEOM_CAPSULE, (0.0085, 0.030), (0.128, 0.029, -0.002), _X, "skin"),
    "finger1": (mujoco.mjtGeom.mjGEOM_CAPSULE, (0.0090, 0.034), (0.134, 0.0095, -0.002), _X, "skin"),
    "finger2": (mujoco.mjtGeom.mjGEOM_CAPSULE, (0.0088, 0.032), (0.131, -0.0095, -0.002), _X, "skin"),
    "finger3": (mujoco.mjtGeom.mjGEOM_CAPSULE, (0.0078, 0.025), (0.120, -0.029, -0.002), _X, "skin"),
    # The thumb sits on the palm's +y side, pointing forwards and out.
    "thumb": (
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        (0.0100, 0.024),
        (0.040, 0.058, -0.008),
        _along(_THUMB_DIR),
        "skin",
    ),
    "wrist": (mujoco.mjtGeom.mjGEOM_CAPSULE, (0.026, 0.012), (-0.010, 0.0, 0.0), _X, "skin"),
    "forearm": (mujoco.mjtGeom.mjGEOM_CAPSULE, (0.036, 0.115), (-0.150, 0.0, 0.0), _X, "sleeve"),
    "upper_arm": (
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        (0.045, _UPPER_HALF),
        tuple(float(c) for c in _UPPER_CENTRE),
        _along(tuple(_UPPER_DIR)),
        "sleeve",
    ),
}
SKIN_SHAPES = tuple(n for n, s in SHAPES.items() if s[4] == "skin")
SLEEVE_SHAPES = tuple(n for n, s in SHAPES.items() if s[4] == "sleeve")
# The fingertips, at size 1, in the body's frame: the deepest points of a reach.
FINGERTIPS = np.array([[0.128 + 0.030 + 0.0085, 0.029, -0.002], [0.134 + 0.034 + 0.009, 0.0095, -0.002]])

# Colours. Labs mostly use blue or purple nitrile gloves and white coats; the rest are there for variety.
SKIN = {
    "skin_light": (0.93, 0.78, 0.67),
    "skin_medium": (0.78, 0.57, 0.42),
    "skin_tan": (0.63, 0.44, 0.31),
    "skin_dark": (0.38, 0.25, 0.18),
}
GLOVES = {
    "glove_blue": (0.22, 0.45, 0.85),
    "glove_purple": (0.50, 0.33, 0.70),
    "glove_white": (0.93, 0.93, 0.90),
    "glove_black": (0.10, 0.10, 0.11),
    "glove_green": (0.30, 0.65, 0.40),
}
SLEEVES = {
    "coat_white": (0.94, 0.94, 0.94),
    "sleeve_blue": (0.20, 0.30, 0.55),
    "sleeve_grey": (0.45, 0.46, 0.48),
    "sleeve_black": (0.12, 0.12, 0.13),
}
HAND_COLOURS = {**SKIN, **GLOVES}


def add_person(spec: mujoco.MjSpec) -> None:
    """Add the person (parked out of sight) to a scene's spec. Call before compiling."""
    body = spec.worldbody.add_body(name=BODY, pos=PARKED.tolist(), mocap=True)
    for name, (kind, size, pos, quat, part) in SHAPES.items():
        colour = SKIN["skin_medium"] if part == "skin" else SLEEVES["coat_white"]
        body.add_geom(
            name=PREFIX + name,
            type=kind,
            size=[*size, 0.0][:3],
            pos=list(pos),
            quat=list(quat),
            rgba=[*colour, 1.0],
            contype=0,
            conaffinity=0,
            group=1,
        )


@dataclass(frozen=True)
class Look:
    """How the person looks in one episode."""

    size: float  # 1 is an average adult hand; 0.85 to 1.15 covers small to large
    hand: str  # a key of HAND_COLOURS
    sleeve: str  # a key of SLEEVES

    @staticmethod
    def draw(rng: np.random.Generator, dark_gloves: bool = False) -> Look:
        """A random look. ``dark_gloves``: mostly gloves, and mostly the dark ones (black, blue, green,
        purple), which v1 found hardest against the robot's dark links (docs/detector-results.md).
        """
        if dark_gloves:
            gloved = rng.random() < 0.8
            weights = {"glove_black": 0.35, "glove_blue": 0.25, "glove_green": 0.15, "glove_purple": 0.15}
            weights["glove_white"] = 1.0 - sum(weights.values())
            names = list(GLOVES)
            hand = (
                str(rng.choice(names, p=[weights[n] for n in names]))
                if gloved
                else str(rng.choice(list(SKIN)))
            )
            sleeve = str(rng.choice(list(SLEEVES), p=[0.55, 0.15, 0.15, 0.15]))
            return Look(size=float(rng.uniform(0.85, 1.15)), hand=hand, sleeve=sleeve)
        gloved = rng.random() < 0.6
        hand = str(rng.choice(list(GLOVES if gloved else SKIN)))
        sleeve = str(rng.choice(list(SLEEVES), p=[0.55, 0.15, 0.15, 0.15]))
        return Look(size=float(rng.uniform(0.85, 1.15)), hand=hand, sleeve=sleeve)


def _ids(model: mujoco.MjModel) -> dict[str, int]:
    return {name: model.geom(PREFIX + name).id for name in SHAPES}


def set_look(model: mujoco.MjModel, look: Look) -> None:
    """Size and colour the person on a compiled model. The size scales every shape and its place together."""
    for name, gid in _ids(model).items():
        _, size, pos, _, part = SHAPES[name]
        model.geom_size[gid, : len(size)] = np.array(size) * look.size
        model.geom_pos[gid] = np.array(pos) * look.size
        model.geom_rbound[gid] = model.geom_size[gid, 0] + (
            model.geom_size[gid, 1] if len(size) == 2 else 0.0
        )
        if name == "palm":
            model.geom_rbound[gid] = float(np.linalg.norm(model.geom_size[gid]))
        colour = HAND_COLOURS[look.hand] if part == "skin" else SLEEVES[look.sleeve]
        model.geom_rgba[gid] = [*colour, 1.0]


@dataclass(frozen=True)
class Reach:
    """One reach: the wrist goes from ``start`` to ``target``, pauses, and goes back.

    Times in seconds from the start of the episode, positions in metres (world frame), speeds in m/s.
    """

    t0: float  # when the hand starts moving in
    start: tuple[float, float, float]
    target: tuple[float, float, float]
    speed_in: float
    pause: float
    speed_out: float
    pitch: float  # rad, fingertips down (+) or up (-)
    roll: float  # rad, about the hand's own axis

    @property
    def t_arrive(self) -> float:
        return self.t0 + self._length / self.speed_in

    @property
    def t_leave(self) -> float:
        return self.t_arrive + self.pause

    @property
    def t_end(self) -> float:
        return self.t_leave + self._length / self.speed_out

    @property
    def _length(self) -> float:
        return float(np.linalg.norm(np.subtract(self.target, self.start)))

    def wrist(self, t: float) -> np.ndarray | None:
        """Where the wrist is at time ``t``; None before the reach starts and after it ends (parked)."""
        a, b = np.array(self.start), np.array(self.target)
        if t < self.t0 or t > self.t_end:
            return None
        if t < self.t_arrive:
            s = _ease((t - self.t0) / (self.t_arrive - self.t0))
            return a + (b - a) * s
        if t < self.t_leave:
            return b
        s = _ease((t - self.t_leave) / (self.t_end - self.t_leave))
        return b + (a - b) * s

    def quat(self) -> np.ndarray:
        """The hand's turn: pointing from start to target (seen from above), then pitched and rolled."""
        d = np.subtract(self.target, self.start)
        yaw = math.atan2(d[1], d[0])
        return _yaw_pitch_roll(yaw, self.pitch, self.roll)


def _ease(s: float) -> float:
    """Smooth start and stop, like a real reach."""
    s = min(max(s, 0.0), 1.0)
    return s * s * (3 - 2 * s)


def _yaw_pitch_roll(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Turn about world z by ``yaw``, then the hand's own y by ``pitch``, then its own x by ``roll``."""
    qz = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    qy = np.array([math.cos(pitch / 2), 0, math.sin(pitch / 2), 0])
    qx = np.array([math.cos(roll / 2), math.sin(roll / 2), 0, 0])
    out = np.zeros(4)
    tmp = np.zeros(4)
    mujoco.mju_mulQuat(tmp, qz, qy)
    mujoco.mju_mulQuat(out, tmp, qx)
    return out


def place(model: mujoco.MjModel, data: mujoco.MjData, pos: np.ndarray | None, quat: np.ndarray) -> None:
    """Put the person's wrist at ``pos`` with turn ``quat``; ``pos`` None parks the person out of sight."""
    m = model.body(BODY).mocapid[0]
    data.mocap_pos[m] = PARKED if pos is None else pos
    data.mocap_quat[m] = quat


# --- drawing a random reach --------------------------------------------------------------------------------

SIDES = {"front": 0.5, "left": 0.25, "right": 0.25}  # left is +y, right is -y (seen from the arm)
# What the reach is for: to stay outside the zones (a person working at the bench's edge), to come into the
# warning zone only, or to reach into the stop zone. Radii are for the wrist; the fingers reach about 18 cm
# further.
GOALS = {"outside": (0.2, 1.40, 1.60), "warning": (0.3, 1.02, 1.28), "stop": (0.5, 0.35, 0.95)}
# A fourth goal, only when asked for (v2's extra pictures): the hand goes to where the robot is, beside or
# just under one of its links, so the robot partly covers it.
NEAR_ROBOT = "near_robot"
ROBOT_POINTS = ("arm/forearm_link", "arm/wrist_1_link", "arm/wrist_2_link", "arm/wrist_3_link")
# m/s. ISO 13855 takes 1.6 m/s for a hand reaching in, 2.0 m/s close to the hazard.
HAND_SPEED = (0.3, 2.0)
CLEARANCE = 0.04  # m: the lowest the hand goes above anything on the bench
HIGHEST = 0.5  # m: the wrist never goes higher (a reach that would have to is drawn again)
ROBOT_PREFIX = "arm/"


def surface(model: mujoco.MjModel, data: mujoco.MjData, x: float, y: float) -> float:
    """The height of whatever is highest on the bench at (x, y): the table, a rack, the centrifuge, its lid.

    The robot doesn't count (it moves; a person reaches past it, not over it): a ray down that hits the robot
    carries on below it.
    """
    geomid = np.zeros(1, dtype=np.int32)
    top = 2.0
    for _ in range(10):
        dist = mujoco.mj_ray(
            model,
            data,
            np.array([x, y, top]),
            np.array([0.0, 0.0, -1.0]),
            None,
            1,
            model.body(BODY).id,
            geomid,
        )
        if dist < 0:
            return 0.0
        hit = top - dist
        if not model.body(model.geom_bodyid[geomid[0]]).name.startswith(ROBOT_PREFIX):
            return hit
        top = hit - 1e-4
    return 0.0


def draw_reach(
    rng: np.random.Generator,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    t_from: float,
    t_to: float,
    goal: str | None = None,
) -> tuple[str, Reach]:
    """A random reach starting between ``t_from`` and ``t_to`` s. Returns (goal, reach).

    The hand starts at the bench's edge on a random side, out of the zones, and goes in a straight line to a
    target chosen for the goal. It stays above everything on the bench along the way (the person's shapes
    don't collide, so this is what keeps it from going through the centrifuge), and never higher than
    ``HIGHEST``: a reach that would have to go higher is drawn again.
    """
    for attempt in range(200):
        # A robot standing high up leaves no low spot beside it: then reach into the stop zone instead.
        g = "stop" if goal == NEAR_ROBOT and attempt >= 100 else goal
        drawn = _draw_once(rng, model, data, t_from, t_to, g)
        if drawn[1].start[2] <= HIGHEST:
            return drawn
    raise RuntimeError("no reach found below HIGHEST")


def _draw_once(
    rng: np.random.Generator,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    t_from: float,
    t_to: float,
    goal: str | None = None,
) -> tuple[str, Reach]:
    side = str(rng.choice(list(SIDES), p=list(SIDES.values())))
    if goal is None:
        names = list(GOALS)
        goal = str(rng.choice(names, p=[GOALS[n][0] for n in names]))
    _, r_lo, r_hi = GOALS["stop" if goal == NEAR_ROBOT else goal]
    # The start: past the bench's edge on that side, far enough out that the fingertips are clear of the
    # warning zone, and out of the safety camera's view.
    along = rng.uniform(-1.0, 1.0)
    edge = rng.uniform(1.45, 1.60)
    if side == "front":
        start = np.array([edge, along * 0.7, 0.0])
        heading = 0.0
    else:
        sign = 1.0 if side == "left" else -1.0
        start = np.array([0.35 + along * 0.35, sign * edge, 0.0])
        heading = sign * math.pi / 2
    # The target: at the goal's radius, in a direction near the person's own side, never behind the arm.
    # (Turning the direction, not moving the point, keeps the target at the goal's radius.)
    angle = float(np.clip(heading + rng.uniform(-0.6, 0.6), -math.pi / 2 - 0.03, math.pi / 2 + 0.03))
    r = rng.uniform(r_lo, r_hi)
    target = np.array([r * math.cos(angle), r * math.sin(angle), 0.0])
    start[2] = rng.uniform(0.25, 0.45)
    target[2] = rng.uniform(0.10, 0.35)
    if goal == NEAR_ROBOT:
        # Beside or under a link of the robot: the fingertips (about 18 cm ahead of the wrist) end near it.
        link = data.xpos[model.body(ROBOT_POINTS[rng.integers(len(ROBOT_POINTS))]).id]
        towards = link[:2] - start[:2]
        towards /= np.linalg.norm(towards)
        side_step = np.array([-towards[1], towards[0]]) * rng.uniform(-0.07, 0.07)
        target[:2] = link[:2] - towards * rng.uniform(0.0, 0.18) + side_step
        target[2] = link[2] + rng.uniform(-0.15, 0.03)
        target[0] = max(target[0], -0.05)  # never behind the arm's base
    # Keep the hand above things: look along the path, the fingers' reach beyond the target included.
    path = target - start
    reach_dir = path / np.linalg.norm(path)
    lowest = 0.0
    for s in np.linspace(0.3, 1.0, 15):
        for ahead in (0.0, 0.18):
            p = start + path * s + reach_dir * ahead
            lowest = max(lowest, surface(model, data, p[0], p[1]) + CLEARANCE + 0.03)
    target[2] = max(target[2], lowest)
    start[2] = max(start[2], target[2])
    speed = rng.uniform(*HAND_SPEED)
    reach = Reach(
        t0=float(rng.uniform(t_from, t_to)),
        start=tuple(float(c) for c in start),
        target=tuple(float(c) for c in target),
        speed_in=float(speed),
        pause=float(rng.uniform(0.3, 2.0)),
        speed_out=float(rng.uniform(0.6, 1.0) * speed),
        pitch=float(rng.uniform(0.0, 0.35)),
        roll=float(rng.uniform(-0.4, 0.4)),
    )
    return goal, reach
