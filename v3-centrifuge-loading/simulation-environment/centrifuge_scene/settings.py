"""Every number that describes the centrifuge scene, in one place.

The arm's base is at the origin, bolted to the table, and the table top is the plane z = 0. x points
away from the arm, towards the middle of the work area; y to its left. Lengths are in metres, masses in
kilograms, angles in radians unless a name says otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
MENAGERIE = ASSETS / "menagerie"
ARM_XML = MENAGERIE / "universal_robots_ur5e" / "ur5e.xml"
GRIPPER_XML = MENAGERIE / "robotiq_2f85" / "2f85.xml"
AUTOBIO = ASSETS / "autobio" / "autobio"
CENTRIFUGE_XML = AUTOBIO / "model" / "instrument" / "centrifuge_eppendorf_5430.xml"
TUBE_MESHES = AUTOBIO / "assets" / "container" / "centrifuge_50ml_screw_vis"
# Meshes made here (the rotor with its holes), cached.
CONVERTED = ASSETS / "converted"

# 1 ms, as in v1: a ball falling 15 cm moves about 1.7 mm per step, less than half its radius, so it
# cannot pass through a 2 mm tube wall in one step.
TIMESTEP = 0.001
GRAVITY = 9.81

ARM_NAME = "arm"
# The arm starts here: upright, gripper pointing straight down, out of the overhead camera's way.
HOME = (-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0)


@dataclass(frozen=True)
class Material:
    """How a surface behaves in contact. Friction is MuJoCo's sliding coefficient."""

    friction: float
    rgba: tuple[float, float, float, float]


# Polypropylene tubes, nearly clear; their orange screw caps; the racks; the bench; the rotor.
TUBE_PLASTIC = Material(friction=0.5, rgba=(0.92, 0.95, 1.0, 0.40))
CAP_PLASTIC = Material(friction=0.6, rgba=(0.93, 0.27, 0.06, 1.0))
RACK_PLASTIC = Material(friction=0.6, rgba=(0.20, 0.45, 0.80, 1.0))
# A grey-blue bench, darker than v1's, so the white centrifuge and balance and the clear tubes stand out.
TABLE = Material(friction=0.7, rgba=(0.42, 0.47, 0.52, 1.0))
ALUMINIUM = Material(friction=0.4, rgba=(0.72, 0.74, 0.78, 1.0))
SCALE_BODY = Material(friction=0.6, rgba=(0.88, 0.88, 0.86, 1.0))
STEEL = Material(friction=0.5, rgba=(0.62, 0.64, 0.66, 1.0))
# The Robotiq's rubber pads, raised from the model's 0.6 to 0.7 as in v1 (its pads press about 3 N, a real
# 2F-85 at least 20 N), so a held tube does not turn in the fingers.
PAD_FRICTION = 1.2


# ---------------------------------------------------------------------------------------------------------
# Tubes
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Tube:
    """A 50 ml conical centrifuge tube, described by its outer outline.

    ``outline`` runs from the tip to the rim as (height, outer radius) points. The collision walls are a
    ring of thin boxes along it (as in v1, the way AutoBio builds its tubes), so balls inside fill the cone
    and the body; the AutoBio mesh is only drawn.
    """

    outline: tuple[tuple[float, float], ...]
    wall: float
    wall_pieces: int
    body_mass: float
    cap_mass: float
    cap_radius: float
    cap_height: float

    @property
    def height(self) -> float:
        return self.outline[-1][0]

    @property
    def radius(self) -> float:
        return max(r for _, r in self.outline)

    @property
    def capped_height(self) -> float:
        return self.height + self.cap_height - CAP_OVERLAP


# Measured from AutoBio's 50 ml tube mesh: a 15 mm cone, then a 29 mm wide body, narrowing to 28 mm at the
# threads. 114.6 mm long without the cap.
# Real tube walls are about 1 mm; the collision wall is 2 mm, as in v1, so 7 mm balls cannot tunnel through.
# A 50 ml Falcon-type tube weighs about 11.5 g, its cap about 2 g.
TUBE = Tube(
    outline=(
        (0.0, 0.0048),
        (0.005, 0.0076),
        (0.010, 0.0108),
        (0.015, 0.0140),
        (0.105, 0.0146),
        (0.1146, 0.0140),
    ),
    wall=0.002,
    wall_pieces=16,
    body_mass=0.0115,
    cap_mass=0.0020,
    cap_radius=0.0175,
    cap_height=0.0123,
)
# The cap's skirt reaches this far down over the tube's threads.
CAP_OVERLAP = 0.0073
# Every real tube weighs a little differently. Each tube's body mass is drawn from this spread (± g/1000),
# so two tubes filled with the same volume still weigh differently.
TUBE_MASS_SPREAD = 0.00015


@dataclass(frozen=True)
class Liquid:
    """The liquid, as small frictionless balls: the same balls as v1."""

    # 7 mm balls. 1 ml of "liquid" weighs 1 g: the balls are made denser than water by the fraction of the
    # space they fill once settled (about 55 %), so the balance reads real weights.
    ball_radius: float = 0.0035
    packing: float = 0.55
    water_density: float = 1000.0

    @property
    def ball_volume(self) -> float:
        return 4 / 3 * math.pi * self.ball_radius**3

    @property
    def ball_density(self) -> float:
        return self.water_density / self.packing

    @property
    def ball_mass(self) -> float:
        return self.ball_volume * self.ball_density


LIQUID = Liquid()

# Colours for the samples (blood, plasma, a green buffer, a purple stain, an orange extract) and the water.
SAMPLE_COLOURS = (
    (0.70, 0.05, 0.08, 1.0),
    (0.95, 0.80, 0.25, 1.0),
    (0.20, 0.65, 0.30, 1.0),
    (0.50, 0.20, 0.65, 1.0),
    (0.95, 0.50, 0.10, 1.0),
)
WATER_COLOUR = (0.15, 0.45, 0.95, 1.0)


@dataclass(frozen=True)
class Samples:
    """How much the seed may put in each sample tube, in balls (one ball is about 0.33 g)."""

    # About 10 to 33 ml. At most 100 balls so a balance tube to match (no cap, so 2 g lighter empty) needs at
    # most about 106 balls of water: an open tube that full does not spill at 35 degrees in the rotor.
    min_balls: int = 30
    max_balls: int = 100
    # Two samples meant to pair hold this many balls more or less than each other, at most.
    pair_spread: int = 1
    # A sample meant to have no partner differs from every other sample by at least this many balls
    # (about 3 g), far more than the balance tolerance.
    odd_gap: int = 9


SAMPLES = Samples()

# The water the arm pours into balance tubes: open 50 ml tubes, each filled this full (of its inside height):
# 126 balls, 41 g. One water tube per balance tube: a balance tube needs up to 106 balls (34.6 g), and the two
# of the "mismatch" job up to 198 together, more than one tube holds (step-3-fill-balance-tube/docs).
WATER_LEVEL = 0.85
WATER_TUBES = ("water0", "water1")
# Their cells in the supply rack: the two corners nearest the arm, with the cells beside them left empty, so
# the arm can grip them from the side (see grasp-plan/docs/results.md). The two empty tubes stand in the only
# cells left, in the back row's middle (each is diagonal to one water tube).
WATER_CELLS = (0, 3)
BLANK_CELLS = (5, 6)
WATER_CELL = WATER_CELLS[0]


# ---------------------------------------------------------------------------------------------------------
# Racks
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Rack:
    """A rack for 50 ml tubes: a floor and a top plate with square holes, 2 rows of 4.

    The cells are 60 mm apart, far wider than the 36 mm of v1's AutoBio rack: the tubes' caps are 35 mm
    wide, and the open fingers need room between two neighbouring tubes.
    """

    rows: int = 2
    columns: int = 4
    pitch: float = 0.060
    hole: float = 0.031
    height: float = 0.060
    plate: float = 0.004
    mass: float = 0.25


RACK = Rack()


# ---------------------------------------------------------------------------------------------------------
# Centrifuge
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Rotor:
    """Our fixed-angle rotor for six 50 ml tubes, in the centrifuge's own frame.

    The housing and lid are AutoBio's Eppendorf 5430. Its own rotor (30 x 1.5 ml) is replaced by one like
    Eppendorf's F-35-6-30: six 15/50 ml conical tubes at 35 degrees. Sizes are chosen so capped tubes clear
    the closed lid (its underside is 203 mm up) and the rotor fits the 141 mm wide chamber.
    """

    # The rotor's axis, in the housing's frame (AutoBio's rotor axis).
    axis_xy: tuple[float, float] = (0.0, 0.036)
    # The flat top of the rotor, and its bottom.
    top: float = 0.148
    bottom: float = 0.070
    radius: float = 0.128
    # Each hole's centre where it meets the top, from the axis.
    hole_radius_at_top: float = 0.070
    # The holes lean in at the top: a tube's axis is this far from vertical.
    tilt_deg: float = 35.0
    # How far a tube goes in, along its axis, from the top's plane; the rest sticks out to be held.
    depth: float = 0.075
    # Radius of each hole as drawn. What collides follows the tube's outline, 0.9 mm wider all round
    # (centrifuge.CLEARANCE), with a lead-in at the top.
    hole: float = 0.0164
    wall: float = 0.002
    wall_pieces: int = 16
    slots: int = 6
    # A real F-35-6-30 is about 2.6 kg; here the rotor never spins, so it is fixed to the housing.
    mass: float = 2.6

    @property
    def tilt(self) -> float:
        return math.radians(self.tilt_deg)


ROTOR = Rotor()

# The lid joint, as AutoBio built it: 0 is fully open (a gas spring holds it there), -1.723 rad closed.
LID_OPEN = 0.0
LID_CLOSED = -1.723
# Closer to shut than this, the latch catches and holds the lid closed.
LID_LATCH = 0.03

# Our balance rule: the pull of all tubes together, out from the axis, may be at most this many grams at
# the holes' radius. Each rotor's manual gives its own limit; this one is chosen for the demo.
BALANCE_TOLERANCE = 0.0010


# ---------------------------------------------------------------------------------------------------------
# The balance
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Scale:
    """A lab balance with a tube holder on its pan. Reads to 0.01 g, as a precision balance does."""

    base: tuple[float, float, float] = (0.10, 0.12, 0.035)  # half sizes
    pan_radius: float = 0.060
    pan_thickness: float = 0.004
    # The holder: a square sleeve a tube stands in, cone tip on the pan.
    holder_hole: float = 0.031
    holder_height: float = 0.050
    holder_wall: float = 0.004
    readability: float = 0.00001  # 0.01 g
    mass: float = 2.0


SCALE = Scale()


# ---------------------------------------------------------------------------------------------------------
# Layout and cameras
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Layout:
    """Where everything stands. Every place a tube is picked or put is 0.35 to 0.65 m from the arm's base."""

    sample_rack: tuple[float, float] = (0.42, 0.30)
    supply_rack: tuple[float, float] = (0.20, 0.42)
    scale: tuple[float, float] = (0.45, 0.02)
    # The centrifuge's own origin (its bottom centre), and its turn: its hinge (local +y) faces away from the
    # arm, so the lid opens away and the arm reaches in from the front.
    centrifuge: tuple[float, float] = (0.50, -0.36)
    centrifuge_yaw: float = math.atan2(-0.36, 0.50) - math.pi / 2


LAYOUT = Layout()


@dataclass(frozen=True)
class Camera:
    name: str
    pos: tuple[float, float, float]
    xyaxes: tuple[float, ...]
    fovy: float


def _look(name: str, pos: tuple[float, float, float], at: tuple[float, float, float], fovy: float) -> Camera:
    """A camera at ``pos`` looking at ``at``, with the picture's top towards +z."""
    f = [a - p for a, p in zip(at, pos, strict=True)]
    n = math.sqrt(sum(c * c for c in f))
    f = [c / n for c in f]
    x = [f[1], -f[0], 0.0]  # forward x up
    nx = math.hypot(x[0], x[1]) or 1.0
    x = [x[0] / nx, x[1] / nx, 0.0]
    y = [x[1] * f[2] - x[2] * f[1], x[2] * f[0] - x[0] * f[2], x[0] * f[1] - x[1] * f[0]]
    return Camera(name, pos, (x[0], x[1], x[2], y[0], y[1], y[2]), fovy)


# The overhead camera sees the whole work area: it is what the arm (and Gemini Robotics-ER) look at to find
# the racks, tubes, balance and centrifuge. The front camera is for people watching. The rotor camera looks
# down into the open centrifuge, to check which holes are full.
OVERHEAD = Camera("overhead", pos=(0.38, -0.02, 1.25), xyaxes=(0, -1, 0, 1, 0, 0), fovy=55.0)
FRONT = _look("front", pos=(1.05, 0.62, 0.78), at=(0.40, -0.10, 0.08), fovy=52.0)
ROTOR_CAMERA = _look("rotor", pos=(0.36, -0.30, 0.62), at=(0.53, -0.39, 0.15), fovy=40.0)
# Two cameras on the racks, for Step 0, both on posts at the far edge of the bench (out of the arm's reach). From
# straight above, the caps hide the samples' liquid and every tube is about 10 pixels wide. The high one sees
# every tube and cap with nothing in front of it; the low one sees the liquid through the racks' open sides,
# even in a tube a third full (its liquid is below the rack's top plate). Chosen by counting, on 12 seeds, the
# pixels of each tube and of its liquid that candidate cameras see (step-0-find/docs/approach.md).
RACKS_MIDDLE = (0.31, 0.36, 0.06)
RACKS_HIGH = _look("racks_high", pos=(0.95, 0.60, 0.70), at=RACKS_MIDDLE, fovy=30.0)
RACKS_LOW = _look("racks_low", pos=(1.00, 0.36, 0.22), at=RACKS_MIDDLE, fovy=33.0)
IMAGE_SIZE = (640, 480)
