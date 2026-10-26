"""Every number that describes the lab scene, in one place.

The arm's base is at the origin, bolted to the table, and the table top is
the plane z = 0. x points away from the arm, y to its left. Lengths are in
metres, masses in kilograms.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
MENAGERIE = ASSETS / "menagerie"
ARM_XML = MENAGERIE / "universal_robots_ur5e" / "ur5e.xml"
GRIPPER_XML = MENAGERIE / "robotiq_2f85" / "2f85.xml"
AUTOBIO = ASSETS / "autobio" / "autobio" / "assets"
GLASSWARE = ASSETS / "glassware" / "Lab Assets" / "GLB" / "All"
# Meshes converted from the downloads: turned z-up and stretched to real size.
CONVERTED = ASSETS / "converted"

# 1 ms: a ball falling 15 cm moves about 1.7 mm per step, less than half its
# radius, so it cannot pass through a 2 mm glass wall in one step.
TIMESTEP = 0.001

# Each arm starts here, in its own base's frame: upright, gripper pointing
# straight down, clear of the racks and of the overhead camera's view of them.
HOME = (-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0)


@dataclass(frozen=True)
class ArmBase:
    """Where an arm is bolted to the table, and which way it faces (yaw, radians)."""

    name: str
    pos: tuple[float, float, float]
    yaw: float


# Two UR5e arms facing each other across the table, 0.9 m apart. Everything
# they work on stands on the line halfway between them, so each arm can
# reach every rack, the tray and the stand. Arm 1's base is the origin.
ARMS = (
    ArmBase("arm1", (0.0, 0.0, 0.0), 0.0),
    ArmBase("arm2", (0.90, 0.0, 0.0), 3.141592653589793),
)


@dataclass(frozen=True)
class Material:
    """How a surface behaves in contact. Friction is MuJoCo's sliding coefficient."""

    friction: float
    rgba: tuple[float, float, float, float]


# Clear borosilicate glass and the polypropylene of centrifuge tubes and racks.
# Glass on plastic or rubber is about 0.4 to 0.6 dry.
GLASS = Material(friction=0.5, rgba=(0.85, 0.92, 0.95, 0.30))
POLYPROPYLENE = Material(friction=0.5, rgba=(0.95, 0.95, 0.92, 0.45))
RACK_PLASTIC = Material(friction=0.6, rgba=(0.20, 0.45, 0.80, 1.0))
TABLE = Material(friction=0.7, rgba=(0.82, 0.82, 0.80, 1.0))


@dataclass(frozen=True)
class Container:
    """One piece of glassware, described by its outer outline.

    ``outline`` runs from the bottom to the rim as (height, outer radius)
    points; a flat disc closes the bottom at the first radius. The collision
    walls are built along the outline and the downloaded mesh is stretched
    to the same height and widest radius, so what the camera sees and what
    the balls hit are the same shape.
    """

    name: str
    mesh: Path
    mesh_up: str  # "y" for the GLB pack, "z" for AutoBio's OBJ files
    outline: tuple[tuple[float, float], ...]
    wall: float  # collision wall thickness, see the note on WALL below
    mass: float
    material: Material
    wall_pieces: int  # boxes around the circumference

    @property
    def height(self) -> float:
        return self.outline[-1][0]

    @property
    def radius(self) -> float:
        return max(r for _, r in self.outline)


# Real glass is 1 to 1.5 mm thick. The collision wall is 2 mm so balls cannot
# tunnel through it; the inside is 0.5 to 1 mm narrower than the real thing.
WALL = 0.002


def _test_tube(name: str, length: float, mass: float) -> Container:
    """A 25 mm glass test tube with a round bottom, as standard 25 x L tubes are."""
    r = 0.0125
    return Container(
        name=name,
        mesh=GLASSWARE / "bottle_glassware_test_tube_large.glb",
        mesh_up="y",
        # The round bottom as a quarter circle in three straight pieces.
        outline=((0.0, 0.0043), (0.0037, 0.0088), (0.0082, 0.0117), (r, r), (length, r)),
        wall=WALL,
        mass=mass,
        material=GLASS,
        wall_pieces=16,
    )


# The receiving tubes: one diameter, three heights, so "50 %" is a different
# volume in each (about 17, 27 and 36 ml). Masses are typical for 25 mm tubes.
RECEIVERS = (
    _test_tube("receiver_100", 0.100, 0.019),
    _test_tube("receiver_150", 0.150, 0.028),
    _test_tube("receiver_200", 0.200, 0.038),
)

SOURCES = (
    _test_tube("source_tube", 0.150, 0.028),
    # AutoBio's 50 ml centrifuge tube, at its own measured size: conical tip,
    # then straight. Polypropylene, about 13 g.
    Container(
        name="source_centrifuge_tube",
        mesh=AUTOBIO / "container" / "centrifuge_50ml_vis" / "visual.obj",
        mesh_up="z",
        outline=((0.0, 0.0035), (0.0156, 0.0145), (0.1146, 0.0145)),
        wall=WALL,
        mass=0.013,
        material=POLYPROPYLENE,
        wall_pieces=16,
    ),
    # A 50 ml Griffin beaker: 42 mm across, 60 mm tall, about 35 g.
    Container(
        name="source_beaker",
        mesh=GLASSWARE / "bottle_glassware_beaker_medium.glb",
        mesh_up="y",
        outline=((0.0, 0.021), (0.060, 0.021)),
        wall=WALL,
        mass=0.035,
        material=GLASS,
        wall_pieces=24,
    ),
    # A 50 ml Erlenmeyer flask: 51 mm base, 80 mm tall, 22 mm neck, about
    # 45 g. The neck starts at 72 % of the height, as in the mesh.
    Container(
        name="source_flask",
        mesh=GLASSWARE / "bottle_glassware_erlenmeyer_flask_medium.glb",
        mesh_up="y",
        outline=((0.0, 0.0255), (0.010, 0.0255), (0.058, 0.011), (0.080, 0.011)),
        wall=WALL,
        mass=0.045,
        material=GLASS,
        wall_pieces=24,
    ),
)

# The test tube and the centrifuge tube stand in the source rack; the beaker
# and the flask stand on the tray beside it.
SOURCES_IN_RACK = ("source_tube", "source_centrifuge_tube")


@dataclass(frozen=True)
class Liquid:
    """The liquid, as small frictionless balls."""

    # 7 mm balls, like small marbles. Published pouring work with rigid
    # balls uses a few hundred; 5 mm balls needed about 2,000 here and ran
    # 25 times slower than real time.
    ball_radius: float = 0.0035
    # 7 mm balls settle at about 55 % of the space they fill (measured per
    # container by liquid.settled_packing, between 50 and 56 %). The
    # balls are made denser than water by the same factor, so one millilitre
    # of "liquid" weighs one gram, and a force sensor feels real weights.
    packing: float = 0.55
    water_density: float = 1000.0
    rgba: tuple[float, float, float, float] = (0.15, 0.45, 0.95, 1.0)
    # Each source starts filled to a height between these fractions of its
    # inside height, drawn from the seed.
    level: tuple[float, float] = (0.20, 0.90)

    @property
    def ball_density(self) -> float:
        return self.water_density / self.packing


LIQUID = Liquid()


@dataclass(frozen=True)
class Rack:
    """AutoBio's 10-slot tube rack: 2 rows of 5 square cells, 30 mm wide, 36 mm apart."""

    mass: float = 0.15
    height: float = 0.060
    pitch: float = 0.036
    rows: int = 2
    columns: int = 5


RACK = Rack()


@dataclass(frozen=True)
class Layout:
    """Where everything stands on the table: in a row along y, halfway between the arms.

    Sources on one side of the stand, receivers on the other. The furthest
    cell is about 0.6 m from either arm's base, well inside its reach.
    """

    stand: tuple[float, float] = (0.45, 0.0)
    source_rack: tuple[float, float] = (0.45, 0.22)
    # The beaker and the flask stand on this tray, beyond the source rack.
    tray: tuple[float, float] = (0.45, 0.37)
    tray_size: tuple[float, float] = (0.20, 0.11)
    empty_rack: tuple[float, float] = (0.45, -0.22)
    filled_rack: tuple[float, float] = (0.45, -0.36)


LAYOUT = Layout()


@dataclass(frozen=True)
class Camera:
    name: str
    pos: tuple[float, float, float]
    xyaxes: tuple[float, ...]
    fovy: float


# The overhead camera sees the whole work area; it is what the arm uses to
# find the racks. The front camera is only for people to watch.
OVERHEAD = Camera("overhead", pos=(0.40, 0.0, 1.20), xyaxes=(0, -1, 0, 1, 0, 0), fovy=50.0)
# From the long side of the table, between the two arms, where a person
# watching would stand; the arms are left and right of the picture.
FRONT = Camera("front", pos=(0.45, -1.05, 0.60), xyaxes=(1, 0, 0, 0, 0.464, 0.886), fovy=55.0)
IMAGE_SIZE = (640, 480)
