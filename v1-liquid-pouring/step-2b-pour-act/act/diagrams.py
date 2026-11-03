"""The diagrams for the docs, drawn as SVG in the same style as Step 2's.

docs/solutions/images/step-2b-pour-act/journey.svg    the five parts of Step 2b
docs/solutions/images/step-2b-pour-act/two-ways.svg   the programmed pour against the learned one
docs/solutions/images/step-2b-pour-act/loop.svg       what happens every 0.1 s while the policy pours
docs/images/act.svg                                   ACT, step by step
docs/images/mlp.svg                                   the CNN + MLP policy, step by step

python -m act.diagrams
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOLUTION_IMAGES = ROOT.parent / "docs" / "solutions" / "images" / "step-2b-pour-act"
IMAGES = ROOT / "docs" / "images"

FONT = "Helvetica,Arial,sans-serif"
INK, GREY, BLUE, GREEN, RED = "#1f2733", "#5b6675", "#2f5fa8", "#1e7a4c", "#b23a48"
PANEL, LINE = "#f6f8fb", "#d5dbe3"
LEARNED, PROGRAMMED, INPUT = "#fbe9ec", "#eef0f3", "#eaf1fb"


class Svg:
    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
            '<defs><marker id="ar" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto">'
            f'<path d="M0,0 L8,4 L0,8 z" fill="{GREY}"/></marker></defs>',
            f'<rect width="{w}" height="{h}" rx="10" fill="#ffffff"/>',
        ]

    def text(self, x, y, s, size=13, colour=INK, bold=False, anchor="start", italic=False):
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        style = ' font-style="italic"' if italic else ""
        self.parts.append(
            f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" fill="{colour}" '
            f'font-weight="{"bold" if bold else "normal"}" text-anchor="{anchor}"{style}>{s}</text>'
        )

    def box(self, x, y, w, h, fill=PANEL, stroke=LINE, dashed=False):
        dash = ' stroke-dasharray="6,4"' if dashed else ""
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}"{dash}/>'
        )

    def card(self, x, y, w, h, title, lines, fill=PANEL, title_size=14, size=12, dashed=False, stroke=LINE):
        """A box with a bold title and centred lines under it."""
        self.box(x, y, w, h, fill, stroke, dashed)
        cx = x + w / 2
        self.text(cx, y + 24, title, title_size, bold=True, anchor="middle")
        for i, line in enumerate(lines):
            self.text(cx, y + 46 + i * (size + 5), line, size, anchor="middle")

    def arrow(self, x1, y1, x2, y2, colour=GREY):
        self.parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{colour}" stroke-width="2" '
            'marker-end="url(#ar)"/>'
        )

    def path(self, d, colour=GREY):
        self.parts.append(
            f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="2" marker-end="url(#ar)"/>'
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join([*self.parts, "</svg>"]) + "\n")


def journey(out: Path) -> None:
    steps = [
        (
            "What changes",
            "Can a model move",
            ["the arm, not just", "read the level?"],
            ["Movement learned too:", "cameras in, joint", "targets out"],
        ),
        (
            "What it sees",
            "What does the policy",
            ["need to see?"],
            ["Receiver strip, whole", "pour picture, wrist", "camera, joints, target"],
        ),
        (
            "Demonstrations",
            "What does it copy?",
            [],
            ["227 programmed pours;", "half nudged, to show", "getting back on course"],
        ),
        (
            "Two policies",
            "Which kind of model?",
            [],
            ["ACT (transformer) and", "a CNN + MLP, same eyes,", "same data, 30 min each"],
        ),
        (
            "How well they copy",
            "Do they do what the",
            ["expert would do?"],
            ["Both within ~0.1 deg", "a tick; ACT better at", "tipping back, stopping"],
        ),
    ]
    s = Svg(1000, 240)
    w, gap = 180, 18
    for i, (title, q, q2, found) in enumerate(steps):
        x = 8 + i * (w + gap)
        s.box(x, 14, w, 212)
        s.parts.append(f'<circle cx="{x + 24}" cy="40" r="14" fill="{BLUE}"/>')
        s.text(x + 24, 45, str(i + 1), 14, "#fff", True, "middle")
        s.text(x + 46, 46, title, 14 if len(title) < 16 else 13, bold=True)
        s.text(x + 12, 78, "QUESTION", 11, BLUE, True)
        for j, line in enumerate([q, *q2]):
            s.text(x + 12, 96 + j * 17, line, 13)
        s.text(x + 12, 158, "FOUND", 11, GREEN, True)
        for j, line in enumerate(found):
            s.text(x + 12, 176 + j * 17, line, 12.5)
        if i < len(steps) - 1:
            s.arrow(x + w + 2, 114, x + w + gap - 2, 114)
    s.save(out / "journey.svg")


def two_ways(out: Path) -> None:
    s = Svg(1000, 340)
    s.text(16, 30, "Step 2: the programmed pour", 15, bold=True)
    s.text(16, 48, "Only the level reading is learned; every move is worked out by code", 12, GREY)
    row = [
        ("Pour camera", ["receiver strip"], INPUT),
        ("Level network", ["level, flow,", "balls in the air"], LEARNED),
        ("Pour rules", ["find the flow, pour", "slowly, tip back", "early, top up"], PROGRAMMED),
        ("Tilt and lip plan", ["tilt angle, lip 15 mm", "out, 20 mm above"], PROGRAMMED),
        ("Arm maths", ["inverse kinematics,", "tube-in-fingers fix"], PROGRAMMED),
        ("Arm B", ["6 joint targets,", "every 0.01 s"], INPUT),
    ]
    w, gap, y = 150, 18, 62
    for i, (title, lines, fill) in enumerate(row):
        x = 8 + i * (w + gap)
        s.card(x, y, w, 100, title, lines, fill, 13, 12)
        if i < len(row) - 1:
            s.arrow(x + w + 2, y + 50, x + w + gap - 2, y + 50)
    s.text(16, 200, "Step 2b: the learned pour", 15, bold=True)
    s.text(16, 218, "One policy goes straight from what the cameras see to Arm B's joint targets", 12, GREY)
    y = 232
    s.card(8, y, 320, 96, "What it sees, every 0.1 s", [], INPUT, 13)
    for j, line in enumerate(
        [
            "receiver strip (3 pictures) + whole pour",
            "picture + wrist camera + Arm B's joints",
            "+ the target level",
        ]
    ):
        s.text(168, y + 46 + j * 17, line, 12, anchor="middle")
    s.arrow(330, y + 48, 362, y + 48)
    s.card(
        364,
        y,
        300,
        96,
        "Policy: ACT or CNN + MLP",
        ["copied from the programmed pour", "(imitation learning)"],
        LEARNED,
        13,
    )
    s.arrow(666, y + 48, 698, y + 48)
    s.card(700, y, 292, 96, "Arm B", ["its 6 joint targets for the next 1 s,", 'and "finished?"'], INPUT, 13)
    # Legend.
    for k, (fill, name) in enumerate(((LEARNED, "learned"), (PROGRAMMED, "programmed"))):
        s.box(760 + k * 115, 18, 18, 18, fill)
        s.text(784 + k * 115, 32, name, 12)
    s.save(out / "two-ways.svg")


def loop(out: Path) -> None:
    s = Svg(1000, 250)
    steps = [
        ("1. Look", ["pour camera and wrist", "camera pictures,", "Arm B's joints"]),
        ("2. Think", ["the policy gives the", "next 1 s of joint", "targets; the first is used"]),
        ("3. Check", ["safety check: would it", "hit Arm A or the", "receiver? Then hold"]),
        ("4. Move", ["slide to the new targets", "over 0.1 s; the", "simulation runs"]),
    ]
    w, gap, y = 205, 40, 24
    for i, (title, lines) in enumerate(steps):
        x = 20 + i * (w + gap)
        s.card(x, y, w, 110, title, lines, LEARNED if i == 1 else PANEL, 15, 12.5)
        if i < len(steps) - 1:
            s.arrow(x + w + 4, y + 55, x + w + gap - 4, y + 55)
    last = 20 + 3 * (w + gap) + w / 2
    first = 20 + w / 2
    s.path(f"M{last},{y + 112} L{last},{y + 160} L{first},{y + 160} L{first},{y + 116}")
    s.text(
        500,
        y + 154,
        'every 0.1 s, until the policy says "finished" 3 times running (or 60 s)',
        13,
        GREY,
        anchor="middle",
    )
    s.text(
        500,
        y + 196,
        "The simulation waits while the policy thinks, so the pictures and the joints always "
        "belong to the same moment.",
        12,
        GREY,
        anchor="middle",
        italic=True,
    )
    s.save(out / "loop.svg")


def act(out: Path) -> None:
    s = Svg(1000, 500)
    inputs = [
        ("Receiver strip", "3 pictures, 48 x 192", "Level network (frozen)", "12 tokens"),
        ("Whole pour picture", "128 x 96", "Small CNN", "48 tokens"),
        ("Wrist camera", "96 x 72", "Small CNN", "24 tokens"),
        ("Arm B's joints", "6 angles + 6 targets", "one layer", "1 token"),
        ("Target level", "e.g. 50 %", "one layer", "1 token"),
        ("Style z", "0 when pouring", "one layer", "1 token"),
    ]
    y0, h = 16, 52
    for i, (name, detail, eye, tokens) in enumerate(inputs):
        y = y0 + i * (h + 10)
        s.box(8, y, 170, h, INPUT)
        s.text(93, y + 22, name, 13, bold=True, anchor="middle")
        s.text(93, y + 40, detail, 11.5, anchor="middle")
        s.arrow(180, y + h / 2, 204, y + h / 2)
        s.box(206, y, 160, h, PROGRAMMED if "frozen" in eye else LEARNED)
        s.text(286, y + 22, eye, 12.5, bold=True, anchor="middle")
        s.text(286, y + 40, tokens, 11.5, anchor="middle")
        s.arrow(368, y + h / 2, 400, y0 + 3 * (h + 10) - 5)
    ty = y0 + 1.2 * (h + 10)
    s.card(
        402,
        ty,
        190,
        150,
        "Encoder",
        [
            "transformer, 4 layers",
            "",
            "all 87 tokens look at",
            "each other: which",
            "matter, and how they",
            "go together",
        ],
        LEARNED,
        14,
        12,
    )
    s.arrow(594, ty + 75, 622, ty + 75)
    s.card(
        624,
        ty,
        190,
        150,
        "Decoder",
        [
            "transformer, 4 layers",
            "",
            "10 learned questions,",
            "one per coming tick:",
            '"what should the arm',
            'do then?"',
        ],
        LEARNED,
        14,
        12,
    )
    s.arrow(816, ty + 75, 840, ty + 75)
    s.card(
        842,
        ty,
        150,
        150,
        "Out",
        ["for each of the", "next 10 ticks (1 s):", "", "6 joint targets", "+ finished?"],
        INPUT,
        14,
        12,
    )
    vy = 400
    s.card(
        402,
        vy - 20,
        412,
        72,
        'Only while training: the style encoder (the "CVAE" part)',
        [
            "sees the expert's real next 10 ticks and sums up their style in z;",
            "a penalty keeps z near 0, so pouring with z = 0 is the usual style",
        ],
        PANEL,
        12.5,
        11.5,
        dashed=True,
    )
    s.path(f"M500,{vy - 22} L500,{ty + 150 + 8}")
    s.text(
        8,
        485,
        "Inputs, then each picture is turned into tokens: one small summary per place in the picture.",
        12,
        GREY,
    )
    s.text(780, 485, "10.9 M weights (10.0 M trained)", 12, GREY)
    s.save(out / "act.svg")


def mlp(out: Path) -> None:
    s = Svg(1000, 300)
    rows = [
        ("Receiver strip", "Level network (frozen)", "12 rows x 128", "1,536"),
        ("Whole pour picture", "Small CNN", "where each of 128 features is strongest", "256"),
        ("Wrist camera", "Small CNN", "where each of 128 features is strongest", "256"),
        ("Joints, target", "as they are", "", "13"),
    ]
    y0, h = 16, 50
    for i, (name, eye, summary, n) in enumerate(rows):
        y = y0 + i * (h + 10)
        s.box(8, y, 150, h, INPUT)
        s.text(83, y + 30, name, 13, bold=True, anchor="middle")
        s.arrow(160, y + h / 2, 184, y + h / 2)
        s.box(186, y, 170, h, PROGRAMMED if "frozen" in eye else LEARNED if eye != "as they are" else PANEL)
        s.text(271, y + 30, eye, 12.5, bold=True, anchor="middle")
        s.arrow(358, y + h / 2, 382, y + h / 2)
        s.box(384, y, 270, h, PANEL)
        s.text(519, y + 22, f"{n} numbers", 12.5, bold=True, anchor="middle")
        s.text(519, y + 39, summary, 11, anchor="middle")
        s.arrow(656, y + h / 2, 684, y0 + 2 * (h + 10) - 5)
    s.card(
        686,
        40,
        140,
        170,
        "Two dense",
        ["layers", "", "2,061 numbers", "-> 1,024", "-> 1,024"],
        LEARNED,
        14,
        12,
    )
    s.arrow(828, 125, 848, 125)
    s.card(
        850,
        40,
        142,
        170,
        "Out",
        ["for each of the", "next 10 ticks:", "", "6 joint targets", "+ finished?"],
        INPUT,
        14,
        12,
    )
    s.text(
        8,
        280,
        '"Where each feature is strongest" is a spatial softmax (Levine et al., 2016): for each of the '
        "128 features, its average place in the picture.",
        12,
        GREY,
    )
    s.text(780, 262, "5.0 M weights (4.1 M trained)", 12, GREY)
    s.save(out / "mlp.svg")


def main() -> None:
    journey(SOLUTION_IMAGES)
    two_ways(SOLUTION_IMAGES)
    loop(SOLUTION_IMAGES)
    act(IMAGES)
    mlp(IMAGES)


if __name__ == "__main__":
    main()
