"""The diagrams for docs/capping.md, drawn from the cap's own numbers (caps.py).

    python -m capping.diagrams

cap-cross-section.png   the cap on the tube, cut through the middle, with its sizes
cap-physics.png         as the cap is turned: how far it rises, and how hard it is to turn
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

from . import caps  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
BLUE, ORANGE = "#2a78d6", "#eb6834"
GLASS, GLASS_EDGE, CAP, RIB = "#dce9f7", "#2a78d6", "#2b2b2b", "#6b6b6b"
OUT = Path("docs/images")
TUBE_R = 12.5  # mm, the 25 mm receivers (v1)
GLASS_WALL = 1.2  # mm, drawn only (real glass is 1 to 1.5 mm)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "font.size": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.8,
        "lines.solid_capstyle": "round",
    }
)


def _dim(ax, a, b, text, side="above", offset=0.0, color=INK) -> None:
    """A dimension arrow from a to b with its text."""
    ax.annotate(
        "", a, b, arrowprops={"arrowstyle": "<->", "color": color, "lw": 0.9, "shrinkA": 0, "shrinkB": 0}
    )
    mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    if side == "above":
        ax.text(mx, my + 0.6 + offset, text, ha="center", va="bottom", color=color, fontsize=9)
    elif side == "below":
        ax.text(mx, my - 0.6 - offset, text, ha="center", va="top", color=color, fontsize=9)
    else:
        ax.text(mx + 0.8 + offset, my, text, ha="left", va="center", color=color, fontsize=9)


def cross_section(out: Path) -> None:
    mm = 1000.0
    outer, inner, rib = caps.OUTER_R * mm, caps.INNER_R * mm, caps.RIB * mm
    top, skirt = caps.TOP * mm, caps.SKIRT * mm
    seat = caps.GAP * mm + 1.0  # inside of the cap's top above the drawn rim (walls reach ~1 mm above it)
    z_in, z_top = seat, seat + top
    z_bottom = z_top - caps.HEIGHT * mm
    fig, ax = plt.subplots(figsize=(8.6, 5.6), dpi=160)
    ax.set_aspect("equal")
    ax.axis("off")
    # The tube: two glass walls, open at the top.
    for s in (-1, 1):
        x0 = s * TUBE_R
        ax.add_patch(
            Rectangle((min(x0, x0 - s * GLASS_WALL), -20), GLASS_WALL, 20, fc=GLASS, ec=GLASS_EDGE, lw=1)
        )
    # The cap: the top, and the two sides with their ribs.
    ax.add_patch(Rectangle((-(outer + rib), z_in), 2 * (outer + rib), top, fc=CAP, ec=CAP))
    for s in (-1, 1):
        x_in = s * inner
        ax.add_patch(Rectangle((min(x_in, s * outer), z_bottom), outer - inner, skirt, fc=CAP, ec=CAP))
        ax.add_patch(
            Rectangle((min(s * outer, s * (outer + rib)), z_bottom), rib, skirt + top, fc=RIB, ec=RIB)
        )
    # Where a real tube's thread would be: small ridges, 3.175 mm apart, 1.5 turns (drawn on one side).
    pitch = caps.PITCH * mm
    for k in range(2):
        zc = z_in - 2.2 - k * pitch
        for s in (-1, 1):
            zc_side = zc - (pitch / 2 if s < 0 else 0)  # a helix: the other side is half a turn lower
            ax.add_patch(
                Polygon(
                    [
                        (s * TUBE_R, zc_side - 0.7),
                        (s * (TUBE_R + 0.45), zc_side),
                        (s * TUBE_R, zc_side + 0.7),
                    ],
                    fc=BLUE,
                    ec=BLUE,
                )
            )
    # Sizes.
    _dim(ax, (-(outer + rib), z_top + 3.2), (outer + rib, z_top + 3.2), "28.8 mm across the grip ribs")
    _dim(ax, (-outer, z_top + 0.8), (outer, z_top + 0.8), "28.0 mm between the ribs", offset=-0.1)
    _dim(ax, (-inner, z_bottom - 1.8), (inner, z_bottom - 1.8), "26.0 mm inside the cap", side="below")
    _dim(ax, (-TUBE_R, -17.5), (TUBE_R, -17.5), "25 mm glass tube", side="below", color=GLASS_EDGE)
    xr = outer + rib + 2.2
    _dim(ax, (xr, z_bottom), (xr, z_top), "14 mm tall", side="right")
    xr2 = xr + 9.5
    _dim(ax, (xr2, z_in), (xr2, z_top), "2.5 mm top", side="right")
    _dim(ax, (xr2, z_bottom), (xr2, z_in), "11.5 mm side\n(the skirt)", side="right")
    # Labels.
    ax.annotate(
        "thread on a real tube: 1.5 turns,\n3.175 mm apart (24-410).\n"
        "Not drawn in the simulation:\ntwo joints do its job",
        (-TUBE_R - 0.3, z_in - 2.2 - pitch / 2),
        xytext=(-34, -6),
        fontsize=8.8,
        color=INK,
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.8},
    )
    ax.annotate(
        "black phenolic cap, 4 g,\n48 grip ribs (0.4 mm)",
        (-(outer + rib), z_top - 1.0),
        xytext=(-34, 7),
        fontsize=8.8,
        color=INK,
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.8},
    )
    ax.text(0, -8, "inside\nof the\ntube", ha="center", va="center", color=INK2, fontsize=8.5)
    ax.set_xlim(-35, 40)
    ax.set_ylim(-21, 8.5)
    ax.set_title("The 24-410 cap on a 25 mm tube, cut through the middle (sizes to scale)")
    fig.tight_layout()
    fig.savefig(out / "cap-cross-section.png")
    plt.close(fig)


def physics(out: Path) -> None:
    end = math.degrees(caps.OPEN_ANGLE)  # 540
    seal = math.degrees(caps.SEAL_ANGLE)  # 20
    pitch = caps.PITCH * 1000
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.6, 5.4), dpi=160, sharex=True)
    for ax in (a1, a2):
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.axvspan(end, 640, color="#f1efe9", lw=0)
    # 1. How far the cap rises: one pitch per turn, until the thread ends.
    a1.plot([0, end], [0, pitch * caps.TURNS], color=BLUE)
    a1.plot([end, 620], [pitch * caps.TURNS] * 2, color=BLUE, ls=(0, (3, 3)))
    a1.plot(end, pitch * caps.TURNS, "o", ms=6, color=BLUE, mec=SURFACE, mew=1.5)
    a1.text(end - 8, pitch * caps.TURNS + 0.35, "4.76 mm up: the end of the thread", ha="right", fontsize=9)
    a1.text(590, 2.4, "off:\nthe cap\nis free", ha="center", va="center", color=INK2, fontsize=9)
    for k in (1, 2, 3, 4, 5, 6):
        a1.axvline(90 * k, color=INK2, lw=0.6, ls=(0, (1, 3)))
    a1.text(45, 4.6, "dotted lines: every 90°", fontsize=8.5, color=INK2)
    a1.set_ylim(0, 5.6)
    a1.set_ylabel("how far it has risen (mm)")
    a1.set_title("As the cap is turned anticlockwise (the simulation's settings)")
    # 2. How hard it is to turn: the seal, then easy, then nothing.
    lo, hi = 0.45, 1.25  # the tightness range of the tests (uncap/episode.py TIGHTNESS)
    a2.fill_between([0, seal], [lo, lo], [hi, hi], color=ORANGE, alpha=0.18, lw=0)
    a2.fill_between([seal, end], [lo / 6] * 2, [hi / 6] * 2, color=ORANGE, alpha=0.18, lw=0)
    a2.plot([0, seal, seal, end, end, 620], [0.6, 0.6, 0.1, 0.1, 0.0, 0.0], color=ORANGE, drawstyle="default")
    a2.annotate(
        "first 20°: the seal breaks\n0.6 N m (tests: 0.45 to 1.25)",
        (seal, 0.6),
        xytext=(60, 0.92),
        fontsize=9,
        arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.8},
    )
    a2.text(280, 0.24, "then it turns easily: 0.1 N m (a sixth of the seal)", ha="center", fontsize=9)
    a2.text(590, 0.16, "nothing\nholds it", ha="center", va="bottom", color=INK2, fontsize=9)
    a2.set_ylim(0, 1.35)
    a2.set_xlim(0, 620)
    a2.set_ylabel("twist needed (N m)")
    a2.set_xlabel("how far the cap has been turned (degrees): 540° = 1.5 turns")
    a2.text(
        0, -0.45, "Shaded: the range in the stress tests and recorded runs (each run draws one tightness).",
        transform=a2.get_yaxis_transform(), fontsize=8.5, color=INK2,
    )  # fmt: skip
    fig.tight_layout()
    fig.savefig(out / "cap-physics.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cross_section(args.out)
    physics(args.out)
    print("written to", args.out)


if __name__ == "__main__":
    main()
