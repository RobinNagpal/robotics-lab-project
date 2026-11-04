"""Map of every candidate camera: how much of the receiver's 35-60 % band it sees in the worst pose, and
whether the arms come near it.

    python plot.py OUT_DIR
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out = Path(sys.argv[1])
c = json.loads((out / "candidates.json").read_text())
AZ = sorted({r["az"] for r in c})
EL = sorted({r["el"] for r in c})
DS = sorted({r["D"] for r in c})
CLEAR = 0.15  # m: a camera closer than this to where an arm went is in the way

fig, axes = plt.subplots(1, len(DS), figsize=(5.2 * len(DS), 4.2), constrained_layout=True)
for ax, D in zip(axes, DS, strict=True):
    grid = np.full((len(EL), len(AZ)), np.nan)
    hit = np.zeros_like(grid, bool)
    for r in c:
        if r["D"] != D:
            continue
        i, j = EL.index(r["el"]), AZ.index(r["az"])
        grid[i, j] = r["band_min"] * 100
        hit[i, j] = r["clearance_m"] < CLEAR or r["below_table"]
    im = ax.imshow(grid, origin="lower", cmap="viridis", vmin=0, vmax=100, aspect="auto")
    for i in range(len(EL)):
        for j in range(len(AZ)):
            if hit[i, j]:
                ax.text(j, i, "x", ha="center", va="center", color="red", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(AZ)), [str(a) if a % 45 == 0 else "" for a in AZ])
    ax.set_yticks(range(len(EL)), EL)
    ax.set_xlabel(
        "direction around the pouring point (degrees)\n"
        "0 = towards Arm A's base, 90 = source rack side, 270 = empty rack side"
    )
    ax.set_ylabel("height angle (degrees above level)")
    ax.set_title(f"camera {D} m away")
fig.colorbar(im, ax=axes, label="% of the 35-60 % band seen, worst pose")
fig.suptitle(
    "Which fixed camera sees the receiver's level? (red x: an arm passes within 15 cm, or below the table)"
)
fig.savefig(out / "camera_map.png", dpi=110)
