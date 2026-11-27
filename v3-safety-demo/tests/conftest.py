"""Puts the demo folder and v3's code on the import path (v3 itself is imported, never changed).

v3's scene package for everything; Step 1's folder for its ``motion`` module (the arm's moves and inverse
kinematics), which the demo's pictures use for the arm's poses.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
V3 = HERE.parent / "v3-centrifuge-loading"
for p in (HERE, V3 / "simulation-environment", V3 / "step-1-weigh"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
