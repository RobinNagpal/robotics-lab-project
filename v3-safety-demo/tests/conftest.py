"""Puts the demo folder and v3's scene package on the import path (v3 itself is imported, never changed)."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
V3 = HERE.parent / "v3-centrifuge-loading"
for p in (HERE, V3 / "simulation-environment"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
