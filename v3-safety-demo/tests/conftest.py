import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
for p in (
    HERE,
    HERE.parent / "v3-centrifuge-loading" / "simulation-environment",
    HERE.parent / "v3-centrifuge-loading" / "step-1-weigh",
):
    sys.path.insert(0, str(p))
