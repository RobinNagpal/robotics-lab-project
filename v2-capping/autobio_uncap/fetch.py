"""AutoBio, pinned, in its own clone at assets/autobio-full/ (assets/ is not committed).

Only what the uncapping task needs is checked out: the Python code and every model file (small), the
ALOHA arm's meshes, the rack's and the 50 ml screw tube's, and the Linux plugin (only to read its
symbols: it does not run here).

    python -m autobio_uncap.fetch
"""

from __future__ import annotations

import subprocess
from pathlib import Path

URL = "https://github.com/autobio-bench/AutoBio"
COMMIT = "08504213356cbfe9a82bcd33a43cd282945a45df"
DIR = Path(__file__).resolve().parents[1] / "assets" / "autobio-full"
PATHS = [
    "/README.md",
    "/autobio/README.md",
    "/autobio/*.py",
    "/autobio/grasp/",
    "/autobio/model/",
    "/autobio/libmjlab.so.3.3.0",
    "/autobio/assets/robot/aloha2/",
    "/autobio/assets/rack/centrifuge_10slot_vis/",
    "/autobio/assets/container/centrifuge_50ml_screw_vis/",
]


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=DIR, check=True, capture_output=True, text=True).stdout


def ensure() -> Path:
    """The clone, made if missing, at the pinned commit."""
    if not (DIR / ".git").exists():
        DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--quiet", "--filter=blob:none", "--no-checkout", URL, str(DIR)], check=True
        )
    if not (DIR / "autobio" / "screw_loosen.py").exists() or _git("rev-parse", "HEAD").strip() != COMMIT:
        _git("sparse-checkout", "set", "--no-cone", *PATHS)
        _git("checkout", "--quiet", COMMIT)
    return DIR


if __name__ == "__main__":
    print(ensure(), _git("rev-parse", "HEAD").strip())
