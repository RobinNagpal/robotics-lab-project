"""Pictures of the programmed pour for the docs: the four phases side by side, and an animation.

python -m pour.figures --start data/starts/seed150_size200.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import pour

# The part of the pour camera's picture with both tubes in it (x0, y0, x1, y1).
VIEW = (150, 0, 490, 480)
SHOWN = {
    "find flow": "1. tipped to 78 degrees",
    "pour": "2. pouring",
    "tip back": "3. tipping back",
    "done": "4. settled at the mark",
}


def _label(image: np.ndarray, text: str, scale: float) -> Image.Image:
    img = Image.fromarray(image).crop(VIEW)
    img = img.resize((int(img.width * scale), int(img.height * scale)))
    ImageDraw.Draw(img).text((8, 8), text, fill=(255, 230, 80))
    return img


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--start", type=Path, default=Path("data/starts/seed150_size200.npy"))
    parser.add_argument("--target", type=float, default=50.0)
    parser.add_argument("--out", type=Path, default=Path("docs/images"))
    parser.add_argument("--overview", type=Path, default=Path("../docs/solutions/images/step-2-pour"))
    args = parser.parse_args()
    movie: list = []
    result, frames = pour.run(args.start, args.target, movie=movie)
    print(result["level_pct"], result["balls"], result["log"])
    args.out.mkdir(parents=True, exist_ok=True)
    args.overview.mkdir(parents=True, exist_ok=True)

    # The four phases: the first picture of tipping in and back, the middle one of the others.
    phases = [pour.PHASES[p] for p in frames["phase"]]
    strip = []
    for phase, text in SHOWN.items():
        idx = [i for i, p in enumerate(phases) if p == phase]
        i = idx[len(idx) // 2] if phase in ("pour", "done") else idx[0]
        level = frames["level"][i]
        strip.append(_label(movie_frame(movie, i), f"{text}  ({level:.0f} %)", 0.75))
    sheet = Image.new("RGB", (sum(s.width for s in strip), strip[0].height))
    x = 0
    for s in strip:
        sheet.paste(s, (x, 0))
        x += s.width
    sheet.save(args.overview / "pour-phases.png")

    # The animation: every 0.2 s, from the start of tipping to the end.
    gif = [_label(img, f"{p}", 0.6) for img, p in movie]
    gif[0].save(
        args.overview / "pour.gif", save_all=True, append_images=gif[1:], duration=120, loop=0, optimize=True
    )


def movie_frame(movie: list, i: int) -> np.ndarray:
    """The whole picture nearest saved frame ``i`` (the movie keeps every second one)."""
    return movie[min(i // 2, len(movie) - 1)][0]


if __name__ == "__main__":
    main()
