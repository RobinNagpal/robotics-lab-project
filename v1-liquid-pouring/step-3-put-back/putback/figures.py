"""Pictures of one whole run for the docs: eight moments seen from the table's front-left corner, and
an animation.

python -m putback.figures --start seed150_size200 --out ../docs/solutions/images/step-3-put-back
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
from PIL import Image, ImageDraw, ImageFont

from . import workflow

FONT = ImageFont.load_default(size=16)
EVERY = 0.4  # s between pictures
GIF_EVERY = 2  # every second picture goes into the animation
# Where the pictures are taken from: above the front-left corner of the table, looking at the racks.
VIEW = {"lookat": (0.42, -0.04, 0.15), "azimuth": -135.0, "elevation": -30.0, "distance": 1.35}
# (caption, which frames it may be taken from, which of them)
MOMENTS = [
    ("1. Arm B tips the source over the receiver", lambda s: s == "find flow", 0.5),
    ("2. Pouring; the CNN reads the level", lambda s: s == "pour", 0.6),
    ("3. Tipped back: 50 % reached", lambda s: s == "done", 0.0),
    ("4. Arm B tips the source upright", lambda s: s == "tipping back", 0.7),
    ("5. Arm B carries it to its own cell", lambda s: s.startswith("arm1 tube"), 0.3),
    ("6. Set down; Arm B lets go", lambda s: s == "arm1 let go", 0.15),
    ("7. Arm A takes the receiver to the destination rack", lambda s: s.startswith("arm2 tube"), 0.4),
    ("8. Both tubes in racks, both arms home", lambda s: True, 1.0),
]


def _caption(img: Image.Image, text: str) -> Image.Image:
    out = Image.new("RGB", (img.width, img.height + 26), (20, 20, 20))
    out.paste(img, (0, 26))
    ImageDraw.Draw(out).text((8, 4), text, fill=(255, 255, 255), font=FONT)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--start", default="seed150_size200")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frames: list[tuple[str, Image.Image]] = []
    renderer: dict = {}
    last = [-math.inf]

    def on_frame(data: mujoco.MjData, stage: str) -> None:
        if data.time - last[0] < EVERY - 1e-6:
            return
        last[0] = data.time
        if "r" not in renderer:
            renderer["r"] = mujoco.Renderer(data.model, 480, 640)
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = VIEW["lookat"]
            cam.azimuth, cam.elevation, cam.distance = VIEW["azimuth"], VIEW["elevation"], VIEW["distance"]
            renderer["cam"] = cam
        r = renderer["r"]
        r._gl_context.make_current()  # the pour camera's renderer is open too: draw with this one's own
        r.update_scene(data, camera=renderer["cam"])
        frames.append((stage, Image.fromarray(r.render().copy())))  # copied: the renderer reuses its memory

    result = workflow.run_one(workflow.STARTS / f"{args.start}.npy", on_frame=on_frame)
    args.out.mkdir(parents=True, exist_ok=True)
    picks = []
    for caption, wanted, where in MOMENTS:
        chosen = [img for stage, img in frames if wanted(stage)]
        if chosen:
            picks.append(_caption(chosen[min(len(chosen) - 1, int(where * (len(chosen) - 1)))], caption))
    w, h = picks[0].size
    sheet = Image.new("RGB", (4 * w + 3 * 8, 2 * h + 8), (255, 255, 255))
    for i, img in enumerate(picks):
        sheet.paste(img, ((i % 4) * (w + 8), (i // 4) * (h + 8)))
    sheet.resize((sheet.width // 2, sheet.height // 2), Image.LANCZOS).save(args.out / "whole-run.png")
    small = [img.resize((320, 240), Image.LANCZOS) for _, img in frames[::GIF_EVERY]]
    small[0].save(
        args.out / "whole-run.gif",
        save_all=True,
        append_images=small[1:],
        duration=120,
        loop=0,
        optimize=True,
    )
    (args.out / "whole-run.json").write_text(json.dumps(result, indent=1, default=str))
    print(f"{len(frames)} frames; done {result['done']}, level {result['level_pct']} %")


if __name__ == "__main__":
    main()
