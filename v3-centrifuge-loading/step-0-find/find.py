"""Step 0: which rack cells hold a tube, and what each tube is, from the cameras.

For each picture, Gemini Robotics-ER 2 points at every tube and says what it is. Each point becomes a cell:
the pixel gives a line into the scene, the depth picture says how far along it the tube is, and the known
cell whose upright axis passes closest is the one. The pictures then vote, cell by cell.

Two ways to use several pictures:
- ``separate``: one question per picture; our code combines the answers (the vote above).
- ``together``: all pictures in one question; Gemini combines them and points in the first picture.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from centrifuge_scene.racks import floor_height

from bench import CELLS, TUBE_TOP, Cell, View

KINDS = ("sample", "water", "empty")
COLOURS = ("red", "yellow", "green", "purple", "orange", "blue", "none")

DESCRIBE = (
    "The blue racks hold 50 ml centrifuge tubes. A tube is one of three kinds: "
    '"sample" (it has an orange screw cap; it holds a coloured liquid), '
    '"water" (open, no cap, full of small blue balls), '
    '"empty" (open, no cap, nothing inside).'
)
FORMAT = (
    "Answer with a JSON list and nothing else, one entry per tube: "
    '[{"point": [y, x], "label": "<kind> <colour>", "kind": "<kind>", "colour": "<colour>"}]. '
    "The points are in [y, x] format normalized to 0-1000. "
    f"colour is the colour of the liquid inside: one of {', '.join(COLOURS[:-1])}, or none if empty."
)
PROMPT_ONE = (
    f"{DESCRIBE} Point to the top of every tube standing in the racks in this picture (at most 20). {FORMAT}"
)
PROMPT_TOGETHER = (
    f"{DESCRIBE} These {{n}} pictures show the same racks from different cameras. "
    "Find every tube standing in the racks (at most 20). Use all the pictures to decide each tube's kind "
    "and colour (a cap can hide the liquid from above; a rack's top plate can hide the liquid of a tube that "
    "is not very full). "
    f"Point to the top of each tube in the first picture. {FORMAT}"
)

# A point counts for a cell when its line passes this close to the cell's upright axis (half the 60 mm pitch).
NEAR = 0.030
# Look this many pixels round a point for the nearest surface: the tube may be clear plastic, see-through.
WINDOW = 4
# When the pictures disagree on a tube's kind, trust them in this order; for its colour, the low camera first.
KIND_ORDER = ("racks_high", "overhead", "racks_low")
COLOUR_ORDER = ("racks_low", "racks_high", "overhead")


@dataclass
class Detection:
    camera: str
    ray: tuple[np.ndarray, np.ndarray] | None
    kind: str
    colour: str
    pixel: tuple[float, float]  # (u across, v down)
    cell: str | None  # "sample_rack/3", or None if no cell is near
    position: np.ndarray | None  # where the depth picture puts it (m)


@dataclass
class Found:
    """One cell's answer after the vote."""

    kind: str
    colour: str
    votes: int  # pictures that saw a tube here
    of: int  # pictures asked
    detections: list[Detection] = field(default_factory=list)


def parse(text: str) -> list[dict]:
    """The JSON list in a model's answer (it may come wrapped in ```json fences, or with words round it)."""
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict) or not isinstance(it.get("point"), list) or len(it["point"]) != 2:
            continue
        label = str(it.get("label", "")).lower()
        kind = str(it.get("kind", "")).lower() or next((k for k in KINDS if k in label), "")
        colour = str(it.get("colour", it.get("color", ""))).lower() or next(
            (c for c in COLOURS if c in label), ""
        )
        out.append({"point": [float(it["point"][0]), float(it["point"][1])], "kind": kind, "colour": colour})
    return out


def locate(view: View, y: float, x: float) -> tuple[str | None, np.ndarray | None, tuple[float, float]]:
    """The cell a point (0-1000, [y, x]) is on, where the depth picture puts it, and its pixel."""
    cam = view.camera
    u, v = x / 1000 * cam.width, y / 1000 * cam.height
    iu, iv = int(np.clip(u, 0, cam.width - 1)), int(np.clip(v, 0, cam.height - 1))
    patch = view.depth[max(0, iv - WINDOW) : iv + WINDOW + 1, max(0, iu - WINDOW) : iu + WINDOW + 1]
    depth = float(patch.min())
    position = cam.unproject(u, v, depth)
    direction = cam.ray(u, v)
    reach = np.linalg.norm(position - cam.pos)  # how far along the line the nearest surface is
    best, best_err = None, None
    for cell in CELLS:
        t, gap = _closest(cam.pos, direction, cell)
        if gap > NEAR:
            continue
        err = abs(t - reach)  # several cells can lie along one line; depth says which
        if best_err is None or err < best_err:
            best, best_err = cell.key, err
    return best, position, (u, v)


def _closest(origin: np.ndarray, direction: np.ndarray, cell: Cell) -> tuple[float, float]:
    """(distance along the line, gap) where the line passes closest to the cell's upright axis."""
    a0 = np.array([*cell.xy, floor_height()])
    a1 = np.array([*cell.xy, TUBE_TOP])
    best = (0.0, np.inf)
    for s in np.linspace(0.0, 1.0, 13):  # the axis is short: checking 13 heights on it is plenty
        p = a0 + (a1 - a0) * s
        t = float((p - origin) @ direction)
        gap = float(np.linalg.norm(origin + t * direction - p))
        if gap < best[1]:
            best = (t, gap)
    return best


def detections(view: View, answer_text: str) -> list[Detection]:
    out = []
    for item in parse(answer_text):
        cell, position, pixel = locate(view, *item["point"])
        ray = (view.camera.pos, view.camera.ray(*pixel))
        out.append(Detection(view.camera.name, ray, item["kind"], item["colour"], pixel, cell, position))
    return out


def vote(per_camera: dict[str, list[Detection]]) -> dict[str, Found]:
    """Combine the pictures' answers cell by cell.

    A cell holds a tube if at least half the pictures saw one there (with one or two pictures: one is enough).
    Its kind is the most common answer, ties going to the camera that sees caps best; its colour likewise,
    ties going to the camera that sees liquid best.
    """
    n = len(per_camera)
    by_cell: dict[str, list[Detection]] = {}
    for dets in per_camera.values():
        seen = set()
        for d in dets:
            if d.cell and d.cell not in seen:  # one vote per picture per cell
                by_cell.setdefault(d.cell, []).append(d)
                seen.add(d.cell)
    found = {}
    for cell, dets in by_cell.items():
        if 2 * len(dets) < n:
            continue
        found[cell] = Found(
            _pick([d.kind for d in dets], [d.camera for d in dets], KIND_ORDER),
            _pick(*_colour_votes(dets), COLOUR_ORDER),
            len(dets),
            n,
            dets,
        )
    return found


def _colour_votes(dets: list[Detection]) -> tuple[list[str], list[str]]:
    """The colours to vote on, and their cameras. "none" means a picture could not see the liquid (a cap or
    the rack's plate hid it), so it counts only if no picture saw any colour."""
    seen = [d for d in dets if d.colour != "none"] or dets
    return [d.colour for d in seen], [d.camera for d in seen]


def _pick(values: list[str], cams: list[str], order: tuple[str, ...]) -> str:
    counts = Counter(values)
    top = max(counts.values())
    tied = {v for v, c in counts.items() if c == top}
    if len(tied) == 1:
        return tied.pop()
    for cam in order:
        for v, c in zip(values, cams, strict=True):
            if c == cam and v in tied:
                return v
    return sorted(tied)[0]


def find_separate(views: dict[str, View], ask) -> tuple[dict[str, Found], list, dict[str, list[Detection]]]:
    """One question per picture; our vote combines them. Returns the cells found, the answers, and every point
    each picture gave (for scoring positions)."""
    per_camera, answers = {}, []
    for name, view in views.items():
        a = ask([view.png], PROMPT_ONE)
        answers.append(a)
        per_camera[name] = detections(view, a.text)
    return vote(per_camera), answers, per_camera


def find_together(views: dict[str, View], ask) -> tuple[dict[str, Found], list, dict[str, list[Detection]]]:
    """All pictures in one question; Gemini combines them and points in the first picture."""
    names = list(views)
    a = ask([views[n].png for n in names], PROMPT_TOGETHER.replace("{n}", str(len(names))))
    per_camera = {names[0]: detections(views[names[0]], a.text)}
    return vote(per_camera), [a], per_camera
