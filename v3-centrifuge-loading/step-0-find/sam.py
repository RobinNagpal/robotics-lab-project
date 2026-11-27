"""Step 0 with Meta's SAM 3: outlines from a word, then our own rules for what each tube is.

SAM 3 answers "where is every <word>?" with an outline (a mask) per object, and nothing else. So:

1. SAM 3 outlines every "tube", and separately every "orange cap".
2. A tube with a cap outline on it is a **sample**.
3. The colour inside the tube's outline (leaving out the cap) is its liquid: the most common colour among the
   strongly coloured pixels. An open tube mostly blue inside is **water**; otherwise it is **empty**.
4. The top of each tube's outline becomes a point, and from there it is the same as with Gemini: the same cell
   matching, the same vote over the cameras, the same scoring (find.py).

Runs through Hugging Face Transformers, which needs no NVIDIA GPU: on this Mac it uses the Apple GPU (MPS) if
it can, else the CPU. The model (facebook/sam3) is gated: the Hugging Face token is read from HF_TOKEN or from
v3-centrifuge-loading/.env, like the Gemini key.

    pixi run --manifest-path ../simulation-environment/pixi.toml -e sam python evaluate.py --model sam3
"""

from __future__ import annotations

import colorsys
import hashlib
import io
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

import gemini

MODEL = "facebook/sam3"
# "tube" alone gets low scores on these clear tubes; "test tube" gave the right number of tubes in 8 of 12
# pictures of seeds 100-103 (never tested on), tied best with "vial" (which fails from overhead) and
# "centrifuge tube".
TUBE_WORD = "test tube"
CAP_WORD = "orange cap"
SCORE = 0.5  # outlines SAM 3 is less sure of than this are dropped
RAW = Path(__file__).resolve().parent / "results" / "sam_raw"

# Hue ranges (degrees) for the liquid colours. Only pixels with this much saturation and brightness count.
HUES = (
    ("red", 0, 15),
    ("orange", 15, 38),
    ("yellow", 38, 70),
    ("green", 70, 185),
    ("blue", 185, 250),
    ("purple", 250, 335),
    ("red", 335, 360),
)
MIN_SATURATION, MIN_VALUE = 0.35, 0.20
# A sample's liquid counts as seen with at least this many coloured pixels that are not blue. Seen through the
# clear plastic its colour is paler, so a lower saturation counts: 0.25 got 101 of 102 sample colours right on
# seeds 100-111 (0.35 got 50).
MIN_SAMPLE_PIXELS = 20
SAMPLE_SATURATION = 0.25
# Share of an open tube's outline that is strongly blue, above which it is water. Halfway between the water
# tube and the bluest empty tube, measured on seeds 100-111 (never tested on): overhead 0.70 vs 0.41, high
# 0.38 vs 0.14, low 0.14 vs 0.02.
WATER_SHARE = {"overhead": 0.55, "racks_high": 0.26, "racks_low": 0.08}


def hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    env = gemini.V3 / ".env"
    if not token and env.exists():
        for line in env.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "HF_TOKEN" and value.strip():
                token = value.strip().strip('"').strip("'")
    return token


def strong_hues(
    rgb: np.ndarray, mask: np.ndarray, min_saturation: float = MIN_SATURATION
) -> tuple[np.ndarray, int]:
    """The hues (degrees) of the strongly coloured pixels inside ``mask``, and how many pixels it has."""
    pixels = rgb[mask].astype(float) / 255
    if len(pixels) == 0:
        return np.zeros(0), 0
    hsv = np.array([colorsys.rgb_to_hsv(*p) for p in pixels])
    strong = (hsv[:, 1] >= min_saturation) & (hsv[:, 2] >= MIN_VALUE)
    return hsv[strong, 0] * 360, len(pixels)


def name_of(hue: float) -> str:
    return next(name for name, lo, hi in HUES if lo <= hue < hi)


def describe(rgb: np.ndarray, tubes: list[np.ndarray], caps: list[np.ndarray], camera: str) -> list[dict]:
    """Our rules: tube and cap outlines to [{point, kind, colour}], in Gemini's format (0-1000, [y, x]).

    The tubes are clear, so the blue rack shows through them: blue alone does not mean water. So a sample's
    colour is its most common colour other than blue (no sample is blue), and an open tube is water only if
    blue covers more of its outline than the rack can (WATER_SHARE, set on seeds the test never uses).
    """
    h, w = rgb.shape[:2]
    cap_any = np.zeros((h, w), bool)
    for c in caps:
        cap_any |= c
    biggest_cap = max((c.sum() for c in caps), default=1)
    items = []
    for mask in tubes:
        if mask.sum() < 30:
            continue
        rows = np.where(mask.any(axis=1))[0]
        top = rows[: max(1, len(rows) // 6)]  # the top sixth of the outline: the cap or the rim
        ys, xs = np.nonzero(mask[top[0] : top[-1] + 1])
        y, x = ys.mean() + top[0], xs.mean()
        hues, n = strong_hues(rgb, mask & ~cap_any)
        blue = (hues >= 185) & (hues < 250)
        if (mask & cap_any).sum() > 0.25 * min(mask.sum(), biggest_cap):
            kind = "sample"
            sample_hues, _ = strong_hues(rgb, mask & ~cap_any, SAMPLE_SATURATION)
            other = sample_hues[~((sample_hues >= 185) & (sample_hues < 250))]
            if len(other) >= MIN_SAMPLE_PIXELS:
                names = [name_of(v) for v in other]
                colour = max(set(names), key=names.count)
            else:
                colour = "none"  # its liquid is hidden here (under the cap, or below the rack's plate)
        elif n and blue.sum() / n >= WATER_SHARE[camera]:
            kind, colour = "water", "blue"
        else:
            kind, colour = "empty", "none"
        items.append({"point": [y / h * 1000, x / w * 1000], "kind": kind, "colour": colour})
    return items


class Sam3:
    """SAM 3, loaded once. Outlines are saved per picture, so asking again costs no time."""

    def __init__(self):
        self._model = None
        self.camera_of: dict[bytes, str] = {}  # which camera took each picture (set before asking)

    def _load(self):
        import torch
        from transformers import Sam3Model, Sam3Processor

        self.torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        token = hf_token()
        self.processor = Sam3Processor.from_pretrained(MODEL, token=token)
        self._model = Sam3Model.from_pretrained(MODEL, token=token).to(self.device).eval()

    def outlines(self, png: bytes, word: str) -> tuple[list[np.ndarray], float]:
        key = hashlib.sha256(png + word.encode() + MODEL.encode()).hexdigest()[:24]
        path = RAW / f"{key}.npz"
        if path.exists():
            saved = np.load(path)
            return list(saved["masks"]), float(saved["seconds"])
        if self._model is None:
            self._load()
        image = Image.open(io.BytesIO(png)).convert("RGB")
        t0 = time.perf_counter()
        inputs = self.processor(images=image, text=word, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self._model(**inputs)
        res = self.processor.post_process_instance_segmentation(
            out, threshold=SCORE, mask_threshold=0.5, target_sizes=inputs.get("original_sizes").tolist()
        )[0]
        seconds = time.perf_counter() - t0
        masks = [m.cpu().numpy().astype(bool) for m in res["masks"]]
        RAW.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, masks=np.array(masks).reshape(-1, image.height, image.width), seconds=seconds
        )
        return masks, seconds

    def __call__(self, pngs: list[bytes], prompt: str) -> gemini.Answer:
        """Answers find.py's question about one picture, in Gemini's format, so the rest is shared."""
        png = pngs[0]
        rgb = np.array(Image.open(io.BytesIO(png)).convert("RGB"))
        tubes, t1 = self.outlines(png, TUBE_WORD)
        caps, t2 = self.outlines(png, CAP_WORD)
        return gemini.Answer(json.dumps(describe(rgb, tubes, caps, self.camera_of[png])), t1 + t2, 0, 0)
