"""Our own detector, built for this setup and trained from scratch: the custom CNN.

It finds objects bottom-up, as Panoptic-DeepLab and CenterNet do. For every
pixel of the picture it predicts:

- whether it is part of a rack footprint, and whether it is part of a tube
  (two separate channels: a rack's outline covers the tubes standing in it),
- how likely it is to be the centre of a rack or a tube (one heatmap each),
- for each class, the offset from the pixel to the centre of the object it
  belongs to.

An object is then a centre peak plus every pixel that points at it. Tubes in
touching cells stand 36 mm apart, about 15 output pixels: their centres never
merge, even where their outlines touch. That is the case YOLO gets wrong.

What it assumes about the setup, which a general detector cannot:
- the camera is fixed, so an object is always about the same size in pixels:
  no scale augmentation, and one size of heatmap peak per class;
- tubes are small, so it reads the full 1280 x 960 picture and predicts at
  half of that, where a tube is about 11 pixels wide (YOLO at imgsz 640
  predicts masks at a quarter of that).

The labels are the same outlines YOLO trains on (data/<split>/labels).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

# OpenCV must load before PyTorch: the other way round, two OpenMP runtimes clash.
import cv2  # noqa: I001
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .classes import CLASSES

TUBE, RACK = CLASSES.index("test_tube"), CLASSES.index("rack")
IMAGE = (1280, 960)  # width, height of the rendered pictures
STRIDE = 2  # the output is half the picture's size
OUT = (IMAGE[0] // STRIDE, IMAGE[1] // STRIDE)
# Heatmap peak size (Gaussian sigma, output pixels) and the farthest a pixel's
# vote may land from a centre and still join it. A tube is about 11 output
# pixels across, a rack about 90 by 40.
SIGMA = {TUBE: 2.0, RACK: 6.0}
MAX_VOTE = {TUBE: 10.0, RACK: 60.0}
# Offsets are learned in these units, so both classes' targets are about 1.
OFFSET_SCALE = {TUBE: 8.0, RACK: 40.0}
# Window for finding peaks: a peak must be the highest point within it.
PEAK_WINDOW = {TUBE: 5, RACK: 15}
MIN_PIXELS = {TUBE: 15, RACK: 300}  # smaller instances are dropped, in output pixels


# --- Targets -------------------------------------------------------------------------


def read_labels(path: Path) -> list[tuple[int, np.ndarray]]:
    """The label file's outlines: (class, points as x, y from 0 to 1)."""
    out = []
    for line in path.read_text().splitlines():
        v = line.split()
        if v:
            out.append((int(v[0]), np.array(v[1:], dtype=np.float32).reshape(-1, 2)))
    return out


def instance_maps(
    labels: list[tuple[int, np.ndarray]],
    flip_x: bool,
    flip_y: bool,
    window: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """One map of instance numbers per class at output size (0 = none), shape (2, H, W).

    ``window`` is (left, top, width, height) of a crop in picture pixels, after
    the flips; None is the whole picture.
    """
    x0, y0, w, h = window or (0, 0, *IMAGE)
    maps = np.zeros((len(CLASSES), h // STRIDE, w // STRIDE), np.int32)
    areas = []
    for k, (cls, pts) in enumerate(labels):
        p = pts.copy()
        if flip_x:
            p[:, 0] = 1 - p[:, 0]
        if flip_y:
            p[:, 1] = 1 - p[:, 1]
        p = (p * np.array(IMAGE) - np.array([x0, y0])) / STRIDE
        poly = np.round(p - 0.5).astype(np.int32)
        areas.append((cv2.contourArea(poly.astype(np.float32)), k, cls, poly))
    # Biggest first, so where two outlines overlap the smaller one is on top.
    for _, k, cls, poly in sorted(areas, key=lambda a: -a[0]):
        cv2.fillPoly(maps[cls], [poly], k + 1)
    return maps


def targets(maps: np.ndarray) -> dict[str, np.ndarray]:
    """Mask, centre heatmap and centre offsets for each class, from the instance maps."""
    h, w = maps.shape[1:]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    mask = (maps > 0).astype(np.float32)
    heat = np.zeros((len(CLASSES), h, w), np.float32)
    offset = np.zeros((2 * len(CLASSES), h, w), np.float32)
    for cls in range(len(CLASSES)):
        for k in np.unique(maps[cls]):
            if k == 0:
                continue
            inside = maps[cls] == k
            cy, cx = ys[inside].mean(), xs[inside].mean()
            s = SIGMA[cls]
            # Only near the centre, for speed: the Gaussian is ~0 beyond 4 sigma.
            r = int(4 * s) + 1
            y0, y1 = max(0, int(cy) - r), min(h, int(cy) + r + 2)
            x0, x1 = max(0, int(cx) - r), min(w, int(cx) + r + 2)
            g = np.exp(-((xs[y0:y1, x0:x1] - cx) ** 2 + (ys[y0:y1, x0:x1] - cy) ** 2) / (2 * s * s))
            heat[cls, y0:y1, x0:x1] = np.maximum(heat[cls, y0:y1, x0:x1], g)
            offset[2 * cls][inside] = (cx - xs[inside]) / OFFSET_SCALE[cls]
            offset[2 * cls + 1][inside] = (cy - ys[inside]) / OFFSET_SCALE[cls]
    return {"mask": mask, "heat": heat, "offset": offset}


# Training crop, picture pixels: a quarter of the picture, so a step costs a
# quarter as much. The net is fully convolutional and objects are always the
# same size, so it then runs on whole pictures unchanged. Multiples of 32.
CROP = (640, 480)
# How far a crop's middle may be from the rack it is placed around, picture pixels.
CROP_JITTER = 200


class Pictures(torch.utils.data.Dataset):
    """The rendered pictures of one split, with targets made from their label outlines.

    With ``augment``, each picture is flipped at random and cut to a crop
    around one of its racks (where the tubes are).
    """

    def __init__(self, root: Path, augment: bool):
        self.images = sorted((root / "images").glob("*.jpg"), key=lambda p: int(p.stem))
        self.labels = root / "labels"
        self.augment = augment

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        image = cv2.cvtColor(cv2.imread(str(self.images[i])), cv2.COLOR_BGR2RGB)
        labels = read_labels(self.labels / f"{self.images[i].stem}.txt")
        flip_x = flip_y = False
        window = None
        if self.augment:
            # A straight-down camera has no up or down: both flips are real pictures.
            flip_x, flip_y = bool(np.random.rand() < 0.5), bool(np.random.rand() < 0.5)
            if flip_x:
                image = image[:, ::-1]
            if flip_y:
                image = image[::-1]
            racks = [p for c, p in labels if c == RACK]
            middle = racks[np.random.randint(len(racks))].mean(0) if racks else np.array([0.5, 0.5])
            middle = middle * np.array(IMAGE)
            if flip_x:
                middle[0] = IMAGE[0] - middle[0]
            if flip_y:
                middle[1] = IMAGE[1] - middle[1]
            middle = middle + np.random.uniform(-CROP_JITTER, CROP_JITTER, 2)
            x0 = int(np.clip(middle[0] - CROP[0] / 2, 0, IMAGE[0] - CROP[0])) // STRIDE * STRIDE
            y0 = int(np.clip(middle[1] - CROP[1] / 2, 0, IMAGE[1] - CROP[1])) // STRIDE * STRIDE
            window = (x0, y0, *CROP)
            image = image[y0 : y0 + CROP[1], x0 : x0 + CROP[0]]
        maps = instance_maps(labels, flip_x, flip_y, window)
        t = targets(maps)
        x = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
        return x, {k: torch.from_numpy(v) for k, v in t.items()}, torch.from_numpy(maps)


def colour_jitter(x: torch.Tensor) -> torch.Tensor:
    """Strong random colour changes per picture, as YOLO's hsv settings: learn shapes, not colours.

    ``x`` is a batch of 0..1 RGB pictures on the training device.
    """
    b = x.shape[0]
    dev = x.device

    def rand(lo, hi):
        return torch.empty(b, 1, 1, 1, device=dev).uniform_(lo, hi)

    # Hue: rotate the colour around the grey axis. Saturation: blend with grey.
    angle = rand(-math.pi, math.pi)
    grey = x.mean(1, keepdim=True)
    k = torch.tensor([1.0, 1.0, 1.0], device=dev).view(1, 3, 1, 1) / math.sqrt(3)
    v = x - grey
    cross = torch.cross(k.expand_as(v), v, dim=1)
    v = v * torch.cos(angle) + cross * torch.sin(angle)
    x = grey + v * rand(0.3, 1.7)
    # Brightness and contrast.
    x = (x - 0.5) * rand(0.7, 1.3) + 0.5 + rand(-0.25, 0.25)
    return x.clamp(0, 1)


# --- Network ---------------------------------------------------------------------------


def _conv(cin: int, cout: int, stride: int = 1, dilation: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, padding=dilation, dilation=dilation, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class Block(nn.Module):
    """Two 3x3 convolutions with a shortcut."""

    def __init__(self, cin: int, cout: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        self.a = _conv(cin, cout, stride)
        self.b = nn.Sequential(
            nn.Conv2d(cout, cout, 3, padding=dilation, dilation=dilation, bias=False), nn.BatchNorm2d(cout)
        )
        self.short = (
            nn.Identity()
            if stride == 1 and cin == cout
            else nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
        )

    def forward(self, x):
        return F.relu(self.b(self.a(x)) + self.short(x))


@dataclass(frozen=True)
class NetConfig:
    in_channels: int = 3
    # Channels at strides 2, 4, 8, 16 and 32 of the picture.
    widths: tuple[int, ...] = (32, 48, 96, 160, 256)
    head: int = 32


class CustomNet(nn.Module):
    """A small U-Net: an encoder down to 1/32 of the picture, a decoder back to 1/2.

    The deepest stage uses dilated convolutions, so a pixel at one end of a
    20 cm rack can still see the rack's middle and point at it.
    """

    def __init__(self, cfg: NetConfig | None = None):
        super().__init__()
        cfg = cfg or NetConfig()
        w = cfg.widths
        self.cfg = cfg
        self.stem = nn.Sequential(_conv(cfg.in_channels, w[0], 2), Block(w[0], w[0]))
        self.down = nn.ModuleList(
            [nn.Sequential(Block(w[i], w[i + 1], 2), Block(w[i + 1], w[i + 1])) for i in range(len(w) - 1)]
        )
        self.context = nn.Sequential(Block(w[-1], w[-1], dilation=2), Block(w[-1], w[-1], dilation=4))
        self.up = nn.ModuleList([Block(w[i + 1] + w[i], w[i]) for i in reversed(range(len(w) - 1))])

        def head(out: int) -> nn.Sequential:
            return nn.Sequential(_conv(w[0], cfg.head), nn.Conv2d(cfg.head, out, 1))

        self.mask = head(len(CLASSES))
        self.heat = head(len(CLASSES))
        self.offset = head(2 * len(CLASSES))
        # Start the heatmap near 0 everywhere, as CenterNet does, so early training is stable.
        nn.init.constant_(self.heat[-1].bias, -4.6)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        skips = [self.stem(x)]
        for d in self.down:
            skips.append(d(skips[-1]))
        y = self.context(skips.pop())
        for up in self.up:
            skip = skips.pop()
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = up(torch.cat([y, skip], 1))
        return {"mask": self.mask(y), "heat": self.heat(y), "offset": self.offset(y)}


def normalise(x: torch.Tensor) -> torch.Tensor:
    """0..1 RGB to roughly zero mean, unit spread."""
    return (x - 0.5) / 0.25


# --- Losses ----------------------------------------------------------------------------


def focal_heat_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """CenterNet's penalty-reduced focal loss, per peak."""
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    # The exact peak may fall between pixels; the highest pixel of each Gaussian counts.
    pooled = F.max_pool2d(target, 3, 1, 1)
    peak = ((target == pooled) & (target > 0.6)).float()
    pos = -torch.log(p) * (1 - p) ** 2 * peak
    neg = -torch.log(1 - p) * p**2 * (1 - target) ** 4 * (1 - peak)
    return (pos.sum() + neg.sum()) / peak.sum().clamp(min=1)


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits)
    inter = (p * target).sum((2, 3))
    return (1 - (2 * inter + 1) / (p.sum((2, 3)) + target.sum((2, 3)) + 1)).mean()


def loss(out: dict[str, torch.Tensor], t: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    mask = F.binary_cross_entropy_with_logits(out["mask"], t["mask"]) + dice_loss(out["mask"], t["mask"])
    heat = focal_heat_loss(out["heat"], t["heat"])
    # Offsets count only on the pixels of an object of that class.
    weight = t["mask"].repeat_interleave(2, dim=1)
    offset = (
        F.smooth_l1_loss(out["offset"], t["offset"], reduction="none") * weight
    ).sum() / weight.sum().clamp(min=1)
    return {"mask": mask, "heat": heat, "offset": offset, "total": mask + heat + offset}


# --- From the network's output to objects --------------------------------------------


@torch.no_grad()
def decode(out: dict[str, torch.Tensor], threshold: float = 0.3) -> list[tuple[int, float, np.ndarray]]:
    """One picture's output (batch of 1) to objects: (class, score, mask at output size)."""
    mask = torch.sigmoid(out["mask"][0])
    heat = torch.sigmoid(out["heat"][0])
    offset = out["offset"][0]
    h, w = mask.shape[1:]
    ys, xs = torch.meshgrid(
        torch.arange(h, device=mask.device, dtype=torch.float32),
        torch.arange(w, device=mask.device, dtype=torch.float32),
        indexing="ij",
    )
    found = []
    for cls in range(len(CLASSES)):
        hm = heat[cls]
        win = PEAK_WINDOW[cls]
        peaks = (hm == F.max_pool2d(hm[None, None], win, 1, win // 2)[0, 0]) & (hm > threshold)
        py, px = torch.nonzero(peaks, as_tuple=True)
        if len(py) == 0:
            continue
        scores = hm[py, px]
        centres = torch.stack([px.float(), py.float()], 1)
        fg = mask[cls] > 0.5
        fy, fx = torch.nonzero(fg, as_tuple=True)
        if len(fy) == 0:
            continue
        votes = torch.stack(
            [
                fx.float() + offset[2 * cls][fy, fx] * OFFSET_SCALE[cls],
                fy.float() + offset[2 * cls + 1][fy, fx] * OFFSET_SCALE[cls],
            ],
            1,
        )
        dist = torch.cdist(votes, centres)
        best, owner = dist.min(1)
        keep = best < MAX_VOTE[cls]
        owner = torch.where(keep, owner, torch.full_like(owner, -1))
        label = torch.full((h, w), -1, dtype=torch.long, device=mask.device)
        label[fy, fx] = owner
        label = label.cpu().numpy()
        for k in range(len(centres)):
            m = label == k
            if m.sum() >= MIN_PIXELS[cls]:
                found.append((cls, float(scores[k]), m))
    return found


def to_picture_size(m: np.ndarray) -> np.ndarray:
    """An output-size mask scaled up to the picture's size."""
    return cv2.resize(m.astype(np.uint8), IMAGE, interpolation=cv2.INTER_LINEAR).astype(bool)


class Detector:
    """Loads trained weights and finds objects in a picture, as (class name, full-size mask)."""

    def __init__(self, weights: Path, device: str = "mps", threshold: float = 0.3):
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        self.net = CustomNet(NetConfig(**ckpt["config"])).to(device).eval()
        self.net.load_state_dict(ckpt["model"])
        self.device = device
        self.threshold = threshold

    @torch.no_grad()
    def __call__(self, image: Path) -> list[tuple[str, np.ndarray]]:
        rgb = cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().div(255).to(self.device)
        out = self.net(normalise(x))
        return [(CLASSES[c], to_picture_size(m)) for c, _, m in decode(out, self.threshold)]
