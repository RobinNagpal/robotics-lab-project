"""The level networks: a small CNN trained from scratch, and a pretrained ResNet-18 to compare it with.

Both take three stacked pictures (9 channels, 192 x 48) and give three numbers: level, flow and balls in
the air (level.data). Neither pools the picture down to one value per channel before the end: the height
in the picture is the level, so the last layers must still know where things are.
"""

from __future__ import annotations

import torch
from torch import nn

from .data import FRAMES, OUTPUTS


def _block(cin: int, cout: int) -> nn.Sequential:
    """Two 3 x 3 convolutions, then halve the picture."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class LevelNet(nn.Module):
    """Four blocks take 192 x 48 down to 12 x 3; the narrow width is then averaged away, keeping the
    12 rows (each about 2 cm of tube), and two layers turn those into the three numbers."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(_block(3 * FRAMES, 32), _block(32, 64), _block(64, 96), _block(96, 128))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 12, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, len(OUTPUTS)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x - 0.5)
        return self.head(f.mean(dim=3))  # average across the tube's width, keep its height


class ResNetLevel(nn.Module):
    """ResNet-18 pretrained on ImageNet photos. Its first layer takes one colour picture (3 channels);
    here it takes three, so those weights are copied to each and divided by three. Its usual end, an
    average over the whole picture, is replaced like LevelNet's: averaged across the width only."""

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        w = base.conv1.weight.data
        base.conv1 = nn.Conv2d(3 * FRAMES, 64, 7, stride=2, padding=3, bias=False)
        base.conv1.weight.data = w.repeat(1, FRAMES, 1, 1) / FRAMES
        # Keep the picture taller: without the stride in the last stage, 192 x 48 ends at 12 x 3.
        base.layer4[0].conv1.stride = (1, 1)
        base.layer4[0].downsample[0].stride = (1, 1)
        self.body = nn.Sequential(*list(base.children())[:-2])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 12, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, len(OUTPUTS)),
        )
        mean = torch.tensor(self.MEAN * FRAMES).view(1, -1, 1, 1)
        std = torch.tensor(self.STD * FRAMES).view(1, -1, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.body((x - self.mean) / self.std)
        return self.head(f.mean(dim=3))


def make(name: str) -> nn.Module:
    return {"custom": LevelNet, "resnet18": ResNetLevel}[name]()
