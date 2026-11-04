"""The two learned pours: ACT, and a CNN + MLP policy with the same eyes to compare it with.

Both see the same things (act/data.py): the receiver strip (three pictures), the wide picture, the wrist
picture, Arm B's joints and the target level. Both give the same things: Arm B's joint targets for the
next 10 ticks (1 s), whether the pour has finished at each, and the level network's three numbers.

The eyes
    strip   Step 2's custom level network (step-2-pour/level/net.py), already trained to read the
            level, the flow and the balls in the air from exactly these pictures. It is kept as it is
            (frozen): the policy reads the receiver the way the tested level network does, and training
            is faster. Its convolutions give 12 rows (about 2 cm of tube each) x 128 features, and its
            own head still gives the three numbers. ``--tune-strip`` trains it on, gently, instead.
    wide    a small CNN of the same kind, trained from nothing: 96 x 128 -> 6 x 8 places x 128 features
    wrist   another: 72 x 96 -> 4 x 6 places x 128 features
    Both small ones start with a 5 x 5 convolution that halves the picture (as ResNet's first layer),
    then three of the level network's blocks.

ACT (Zhao et al., 2023, "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware")
    Every place in every picture becomes one token (12 strip rows + 48 wide + 24 wrist), and the joints, the
    target and a "style" value z one token each. A transformer encoder lets them all look at each other;
    a transformer decoder then asks 10 learned questions, one per future tick, and each answer becomes
    that tick's joint targets. While training, a second small encoder (the CVAE encoder) looks at the
    expert's actual actions and sums up their style in z; the main network learns to use it, and z is
    kept close to zero by a penalty (the KL term). When pouring, z is zero: the usual style.

CNN + MLP (behaviour cloning, as in robomimic's "BC")
    Each picture's features are boiled down to a few numbers (the strip's rows, as the level network
    does; for the others, where each feature is strongest: "spatial softmax", Levine et al., 2016),
    joined with the joints and target, and two dense layers give all 10 ticks at once.
"""

from __future__ import annotations

from pathlib import Path

import torch
from level.net import LevelNet, _block
from torch import nn

from .data import CHUNK

JOINTS = 6
LEVEL_RUN = Path(__file__).resolve().parents[2] / "step-2-pour" / "runs" / "custom" / "best.pt"


class StripEye(nn.Module):
    """Step 2's custom level network: 12 rows x 128 features of the receiver, and its three numbers.

    ``tune`` False (the default): frozen, always as trained (its batch norms too), and run without
    keeping what training would need, which saves time.
    """

    def __init__(self, pretrained: bool = True, tune: bool = False):
        super().__init__()
        self.net = LevelNet()
        if pretrained:
            self.net.load_state_dict(torch.load(LEVEL_RUN, map_location="cpu"))
        self.tune = tune or not pretrained
        if not self.tune:
            self.net.requires_grad_(False)

    def train(self, mode: bool = True) -> StripEye:
        super().train(mode)
        if not self.tune:
            self.net.eval()
        return self

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.set_grad_enabled(self.tune and torch.is_grad_enabled()):
            rows = self.net.features(x - 0.5).mean(dim=3)  # B, 128, 12: averaged across the tube's width
            return rows, self.net.head(rows)


class SmallEye(nn.Module):
    """A 5 x 5 convolution that halves the picture, then three blocks that each halve it again."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            _block(32, 64),
            _block(64, 96),
            _block(96, 128),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x - 0.5)


def _places(shape: tuple[int, int]) -> int:
    h, w = shape
    for _ in range(4):
        h, w = h // 2, w // 2
    return h * w


STRIP_PLACES, WIDE_PLACES, WRIST_PLACES = 12, _places((96, 128)), _places((72, 96))


class ACT(nn.Module):
    def __init__(
        self,
        d: int = 256,
        heads: int = 8,
        enc_layers: int = 4,
        dec_layers: int = 4,
        ff: int = 1024,
        dropout: float = 0.1,
        latent: int = 32,
        pretrained: bool = True,
        tune_strip: bool = False,
    ):
        super().__init__()
        self.latent = latent
        self.strip, self.wide, self.wrist = StripEye(pretrained, tune_strip), SmallEye(), SmallEye()
        self.strip_in, self.wide_in, self.wrist_in = nn.Linear(128, d), nn.Linear(128, d), nn.Linear(128, d)
        self.joints_in, self.target_in, self.z_in = nn.Linear(12, d), nn.Linear(1, d), nn.Linear(latent, d)
        n_tokens = 3 + STRIP_PLACES + WIDE_PLACES + WRIST_PLACES
        self.pos = nn.Parameter(torch.randn(n_tokens, d) * 0.02)  # where each token comes from

        def layer(kind):
            return kind(d, heads, ff, dropout, batch_first=True, norm_first=True)

        self.encoder = nn.TransformerEncoder(
            layer(nn.TransformerEncoderLayer), enc_layers, enable_nested_tensor=False
        )
        self.decoder = nn.TransformerDecoder(layer(nn.TransformerDecoderLayer), dec_layers)
        self.queries = nn.Parameter(torch.randn(CHUNK, d) * 0.02)  # one question per future tick
        self.out = nn.Linear(d, JOINTS + 1)  # joint targets, and "finished" (a logit)
        # The CVAE encoder: [summary token, joints, the expert's CHUNK actions] -> z's mean and spread.
        self.summary = nn.Parameter(torch.zeros(1, 1, d))
        self.style_joints_in, self.style_action_in = nn.Linear(12, d), nn.Linear(JOINTS, d)
        self.style_pos = nn.Parameter(torch.randn(2 + CHUNK, d) * 0.02)
        self.style = nn.TransformerEncoder(layer(nn.TransformerEncoderLayer), 2, enable_nested_tensor=False)
        self.to_z = nn.Linear(d, 2 * latent)

    def encode_style(self, joints: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = joints.shape[0]
        tokens = torch.cat(
            [
                self.summary.expand(b, -1, -1),
                self.style_joints_in(joints)[:, None],
                self.style_action_in(actions),
            ],
            1,
        )
        h = self.style(tokens + self.style_pos)[:, 0]
        mu, logvar = self.to_z(h).chunk(2, dim=-1)
        return mu, logvar

    def forward(
        self, b: dict[str, torch.Tensor], actions: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """``actions``: the expert's (scaled) actions while training, to find z; None when pouring (z = 0)."""
        n = b["joints"].shape[0]
        strip, extra = self.strip(b["strip"])
        if actions is not None:
            mu, logvar = self.encode_style(b["joints"], actions)
            z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        else:
            mu = logvar = None
            z = torch.zeros(n, self.latent, device=strip.device)

        def tokens(f, proj):  # B, C, H, W (or B, C, H) -> B, H*W, d
            return proj(f.flatten(2).transpose(1, 2))

        seq = torch.cat(
            [
                self.z_in(z)[:, None],
                self.joints_in(b["joints"])[:, None],
                self.target_in(b["target"])[:, None],
                tokens(strip, self.strip_in),
                tokens(self.wide(b["wide"]), self.wide_in),
                tokens(self.wrist(b["wrist"]), self.wrist_in),
            ],
            1,
        )
        memory = self.encoder(seq + self.pos)
        h = self.decoder(self.queries.expand(n, -1, -1), memory)
        out = self.out(h)
        return {
            "actions": out[..., :JOINTS],
            "done": out[..., JOINTS],
            "extra": extra,
            "mu": mu,
            "logvar": logvar,
        }


class SpatialSoftmax(nn.Module):
    """Where each feature is strongest: its (x, y), averaged over the picture weighted by softmax."""

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        b, c, h, w = f.shape
        weights = torch.softmax(f.flatten(2), dim=-1).view(b, c, h, w)
        ys = torch.linspace(-1, 1, h, device=f.device)
        xs = torch.linspace(-1, 1, w, device=f.device)
        x = (weights.sum(2) * xs).sum(-1)
        y = (weights.sum(3) * ys).sum(-1)
        return torch.cat([x, y], 1)  # B, 2C


class MLPPolicy(nn.Module):
    def __init__(
        self, hidden: int = 1024, dropout: float = 0.1, pretrained: bool = True, tune_strip: bool = False
    ):
        super().__init__()
        self.strip, self.wide, self.wrist = StripEye(pretrained, tune_strip), SmallEye(), SmallEye()
        self.keypoints = SpatialSoftmax()
        size = 128 * 12 + 2 * 128 + 2 * 128 + 12 + 1
        self.head = nn.Sequential(
            nn.Linear(size, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, CHUNK * (JOINTS + 1)),
        )

    def forward(
        self, b: dict[str, torch.Tensor], actions: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        strip, extra = self.strip(b["strip"])
        x = torch.cat(
            [
                strip.flatten(1),  # the strip's 12 rows, as the level network reads them
                self.keypoints(self.wide(b["wide"])),
                self.keypoints(self.wrist(b["wrist"])),
                b["joints"],
                b["target"],
            ],
            1,
        )
        out = self.head(x).view(-1, CHUNK, JOINTS + 1)
        return {
            "actions": out[..., :JOINTS],
            "done": out[..., JOINTS],
            "extra": extra,
            "mu": None,
            "logvar": None,
        }


MODELS = {"act": ACT, "mlp": MLPPolicy}


def make(name: str, **kwargs) -> nn.Module:
    return MODELS[name](**kwargs)


def loss(out: dict, b: dict, kl_weight: float = 10.0) -> tuple[torch.Tensor, dict[str, float]]:
    """L1 on the actions (as ACT), plus "finished", the three level numbers, and ACT's KL penalty.

    The level numbers only count when the strip's eye is being trained (they come from it alone)."""
    action = (out["actions"] - b["actions"]).abs().mean()
    done = nn.functional.binary_cross_entropy_with_logits(out["done"], b["done"])
    extra = nn.functional.smooth_l1_loss(out["extra"], b["extra"])
    total = action + 0.1 * done + (0.5 * extra if out["extra"].requires_grad else 0.0)
    parts = {"action": action.item(), "done": done.item(), "extra": extra.item()}
    if out["mu"] is not None:
        mu, logvar = out["mu"], out["logvar"]
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()
        total = total + kl_weight * kl
        parts["kl"] = kl.item()
    return total, parts
