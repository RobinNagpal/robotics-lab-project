"""ACT for Arm B's unscrewing, from sensor values (no camera).

ACT (Zhao et al., 2023, "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"), as in v1's
Step 2b, with the pictures taken out:

- Each of the HISTORY input moments (learn/data.py: joints, speeds, gripper, wrist force and torque,
  motor torques) becomes one token, and a "style" value z one more. A transformer encoder lets them look
  at each other: the history is what lets it see, for example, the pull during a lift test.
- A transformer decoder asks CHUNK learned questions, one per future sample; each answer becomes that
  sample's 6 joint targets, gripper command, and "done".
- While training, a small second encoder (the CVAE encoder) looks at the teacher's actual next actions
  and sums up their style in z; z is kept near zero by a penalty (the KL term). When running, z is zero.
"""

from __future__ import annotations

import torch
from torch import nn

from .data import ACTIONS, CHUNK, HISTORY, OBS


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
    ):
        super().__init__()
        self.latent = latent
        self.obs_in = nn.Linear(OBS, d)
        self.z_in = nn.Linear(latent, d)
        self.pos = nn.Parameter(torch.randn(1 + HISTORY, d) * 0.02)

        def layer(kind):
            return kind(d, heads, ff, dropout, batch_first=True, norm_first=True)

        self.encoder = nn.TransformerEncoder(
            layer(nn.TransformerEncoderLayer), enc_layers, enable_nested_tensor=False
        )
        self.decoder = nn.TransformerDecoder(layer(nn.TransformerDecoderLayer), dec_layers)
        self.queries = nn.Parameter(torch.randn(CHUNK, d) * 0.02)
        self.out = nn.Linear(d, ACTIONS + 1)
        # The CVAE encoder: [summary, the newest input, the teacher's CHUNK actions] -> z's mean and spread.
        self.summary = nn.Parameter(torch.zeros(1, 1, d))
        self.style_obs_in, self.style_action_in = nn.Linear(OBS, d), nn.Linear(ACTIONS, d)
        self.style_pos = nn.Parameter(torch.randn(2 + CHUNK, d) * 0.02)
        self.style = nn.TransformerEncoder(layer(nn.TransformerEncoderLayer), 2, enable_nested_tensor=False)
        self.to_z = nn.Linear(d, 2 * latent)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor | None = None) -> dict:
        """``obs``: B, HISTORY, OBS (scaled). ``actions``: the teacher's (scaled) while training."""
        n = obs.shape[0]
        if actions is not None:
            tokens = torch.cat(
                [
                    self.summary.expand(n, -1, -1),
                    self.style_obs_in(obs[:, -1])[:, None],
                    self.style_action_in(actions),
                ],
                1,
            )
            mu, logvar = self.to_z(self.style(tokens + self.style_pos)[:, 0]).chunk(2, dim=-1)
            z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        else:
            mu = logvar = None
            z = torch.zeros(n, self.latent, device=obs.device)
        seq = torch.cat([self.z_in(z)[:, None], self.obs_in(obs)], 1) + self.pos
        h = self.decoder(self.queries.expand(n, -1, -1), self.encoder(seq))
        out = self.out(h)
        return {"actions": out[..., :ACTIONS], "done": out[..., ACTIONS], "mu": mu, "logvar": logvar}


def loss(out: dict, b: dict, kl_weight: float = 10.0) -> tuple[torch.Tensor, dict[str, float]]:
    """L1 on the actions (as ACT), "done" (cross-entropy), and ACT's KL penalty."""
    action = (out["actions"] - b["actions"]).abs().mean()
    done = nn.functional.binary_cross_entropy_with_logits(out["done"], b["done"])
    total = action + 0.5 * done
    parts = {"action": action.item(), "done": done.item()}
    if out["mu"] is not None:
        mu, logvar = out["mu"], out["logvar"]
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()
        total = total + kl_weight * kl
        parts["kl"] = kl.item()
    return total, parts
