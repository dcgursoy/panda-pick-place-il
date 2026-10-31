"""Behavior cloning: the observation encoder, an MLP, and an action chunk regressed with MSE.

With ``pred_horizon = 16`` this is the chunked BC that compares fairly with
Diffusion Policy (same encoder, same windows, same execution of the first
``action_horizon`` actions); with ``pred_horizon = 1`` it is the single-step ablation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from panda_il.policies.normalize import MinMaxNormalizer
from panda_il.policies.obs_encoder import ObsEncoderConfig, ObservationEncoder


@dataclass
class BCConfig:
    encoder: ObsEncoderConfig = field(default_factory=ObsEncoderConfig)
    action_dim: int = 7
    pred_horizon: int = 16
    action_horizon: int = 8
    hidden: int = 512
    layers: int = 3


class BCPolicy(nn.Module):
    def __init__(self, cfg: BCConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = ObservationEncoder(cfg.encoder)
        self.action_normalizer = MinMaxNormalizer(cfg.action_dim)
        dims = [self.encoder.output_dim] + [cfg.hidden] * cfg.layers
        mlp: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:], strict=True):
            mlp += [nn.Linear(a, b), nn.ReLU()]
        mlp.append(nn.Linear(dims[-1], cfg.pred_horizon * cfg.action_dim))
        self.head = nn.Sequential(*mlp)

    def fit_normalizers(self, lowdim, actions) -> None:
        self.encoder.lowdim_normalizer.fit(lowdim)
        self.action_normalizer.fit(actions)

    def forward(self, batch: dict[str, Tensor], train: bool) -> Tensor:
        """Normalised action chunk (B, Tp, 7)."""
        features = self.encoder(batch, train=train)
        return self.head(features).reshape(-1, self.cfg.pred_horizon, self.cfg.action_dim)

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        target = self.action_normalizer(batch["action"][:, : self.cfg.pred_horizon])
        return nn.functional.mse_loss(self(batch, train=True).float(), target)

    @torch.no_grad()
    def predict(self, batch: dict[str, Tensor]) -> Tensor:
        """Unnormalised action chunk (B, Tp, 7) for an observation window."""
        return self.action_normalizer.unnormalize(self(batch, train=False).float())
