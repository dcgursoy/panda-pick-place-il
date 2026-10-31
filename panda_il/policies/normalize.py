"""Min-max normalisation to [-1, 1] per dimension, as Diffusion Policy does.

Dimensions whose range is below ``range_eps`` (the expert never rolls or pitches
the gripper, for example) are mapped to 0 instead of being blown up. The
statistics are fit on exactly the episodes a policy trains on and saved inside
its checkpoint.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn


class MinMaxNormalizer(nn.Module):
    def __init__(self, dim: int, range_eps: float = 1e-4):
        super().__init__()
        self.range_eps = range_eps
        self.register_buffer("scale", torch.ones(dim))
        self.register_buffer("offset", torch.zeros(dim))
        self.register_buffer("fitted", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def fit(self, data: np.ndarray | Tensor) -> MinMaxNormalizer:
        data = torch.as_tensor(np.asarray(data), dtype=torch.float32).reshape(
            -1, self.scale.numel()
        )
        low = data.min(dim=0).values
        high = data.max(dim=0).values
        span = high - low
        constant = span < self.range_eps
        scale = torch.where(constant, torch.zeros_like(span), 2.0 / span.clamp_min(self.range_eps))
        offset = torch.where(constant, torch.zeros_like(span), -1.0 - low * scale)
        self.scale.copy_(scale)
        self.offset.copy_(offset)
        self.fitted.fill_(True)
        return self

    def normalize(self, x: Tensor) -> Tensor:
        return x * self.scale + self.offset

    def unnormalize(self, x: Tensor) -> Tensor:
        safe_scale = torch.where(self.scale == 0, torch.ones_like(self.scale), self.scale)
        return torch.where(self.scale == 0, torch.zeros_like(x), (x - self.offset) / safe_scale)

    def forward(self, x: Tensor) -> Tensor:
        return self.normalize(x)


def images_to_float(images: Tensor) -> Tensor:
    """(..., H, W, 3) uint8 -> (..., 3, H, W) float32 in [-1, 1]."""
    x = images.to(torch.float32).div_(127.5).sub_(1.0)
    return x.movedim(-1, -3)
