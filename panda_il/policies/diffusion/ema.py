"""Exponential moving average of the weights, with Diffusion Policy's warm-up schedule:
decay = clamp(1 - (1 + step / inv_gamma) ** -power, 0, max_decay), updated after every batch.
Parameters are averaged; buffers (normaliser statistics, GroupNorm has none) are copied."""

from __future__ import annotations

import copy

import torch
from torch import nn


class EMAModel:
    def __init__(
        self,
        model: nn.Module,
        power: float = 0.75,
        inv_gamma: float = 1.0,
        max_decay: float = 0.9999,
        update_after_step: int = 0,
    ):
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.power = power
        self.inv_gamma = inv_gamma
        self.max_decay = max_decay
        self.update_after_step = update_after_step
        self.step_count = 0

    def decay(self) -> float:
        step = max(0, self.step_count - self.update_after_step - 1)
        if step <= 0:
            return 0.0
        return float(min(max(1 - (1 + step / self.inv_gamma) ** -self.power, 0.0), self.max_decay))

    @torch.no_grad()
    def step(self, model: nn.Module) -> float:
        self.step_count += 1
        decay = self.decay()
        for ema_param, param in zip(self.model.parameters(), model.parameters(), strict=True):
            if param.requires_grad:
                ema_param.mul_(decay).add_(param.detach(), alpha=1 - decay)
            else:
                ema_param.copy_(param)
        for ema_buffer, buffer in zip(self.model.buffers(), model.buffers(), strict=True):
            ema_buffer.copy_(buffer)
        return decay
