"""DDPM (training and reference sampling) and DDIM (fast sampling), as plain tensors.

Follows the configuration Diffusion Policy uses through diffusers: 100 training
timesteps, the squared-cosine ("squaredcos_cap_v2") beta schedule, epsilon
prediction, samples clipped to [-1, 1], the "fixed_small" posterior variance for
DDPM, and eta = 0 for DDIM with "leading" timestep spacing. The test suite
checks these against diffusers' implementation.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def squaredcos_cap_v2_betas(num_steps: int, max_beta: float = 0.999) -> Tensor:
    def alpha_bar(t: float) -> float:
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = [
        min(1 - alpha_bar((i + 1) / num_steps) / alpha_bar(i / num_steps), max_beta)
        for i in range(num_steps)
    ]
    return torch.tensor(betas, dtype=torch.float32)


NoiseGenerator = torch.Generator | list[torch.Generator] | None


def sample_noise(shape, generator: NoiseGenerator, device, dtype=torch.float32) -> Tensor:
    """Gaussian noise drawn on the CPU, so a sample does not depend on the device.

    ``generator`` is one torch.Generator for the whole batch, or one per batch
    element (a list), which makes each element's sample independent of what
    else is in the batch (evaluation seeds every episode from its IC index)."""
    if isinstance(generator, (list, tuple)):
        if len(generator) != shape[0]:
            raise ValueError(f"{len(generator)} generators for a batch of {shape[0]}")
        noise = torch.stack([torch.randn(tuple(shape[1:]), generator=g) for g in generator])
    else:
        noise = torch.randn(tuple(shape), generator=generator)
    return noise.to(device=device, dtype=dtype)


class DDPMScheduler:
    def __init__(
        self, num_train_timesteps: int = 100, clip_sample: bool = True, clip_range: float = 1.0
    ):
        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample
        self.clip_range = clip_range
        self.betas = squaredcos_cap_v2_betas(num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.timesteps = torch.arange(num_train_timesteps - 1, -1, -1)

    def to(self, device) -> DDPMScheduler:
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.timesteps = self.timesteps.to(device)
        return self

    def add_noise(self, clean: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        """x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps, with per-sample timesteps."""
        abar = self.alphas_cumprod.to(clean.device)[timesteps].to(clean.dtype)
        while abar.ndim < clean.ndim:
            abar = abar[..., None]
        return abar.sqrt() * clean + (1 - abar).sqrt() * noise

    def _predict_x0(self, eps: Tensor, t: int, sample: Tensor) -> Tensor:
        abar = self.alphas_cumprod[t]
        x0 = (sample - (1 - abar).sqrt() * eps) / abar.sqrt()
        if self.clip_sample:
            x0 = x0.clamp(-self.clip_range, self.clip_range)
        return x0

    def step(self, eps: Tensor, t: int, sample: Tensor, generator: NoiseGenerator = None) -> Tensor:
        """One reverse step x_t -> x_{t-1} (DDPM, fixed_small variance)."""
        t = int(t)
        abar_t = self.alphas_cumprod[t]
        abar_prev = self.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0, device=sample.device)
        beta_t = 1 - abar_t / abar_prev
        alpha_t = 1 - beta_t
        x0 = self._predict_x0(eps, t, sample)
        coef_x0 = abar_prev.sqrt() * beta_t / (1 - abar_t)
        coef_xt = alpha_t.sqrt() * (1 - abar_prev) / (1 - abar_t)
        mean = coef_x0 * x0 + coef_xt * sample
        if t == 0:
            return mean
        variance = (beta_t * (1 - abar_prev) / (1 - abar_t)).clamp_min(1e-20)
        noise = sample_noise(sample.shape, generator, sample.device, sample.dtype)
        return mean + variance.sqrt() * noise


class DDIMScheduler(DDPMScheduler):
    def __init__(
        self,
        num_train_timesteps: int = 100,
        num_inference_steps: int = 10,
        clip_sample: bool = True,
        clip_range: float = 1.0,
    ):
        super().__init__(num_train_timesteps, clip_sample, clip_range)
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps: int) -> None:
        """'leading' spacing: 0, ratio, 2 ratio, ... reversed, as diffusers does by default."""
        self.num_inference_steps = num_inference_steps
        ratio = self.num_train_timesteps // num_inference_steps
        self.timesteps = (torch.arange(num_inference_steps) * ratio).flip(0).to(self.betas.device)

    def step(self, eps: Tensor, t: int, sample: Tensor, generator: NoiseGenerator = None) -> Tensor:
        """Deterministic DDIM step (eta = 0) from timestep t to t - ratio."""
        t = int(t)
        ratio = self.num_train_timesteps // self.num_inference_steps
        prev_t = t - ratio
        abar_prev = (
            self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0, device=sample.device)
        )
        x0 = self._predict_x0(eps, t, sample)
        # diffusers (and Diffusion Policy) keep the network's epsilon for the direction term
        # even though x0 was clipped (use_clipped_model_output=False).
        return abar_prev.sqrt() * x0 + (1 - abar_prev).sqrt() * eps
