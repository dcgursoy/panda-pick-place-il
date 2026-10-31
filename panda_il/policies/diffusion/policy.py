"""Diffusion Policy (Chi et al. 2023): observation features condition a 1D UNet that
denoises an action chunk. Trained with DDPM (100 steps), sampled with DDIM (10 steps)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from panda_il.policies.diffusion.schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    NoiseGenerator,
    sample_noise,
)
from panda_il.policies.diffusion.unet1d import ConditionalUnet1D
from panda_il.policies.normalize import MinMaxNormalizer
from panda_il.policies.obs_encoder import ObsEncoderConfig, ObservationEncoder


@dataclass
class DiffusionConfig:
    encoder: ObsEncoderConfig = field(default_factory=ObsEncoderConfig)
    action_dim: int = 7
    pred_horizon: int = 16
    action_horizon: int = 8
    diffusion_step_embed_dim: int = 128
    down_dims: tuple[int, ...] = (128, 256, 512)
    kernel_size: int = 5
    n_groups: int = 8
    num_train_timesteps: int = 100
    num_inference_steps: int = 10
    sampler: str = "ddim"  # "ddim" (fast) or "ddpm" (reference)


class DiffusionPolicy(nn.Module):
    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = ObservationEncoder(cfg.encoder)
        self.action_normalizer = MinMaxNormalizer(cfg.action_dim)
        self.unet = ConditionalUnet1D(
            input_dim=cfg.action_dim,
            global_cond_dim=self.encoder.output_dim,
            diffusion_step_embed_dim=cfg.diffusion_step_embed_dim,
            down_dims=cfg.down_dims,
            kernel_size=cfg.kernel_size,
            n_groups=cfg.n_groups,
        )
        self.train_scheduler = DDPMScheduler(cfg.num_train_timesteps)
        self.ddim = DDIMScheduler(cfg.num_train_timesteps, cfg.num_inference_steps)
        self.ddpm = DDPMScheduler(cfg.num_train_timesteps)

    def fit_normalizers(self, lowdim, actions) -> None:
        self.encoder.lowdim_normalizer.fit(lowdim)
        self.action_normalizer.fit(actions)

    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        device = next(self.parameters()).device
        for scheduler in (self.train_scheduler, self.ddim, self.ddpm):
            scheduler.to(device)
        return module

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        actions = self.action_normalizer(batch["action"][:, : self.cfg.pred_horizon])
        cond = self.encoder(batch, train=True)
        noise = torch.randn_like(actions)
        timesteps = torch.randint(
            0, self.cfg.num_train_timesteps, (actions.shape[0],), device=actions.device
        )
        noisy = self.train_scheduler.add_noise(actions, noise, timesteps)
        pred = self.unet(noisy, timesteps, cond)
        return nn.functional.mse_loss(pred.float(), noise.float())

    @torch.no_grad()
    def predict(self, batch: dict[str, Tensor], generator: NoiseGenerator = None) -> Tensor:
        """Unnormalised action chunk (B, Tp, 7) sampled for an observation window."""
        cond = self.encoder(batch, train=False)
        scheduler = self.ddim if self.cfg.sampler == "ddim" else self.ddpm
        shape = (cond.shape[0], self.cfg.pred_horizon, self.cfg.action_dim)
        sample = sample_noise(shape, generator, cond.device)
        for t in scheduler.timesteps.tolist():
            eps = self.unet(sample, t, cond).float()
            sample = scheduler.step(eps, t, sample, generator=generator)
        return self.action_normalizer.unnormalize(sample)
