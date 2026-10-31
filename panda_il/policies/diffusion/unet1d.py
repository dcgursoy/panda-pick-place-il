"""The 1D temporal conditional UNet of Diffusion Policy (Chi et al. 2023).

Layout follows the official ``ConditionalUnet1D``: the noisy action sequence
``(B, T, action_dim)`` is treated as a 1D signal with ``action_dim`` channels.
Every residual block is conditioned through FiLM (a per-channel scale and bias)
on the concatenation of the diffusion-step embedding and the observation
features (``global_cond``). Two residual blocks per level on the way down, a
strided conv halves the horizon between levels, two blocks in the middle, and
skip connections are concatenated on the way up.

With three levels the horizon is halved twice, so it must be divisible by 4
(16 in this project).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn


class SinusoidalPosEmb(nn.Module):
    """Standard sinusoidal embedding of the (integer) diffusion step."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        half_dim = self.dim // 2
        scale = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=x.device, dtype=torch.float32) * -scale)
        args = x.to(torch.float32)[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class Downsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class ConditionalResidualBlock1D(nn.Module):
    """Two Conv1dBlocks with FiLM conditioning in between and a residual 1x1 conv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.cond_predict_scale = cond_predict_scale
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups),
            ]
        )
        cond_channels = out_channels * 2 if cond_predict_scale else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale, bias = embed[:, 0], embed[:, 1]
            out = scale * out + bias
        else:
            out = out + embed[:, :, None]
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    """Predicts the noise (or the clean sample) for a batch of action sequences.

    Args:
        input_dim: action dimension (channels of the 1D signal).
        global_cond_dim: size of the flattened observation features.
        diffusion_step_embed_dim: size of the sinusoidal step embedding ("dsed").
        down_dims: channel widths of the levels; the paper uses (256, 512, 1024) for
            low-dimensional tasks and (512, 1024, 2048) for image tasks.
        kernel_size: conv kernel size (5 in every official config).
        n_groups: GroupNorm groups.
        cond_predict_scale: FiLM with scale and bias (True in every official config).
    """

    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 256,
        down_dims: Sequence[int] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
    ):
        super().__init__()
        all_dims = [input_dim, *down_dims]
        start_dim = down_dims[0]
        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim
        in_out = list(zip(all_dims[:-1], all_dims[1:], strict=True))
        block_kwargs = dict(
            cond_dim=cond_dim,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(mid_dim, mid_dim, **block_kwargs),
                ConditionalResidualBlock1D(mid_dim, mid_dim, **block_kwargs),
            ]
        )

        self.down_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= len(in_out) - 1
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(dim_in, dim_out, **block_kwargs),
                        ConditionalResidualBlock1D(dim_out, dim_out, **block_kwargs),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.up_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= len(in_out) - 1
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(dim_out * 2, dim_in, **block_kwargs),
                        ConditionalResidualBlock1D(dim_in, dim_in, **block_kwargs),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size, n_groups=n_groups),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    @property
    def num_levels(self) -> int:
        return len(self.down_modules)

    def forward(self, sample: Tensor, timestep: Tensor | int, global_cond: Tensor) -> Tensor:
        """
        Args:
            sample: (B, T, input_dim) noisy action sequence.
            timestep: scalar or (B,) diffusion step indices.
            global_cond: (B, global_cond_dim) observation features.
        Returns:
            (B, T, input_dim) prediction.
        """
        x = sample.moveaxis(-1, -2)  # (B, C, T)
        batch = x.shape[0]
        timesteps = torch.as_tensor(timestep, device=x.device)
        if timesteps.ndim == 0:
            timesteps = timesteps[None]
        timesteps = timesteps.expand(batch)

        global_feature = self.diffusion_step_encoder(timesteps)
        global_feature = torch.cat([global_feature, global_cond], dim=-1)

        skips: list[Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            skips.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, skips.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        return x.moveaxis(-1, -2)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
