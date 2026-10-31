"""Turn an observation window into one feature vector, shared by BC and Diffusion Policy.

State mode:  the normalised 29-D state of each of the ``obs_horizon`` steps, concatenated.
Image mode:  per camera, the ResNet18-GroupNorm + SpatialSoftmax encoder (64 features),
             plus the normalised 22-D proprio, per step; concatenated over the steps.
Images are cropped to ``crop`` pixels: a random crop in training, the centre crop at evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from panda_il.policies.encoders import ImageEncoder, crop_images
from panda_il.policies.normalize import MinMaxNormalizer, images_to_float


@dataclass
class ObsEncoderConfig:
    obs_mode: str = "state"  # "state" or "image"
    obs_horizon: int = 2
    lowdim_dim: int = 29  # 29 for state, 22 for image (proprio)
    cameras: tuple[str, ...] = ("front", "wrist")
    image_feature_dim: int = 64
    spatial_softmax_keypoints: int = 32
    crop: int = 84
    group_norm: bool = True
    pretrained_backbone: bool = False


class ObservationEncoder(nn.Module):
    def __init__(self, cfg: ObsEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.lowdim_normalizer = MinMaxNormalizer(cfg.lowdim_dim)
        self.image_encoders = nn.ModuleDict()
        if cfg.obs_mode == "image":
            for cam in cfg.cameras:
                self.image_encoders[cam] = ImageEncoder(
                    feature_dim=cfg.image_feature_dim,
                    num_keypoints=cfg.spatial_softmax_keypoints,
                    group_norm=cfg.group_norm,
                    pretrained=cfg.pretrained_backbone,
                )
        per_step = cfg.lowdim_dim + len(self.image_encoders) * cfg.image_feature_dim
        self.output_dim = per_step * cfg.obs_horizon

    def forward(self, batch: dict[str, Tensor], train: bool) -> Tensor:
        """``batch["lowdim"]`` is (B, To, D) and ``batch["image_<cam>"]`` (B, To, H, W, 3) uint8.
        Returns (B, output_dim)."""
        lowdim = batch["lowdim"]
        if lowdim.ndim == 2:
            lowdim = lowdim[:, None]
        batch_size, horizon, _ = lowdim.shape
        parts = [self.lowdim_normalizer(lowdim)]
        for cam, encoder in self.image_encoders.items():
            images = batch[f"image_{cam}"]
            if images.ndim == 4:
                images = images[:, None]
            frames = images_to_float(images.reshape(batch_size * horizon, *images.shape[2:]))
            frames = crop_images(frames, self.cfg.crop, random=train)
            features = encoder(frames).reshape(batch_size, horizon, -1)
            parts.append(features)
        return torch.cat(parts, dim=-1).reshape(batch_size, -1)
