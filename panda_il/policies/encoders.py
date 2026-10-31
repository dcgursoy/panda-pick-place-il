"""Image encoder shared by BC-ResNet and the image Diffusion Policy.

This is the robomimic encoder that Diffusion Policy's simulated image results
use: a torchvision ResNet18 trunk (no pretraining) with every BatchNorm replaced
by GroupNorm so that EMA weight averaging is well behaved, truncated before the
average pool, followed by SpatialSoftmax (32 keypoints), a linear layer to 64
features and a ReLU. One encoder per camera.
"""

from __future__ import annotations

import torch
import torchvision
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812


def replace_batchnorm_with_groupnorm(root: nn.Module, features_per_group: int = 16) -> nn.Module:
    """Recursively swap each BatchNorm2d for GroupNorm(num_features // features_per_group)."""
    for name, child in root.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(
                root,
                name,
                nn.GroupNorm(
                    num_groups=child.num_features // features_per_group,
                    num_channels=child.num_features,
                ),
            )
        else:
            replace_batchnorm_with_groupnorm(child, features_per_group)
    return root


class SpatialSoftmax(nn.Module):
    """Expected (x, y) position of ``num_kp`` attention maps, each in [-1, 1]."""

    def __init__(self, in_channels: int, num_kp: int = 32, temperature: float = 1.0):
        super().__init__()
        self.num_kp = num_kp
        self.temperature = temperature
        self.keypoint_conv = nn.Conv2d(in_channels, num_kp, kernel_size=1)

    def forward(self, features: Tensor) -> Tensor:
        batch, _, height, width = features.shape
        logits = self.keypoint_conv(features).reshape(batch, self.num_kp, height * width)
        attention = F.softmax(logits / self.temperature, dim=-1)
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=features.device, dtype=attention.dtype),
            torch.linspace(-1.0, 1.0, width, device=features.device, dtype=attention.dtype),
            indexing="ij",
        )
        expected_x = (attention * xs.reshape(1, 1, -1)).sum(dim=-1)
        expected_y = (attention * ys.reshape(1, 1, -1)).sum(dim=-1)
        return torch.stack([expected_x, expected_y], dim=-1).reshape(batch, self.num_kp * 2)


class ImageEncoder(nn.Module):
    """ResNet18-GroupNorm trunk -> SpatialSoftmax -> Linear -> ReLU, for one camera.

    Input: (B, 3, H, W) float images scaled to [-1, 1]. Output: (B, feature_dim).
    """

    def __init__(
        self,
        feature_dim: int = 64,
        num_keypoints: int = 32,
        group_norm: bool = True,
        pretrained: bool = False,
    ):
        super().__init__()
        if pretrained and group_norm:
            raise ValueError(
                "ImageNet weights are BatchNorm statistics; use group_norm=False with pretrained=True"
            )
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = torchvision.models.resnet18(weights=weights)
        trunk = nn.Sequential(*list(resnet.children())[:-2])  # drop avgpool and fc
        if group_norm:
            replace_batchnorm_with_groupnorm(trunk)
        self.trunk = trunk
        self.pool = SpatialSoftmax(512, num_kp=num_keypoints)
        self.proj = nn.Linear(num_keypoints * 2, feature_dim)
        self.feature_dim = feature_dim

    def forward(self, images: Tensor) -> Tensor:
        return F.relu(self.proj(self.pool(self.trunk(images))))


def crop_images(images: Tensor, crop_size: int, random: bool) -> Tensor:
    """Random crop (training) or centre crop (evaluation) of a (B, 3, H, W) batch.

    A single crop offset per batch call is what Diffusion Policy's CropRandomizer does
    per sample; here each sample gets its own offset, which is the same augmentation.
    """
    batch, _, height, width = images.shape
    if crop_size > height or crop_size > width:
        raise ValueError(f"crop {crop_size} larger than image {height}x{width}")
    if not random:
        top = (height - crop_size) // 2
        left = (width - crop_size) // 2
        return images[:, :, top : top + crop_size, left : left + crop_size]
    tops = torch.randint(0, height - crop_size + 1, (batch,), device=images.device)
    lefts = torch.randint(0, width - crop_size + 1, (batch,), device=images.device)
    rows = tops[:, None] + torch.arange(crop_size, device=images.device)[None, :]
    cols = lefts[:, None] + torch.arange(crop_size, device=images.device)[None, :]
    batch_index = torch.arange(batch, device=images.device)[:, None, None]
    return images[batch_index, :, rows[:, :, None], cols[:, None, :]].permute(0, 3, 1, 2)
