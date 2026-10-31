import torch
from torch import nn

from panda_il.policies.diffusion.unet1d import ConditionalUnet1D, count_parameters
from panda_il.policies.encoders import ImageEncoder, SpatialSoftmax, crop_images


def test_unet_output_shape_and_timestep_forms():
    unet = ConditionalUnet1D(
        input_dim=7, global_cond_dim=58, diffusion_step_embed_dim=32, down_dims=(16, 32, 64)
    )
    sample = torch.randn(4, 16, 7)
    cond = torch.randn(4, 58)
    out_vec = unet(sample, torch.randint(0, 100, (4,)), cond)
    out_int = unet(sample, 5, cond)
    assert out_vec.shape == (4, 16, 7)
    assert out_int.shape == (4, 16, 7)
    assert unet.num_levels == 3


def test_unet_matches_official_parameter_count():
    # The official state-based Push-T notebook prints 6.54e7 parameters for
    # input_dim 2, global_cond_dim 10, dsed 256, down_dims (256, 512, 1024), kernel 5.
    unet = ConditionalUnet1D(input_dim=2, global_cond_dim=10)
    n = count_parameters(unet)
    assert abs(n - 6.54e7) / 6.54e7 < 0.01, n


def test_unet_horizon_must_be_divisible_by_four():
    unet = ConditionalUnet1D(
        input_dim=7, global_cond_dim=8, diffusion_step_embed_dim=16, down_dims=(8, 16, 32)
    )
    out = unet(torch.randn(2, 12, 7), 1, torch.randn(2, 8))
    assert out.shape == (2, 12, 7)


def test_encoder_uses_groupnorm_and_outputs_feature_dim():
    encoder = ImageEncoder(feature_dim=64, num_keypoints=32)
    assert not any(isinstance(m, nn.BatchNorm2d) for m in encoder.modules())
    assert any(isinstance(m, nn.GroupNorm) for m in encoder.modules())
    for size in (84, 96):
        out = encoder(torch.rand(2, 3, size, size) * 2 - 1)
        assert out.shape == (2, 64)
        assert (out >= 0).all()


def test_spatial_softmax_keypoints_are_in_unit_square():
    pool = SpatialSoftmax(in_channels=8, num_kp=5)
    out = pool(torch.randn(3, 8, 6, 7))
    assert out.shape == (3, 10)
    assert out.abs().max() <= 1.0


def test_crop_images_center_and_random():
    images = torch.arange(2 * 3 * 96 * 96, dtype=torch.float32).reshape(2, 3, 96, 96)
    center = crop_images(images, 84, random=False)
    assert center.shape == (2, 3, 84, 84)
    assert torch.equal(center, images[:, :, 6:90, 6:90])
    torch.manual_seed(0)
    rand = crop_images(images, 84, random=True)
    assert rand.shape == (2, 3, 84, 84)
    # every random crop is a contiguous window of the source image
    for b in range(2):
        top_left = rand[b, 0, 0, 0].item()
        row, col = divmod(int(top_left) % (96 * 96), 96)
        assert torch.equal(rand[b], images[b, :, row : row + 84, col : col + 84])
