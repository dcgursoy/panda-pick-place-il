import numpy as np
import pytest
import torch

from panda_il.policies.bc import BCConfig, BCPolicy
from panda_il.policies.diffusion.ema import EMAModel
from panda_il.policies.diffusion.policy import DiffusionConfig, DiffusionPolicy
from panda_il.policies.diffusion.schedulers import DDIMScheduler, DDPMScheduler
from panda_il.policies.normalize import MinMaxNormalizer, images_to_float
from panda_il.policies.obs_encoder import ObsEncoderConfig, ObservationEncoder

diffusers = pytest.importorskip("diffusers")


def test_minmax_normalizer_maps_to_unit_range_and_zeroes_constant_dims():
    data = np.stack([np.linspace(-2, 3, 50), np.zeros(50), np.linspace(10, 11, 50)], axis=1)
    norm = MinMaxNormalizer(3).fit(data)
    x = norm.normalize(torch.as_tensor(data, dtype=torch.float32))
    assert torch.allclose(x[:, 0].min(), torch.tensor(-1.0)) and torch.allclose(
        x[:, 0].max(), torch.tensor(1.0)
    )
    assert torch.all(x[:, 1] == 0)
    back = norm.unnormalize(x)
    assert torch.allclose(
        back[:, [0, 2]], torch.as_tensor(data[:, [0, 2]], dtype=torch.float32), atol=1e-5
    )
    assert torch.all(back[:, 1] == 0)


def test_images_to_float_layout_and_range():
    images = torch.zeros(2, 3, 8, 8, 3, dtype=torch.uint8)
    images[..., 0] = 255
    x = images_to_float(images)
    assert (
        x.shape == (2, 3, 3, 8, 8)
        and x[..., 0, :, :].max() == 1.0
        and x[..., 1, :, :].min() == -1.0
    )


def test_schedulers_match_diffusers():
    hf_ddpm = diffusers.DDPMScheduler(
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
        variance_type="fixed_small",
    )
    ours = DDPMScheduler(100)
    assert torch.allclose(ours.alphas_cumprod, hf_ddpm.alphas_cumprod.float(), atol=1e-6)
    torch.manual_seed(0)
    x0, noise, t = (
        torch.rand(4, 16, 7) * 2 - 1,
        torch.randn(4, 16, 7),
        torch.tensor([0, 10, 50, 99]),
    )
    assert torch.allclose(ours.add_noise(x0, noise, t), hf_ddpm.add_noise(x0, noise, t), atol=1e-6)
    eps, xt = torch.randn(4, 16, 7), torch.randn(4, 16, 7)
    g1, g2 = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
    assert torch.allclose(
        ours.step(eps, 50, xt, generator=g1),
        hf_ddpm.step(eps, 50, xt, generator=g2).prev_sample,
        atol=1e-5,
    )
    assert torch.allclose(ours.step(eps, 0, xt), hf_ddpm.step(eps, 0, xt).prev_sample, atol=1e-5)

    hf_ddim = diffusers.DDIMScheduler(
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        set_alpha_to_one=True,
        steps_offset=0,
        prediction_type="epsilon",
    )
    for n in (10, 16):
        hf_ddim.set_timesteps(n)
        ddim = DDIMScheduler(100, n)
        assert ddim.timesteps.tolist() == hf_ddim.timesteps.tolist()
        a, b = xt.clone(), xt.clone()
        for step in ddim.timesteps.tolist():
            e = torch.randn(4, 16, 7) * 0.3
            a, b = ddim.step(e, step, a), hf_ddim.step(e, step, b, eta=0.0).prev_sample
        assert torch.allclose(a, b, atol=1e-5)


def _batch(obs_mode: str, batch: int = 3, horizon: int = 2):
    lowdim_dim = 29 if obs_mode == "state" else 22
    item = {
        "lowdim": torch.randn(batch, horizon, lowdim_dim),
        "action": torch.rand(batch, 16, 7) * 2 - 1,
    }
    if obs_mode == "image":
        for cam in ("front", "wrist"):
            item[f"image_{cam}"] = torch.randint(
                0, 256, (batch, horizon, 96, 96, 3), dtype=torch.uint8
            )
    return item


@pytest.mark.parametrize("obs_mode", ["state", "image"])
def test_observation_encoder_output_dim(obs_mode):
    cfg = ObsEncoderConfig(obs_mode=obs_mode, lowdim_dim=29 if obs_mode == "state" else 22)
    encoder = ObservationEncoder(cfg)
    out = encoder(_batch(obs_mode), train=True)
    expected = 2 * (29 if obs_mode == "state" else 22 + 2 * 64)
    assert out.shape == (3, expected) and encoder.output_dim == expected
    assert encoder(_batch(obs_mode), train=False).shape == (3, expected)


@pytest.mark.parametrize("pred_horizon", [16, 1])
def test_bc_policy_loss_and_prediction(pred_horizon):
    cfg = BCConfig(
        encoder=ObsEncoderConfig(obs_mode="state", lowdim_dim=29),
        pred_horizon=pred_horizon,
        hidden=32,
        layers=2,
    )
    policy = BCPolicy(cfg)
    batch = _batch("state", batch=4)
    policy.fit_normalizers(
        batch["lowdim"].reshape(-1, 29).numpy(), batch["action"].reshape(-1, 7).numpy()
    )
    loss = policy.compute_loss(batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert policy.predict(batch).shape == (4, pred_horizon, 7)


def test_diffusion_policy_loss_prediction_and_determinism():
    cfg = DiffusionConfig(
        encoder=ObsEncoderConfig(obs_mode="state", lowdim_dim=29),
        diffusion_step_embed_dim=16,
        down_dims=(16, 32, 64),
        num_inference_steps=4,
    )
    policy = DiffusionPolicy(cfg)
    batch = _batch("state", batch=2)
    policy.fit_normalizers(
        batch["lowdim"].reshape(-1, 29).numpy(), batch["action"].reshape(-1, 7).numpy()
    )
    loss = policy.compute_loss(batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    a = policy.predict(batch, generator=torch.Generator().manual_seed(3))
    b = policy.predict(batch, generator=torch.Generator().manual_seed(3))
    assert a.shape == (2, 16, 7) and torch.equal(a, b)
    policy.cfg.sampler = "ddpm"
    assert policy.predict(batch, generator=torch.Generator().manual_seed(3)).shape == (2, 16, 7)


def test_ema_schedule_and_copy():
    model = torch.nn.Linear(3, 2)
    ema = EMAModel(model, power=0.75, max_decay=0.9999)
    assert ema.decay() == 0.0
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema.step(model)  # step 1: decay 0 -> copy
    assert torch.all(ema.model.weight == 1.0)
    with torch.no_grad():
        model.weight.fill_(0.0)
    decay = ema.step(model)
    assert 0 < decay < 1 and torch.allclose(
        ema.model.weight, torch.full_like(ema.model.weight, decay)
    )
    ema.step_count = 10_000_000  # (1 + step) ** -0.75 < 1e-4 only beyond 215k steps
    assert ema.decay() == pytest.approx(0.9999)
