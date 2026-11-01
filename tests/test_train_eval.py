"""Smoke tests for the training loop, checkpoint round trip and batched evaluation."""

import json

import numpy as np
import pytest
import torch

from panda_il.config import EnvConfig
from panda_il.data import splits as splits_mod
from panda_il.data.dataset import WindowConfig
from panda_il.data.schema import Dataset
from panda_il.env.sets import load_set
from panda_il.eval.evaluate import Evaluator, episode_seed
from panda_il.eval.failures import LABELS
from panda_il.policies.bc import BCConfig, BCPolicy
from panda_il.policies.checkpoint import load_checkpoint
from panda_il.policies.diffusion.policy import DiffusionConfig, DiffusionPolicy
from panda_il.policies.obs_encoder import ObsEncoderConfig
from panda_il.train.loop import TrainConfig, cosine_with_warmup, train
from tests import synthetic as syn


@pytest.fixture
def data_root(tmp_path):
    ds = Dataset.create(tmp_path / "datasets" / "synthetic", syn.synthetic_metadata())
    for i in range(12):
        ds.write_episode(syn.synthetic_record(n_steps=24 + i, seed=i, success=(i % 4 != 3)))
    ds = Dataset(tmp_path / "datasets" / "synthetic")
    splits_mod.save_splits(ds, splits_mod.make_splits(ds, val_size=3, seed=0))
    return tmp_path


def test_cosine_schedule_shape():
    assert cosine_with_warmup(0, 10, 100) == 0.0
    assert cosine_with_warmup(10, 10, 100) == pytest.approx(1.0)
    assert cosine_with_warmup(55, 10, 100) == pytest.approx(0.5)
    assert cosine_with_warmup(100, 10, 100) == pytest.approx(0.0, abs=1e-12)


def test_bc_training_smoke_and_checkpoint_round_trip(data_root):
    train_cfg = TrainConfig(
        dataset="synthetic",
        subset=4,
        seed=1,
        obs_mode="state",
        steps=4,
        batch_size=8,
        warmup_steps=2,
        log_every=2,
        val_every=4,
        val_batches=1,
        checkpoint_every=4,
        device="cpu",
        run_name="smoke/bc",
        data_root=str(data_root),
    )
    policy_cfg = BCConfig(
        encoder=ObsEncoderConfig(obs_mode="state", lowdim_dim=29), hidden=32, layers=2
    )
    window = WindowConfig(obs_mode="state")
    run_dir = train("bc", BCPolicy(policy_cfg), policy_cfg, train_cfg, window)
    assert (run_dir / "last.pt").exists() and (run_dir / "step_4.pt").exists()
    config = json.loads((run_dir / "config.json").read_text())
    assert len(config["train_episodes"]) == 4 and config["train"]["subset"] == 4
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["steps"] == 4 and np.isfinite(summary["final_train_loss"])
    policy, info = load_checkpoint(run_dir / "last.pt", torch.device("cpu"))
    assert info["weights"] == "ema" and info["step"] == 4 and info["method"] == "bc"
    batch = {"lowdim": torch.randn(3, 2, 29)}
    assert policy.predict(batch).shape == (3, 16, 7)


def test_diffusion_training_smoke(data_root):
    train_cfg = TrainConfig(
        dataset="synthetic",
        seed=0,
        obs_mode="state",
        steps=2,
        batch_size=4,
        warmup_steps=1,
        log_every=1,
        val_every=2,
        val_batches=1,
        checkpoint_every=2,
        ema=False,
        device="cpu",
        run_name="smoke/dp",
        data_root=str(data_root),
    )
    policy_cfg = DiffusionConfig(
        encoder=ObsEncoderConfig(obs_mode="state", lowdim_dim=29),
        diffusion_step_embed_dim=16,
        down_dims=(16, 32, 64),
        num_inference_steps=2,
    )
    run_dir = train("diffusion", DiffusionPolicy(policy_cfg), policy_cfg, train_cfg, WindowConfig())
    policy, info = load_checkpoint(run_dir / "last.pt", torch.device("cpu"))
    assert info["weights"] == "raw" and isinstance(policy, DiffusionPolicy)
    batch = {"lowdim": torch.randn(2, 2, 29)}
    gens = [torch.Generator().manual_seed(s) for s in (3, 4)]
    a = policy.predict(batch, generator=gens)
    fresh = [torch.Generator().manual_seed(4)]
    b = policy.predict({"lowdim": batch["lowdim"][1:]}, generator=fresh)
    assert a.shape == (2, 16, 7) and torch.allclose(a[1], b[0], atol=1e-5)


def test_policy_config_round_trip_keeps_cameras():
    from dataclasses import asdict

    from panda_il.policies.checkpoint import policy_config_from_dict

    cfg = DiffusionConfig(encoder=ObsEncoderConfig(obs_mode="image", lowdim_dim=22))
    back = policy_config_from_dict("diffusion", asdict(cfg))
    assert back == cfg and back.encoder.cameras == ("front", "wrist")
    bc = BCConfig(encoder=ObsEncoderConfig(obs_mode="image", lowdim_dim=22), pred_horizon=1)
    assert policy_config_from_dict("bc", asdict(bc)) == bc


def _state_policy():
    cfg = BCConfig(encoder=ObsEncoderConfig(obs_mode="state", lowdim_dim=29), hidden=16, layers=1)
    policy = BCPolicy(cfg)
    rng = np.random.default_rng(0)
    policy.fit_normalizers(rng.normal(size=(50, 29)), rng.uniform(-1, 1, size=(50, 7)))
    return policy.eval()


def test_episode_seed_is_a_pure_function():
    assert episode_seed(0, 1) == episode_seed(0, 1)
    assert len({episode_seed(0, 1), episode_seed(0, 2), episode_seed(1, 2)}) == 3


def test_state_evaluator_runs_in_lockstep_and_is_reproducible():
    env_cfg = EnvConfig(obs_mode="state", max_steps=6)
    window = WindowConfig(obs_mode="state", obs_horizon=2, pred_horizon=16, action_horizon=8)
    ics = load_set("dev_set_v1").ics[:3]
    evaluator = Evaluator(env_cfg, window, n_envs=2)
    try:
        results, frames = evaluator.run(_state_policy(), ics, [0, 1, 2], seed=0)
        again, _ = evaluator.run(_state_policy(), ics, [0, 1, 2], seed=0)
    finally:
        evaluator.close()
    assert [r.ic_index for r in results] == [0, 1, 2] and frames == {}
    assert all(r.steps == 6 and r.label in LABELS and not r.success for r in results)
    assert evaluator.env_steps == 36 and evaluator.infer_calls == 4  # 2 batches x 2 chunks
    assert [r.cube_final for r in results] == [r.cube_final for r in again]


@pytest.mark.render
def test_evaluator_frames_compose_three_panels():
    env_cfg = EnvConfig(obs_mode="state", max_steps=2)
    evaluator = Evaluator(env_cfg, WindowConfig(obs_mode="state"), n_envs=1)
    try:
        _, frames = evaluator.run(_state_policy(), load_set("dev_set_v1").ics[:1], [0], 0, {0})
    finally:
        evaluator.close()
    assert len(frames[0]) == 3
    assert frames[0][0].shape == (env_cfg.render_height, 3 * env_cfg.render_width + 16, 3)


def test_evaluation_limit_is_the_study_limit_not_the_recording_cap(tmp_path):
    from dataclasses import asdict

    from panda_il.eval.evaluate import env_config_for

    metadata = syn.synthetic_metadata()
    metadata.env_config = asdict(EnvConfig(obs_mode="both", max_steps=300))  # recording cap
    Dataset.create(tmp_path / "ds", metadata)
    ds = Dataset(tmp_path / "ds")
    assert env_config_for(ds, "state").max_steps == EnvConfig().max_steps == 224
    assert env_config_for(ds, "state", max_steps=40).max_steps == 40
