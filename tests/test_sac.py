"""The SAC baseline: env wrapper, a tiny training run, checkpoint loading through the evaluator."""

import json

import numpy as np
import pytest
import torch

from panda_il.config import EnvConfig
from panda_il.data.dataset import WindowConfig
from panda_il.env.panda_env import make_env
from panda_il.env.sets import load_set
from panda_il.eval.evaluate import Evaluator
from panda_il.policies.checkpoint import load_checkpoint
from panda_il.train.train_sac import (
    SACConfig,
    StateObservation,
    make_sac_env,
    sac_env_config,
    train,
)

sb3 = pytest.importorskip("stable_baselines3")


def test_state_wrapper_spaces_and_sb3_checker():
    from stable_baselines3.common.env_checker import check_env

    env = make_sac_env(EnvConfig(obs_mode="state", max_steps=5))
    assert env.observation_space.shape == (29,) and env.observation_space.dtype == np.float32
    assert env.action_space.shape == (7,)
    obs, info = env.reset(seed=0)
    assert obs.shape == (29,) and "success" in info
    check_env(env, warn=False)
    env.close()


def test_reach_reward_config_zeroes_the_other_terms():
    cfg = sac_env_config("reach")
    assert cfg.reward.reach == 1.0 and cfg.reward.success == 0.0 and cfg.reward.grasp == 0.0
    assert cfg.max_steps == EnvConfig().max_steps  # the study-wide limit
    with pytest.raises(ValueError):
        sac_env_config("bogus")


def test_sac_smoke_train_checkpoint_and_evaluate(tmp_path):
    cfg = SACConfig(
        seed=3,
        total_steps=48,
        n_envs=2,
        reward="reach",
        learning_starts=8,
        buffer_size=500,
        batch_size=8,
        gradient_steps=1,
        net=(16, 16),
        threads=1,
        eval_every=24,
        eval_episodes=2,
        eval_envs=2,
        checkpoint_every=24,
        run_name="smoke/sac",
        data_root=str(tmp_path),
    )
    run_dir = train(cfg)
    assert (run_dir / "final.zip").exists() and (run_dir / "progress.csv").exists()
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["total_steps"] >= 48 and len(summary["dev_curve"]) >= 1
    config = json.loads((run_dir / "config.json").read_text())
    assert config["train"]["reward"] == "reach"

    policy, info = load_checkpoint(run_dir / "final.zip", torch.device("cpu"))
    assert info["method"] == "sac" and info["policy_config"]["pred_horizon"] == 1
    batch = {"lowdim": torch.randn(3, 1, 29)}
    assert policy.predict(batch).shape == (3, 1, 7)
    evaluator = Evaluator(
        EnvConfig(obs_mode="state", max_steps=4),
        WindowConfig(obs_mode="state", obs_horizon=1, pred_horizon=1, action_horizon=1),
        n_envs=2,
    )
    try:
        results, _ = evaluator.run(policy, load_set("dev_set_v1").ics[:2], [0, 1], seed=0)
    finally:
        evaluator.close()
    assert [r.steps for r in results] == [4, 4]


def test_state_observation_wrapper_keeps_info():
    env = StateObservation(make_env(EnvConfig(obs_mode="state", max_steps=3)))
    obs, _ = env.reset(seed=1)
    obs2, reward, terminated, truncated, info = env.step(np.zeros(7, dtype=np.float32))
    assert obs2.shape == (29,) and isinstance(reward, float) and "reward_terms" in info
    env.close()


def test_sac_resume_continues_the_step_count(tmp_path):
    base = SACConfig(
        seed=1,
        total_steps=24,
        n_envs=2,
        reward="reach",
        learning_starts=8,
        buffer_size=200,
        batch_size=4,
        gradient_steps=1,
        net=(8, 8),
        threads=1,
        eval_every=1000,
        eval_episodes=1,
        eval_envs=1,
        checkpoint_every=1000,
        run_name="smoke/sac_base",
        data_root=str(tmp_path),
    )
    first = train(base)
    assert (first / "replay_buffer.pkl").exists()
    resumed = train(
        SACConfig(**{**base.__dict__, "resume": str(first), "run_name": "smoke/sac_resumed"})
    )
    summary = json.loads((resumed / "summary.json").read_text())
    assert summary["total_steps"] >= 48 and summary["steps_this_run"] >= 24
    assert "replay buffer" in json.loads((resumed / "config.json").read_text())["resumed_from"]
