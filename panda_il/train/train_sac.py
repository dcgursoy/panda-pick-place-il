"""The SAC baseline (Stable-Baselines3) on the 29-D state with the shaped reward.

    uv run python -m panda_il.train.train_sac --seed 0                       # 1 M env steps, shaped reward
    uv run python -m panda_il.train.train_sac --reward reach --total-steps 100000   # reach-only wiring test
    uv run python -m panda_il.train.train_sac --total-steps 20000 --run-name sac/smoke   # throughput

Same environment as the imitation-learning policies (action interface, IK,
control rate, 224-step limit, success predicate, cube randomisation); the only
differences are the reward (DESIGN.md section 15) and that SAC sees the cube
pose, like the state-based IL policies. Episodes terminate on success and
truncate at the limit (SB3 bootstraps through truncations). Dev-set rollouts
every ``eval_every`` env steps reuse the batched evaluator through
``policies/sac_adapter.py``; the run directory holds SB3's ``progress.csv``
(env steps, wall clock, losses, entropy), TensorBoard events, checkpoints and
``summary.json``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import tyro
from gymnasium import spaces

from panda_il.config import EnvConfig, RewardConfig
from panda_il.data.dataset import WindowConfig
from panda_il.data.recorder import env_config_from_metadata
from panda_il.data.schema import Dataset, git_state
from panda_il.env.panda_env import STATE_DIM, PandaPickPlaceEnv
from panda_il.env.sets import load_set
from panda_il.paths import Paths
from panda_il.policies.sac_adapter import SACPolicyAdapter


@dataclass
class SACConfig:
    seed: int = 0
    total_steps: int = 1_000_000
    n_envs: int = 4
    reward: str = "shaped"  # "shaped" or "reach" (reach-only reward, wiring test)
    learning_starts: int = 10_000
    buffer_size: int = 1_000_000
    batch_size: int = 256
    gamma: float = 0.98
    tau: float = 0.005
    lr: float = 3e-4
    train_freq: int = 1  # SB3 "step" = one transition per env
    gradient_steps: int = 4  # one gradient step per transition with 4 envs
    net: tuple[int, int] = (256, 256)
    ent_coef: str = "auto"
    threads: int = 4
    eval_every: int = 50_000  # env steps between dev-set rollouts
    eval_episodes: int = 20
    eval_envs: int = 10
    checkpoint_every: int = 100_000
    run_name: str | None = None  # default: sac/<reward>/seed<k>
    data_root: str | None = None
    check_env: bool = False  # run SB3's env checker first


class StateObservation(gym.ObservationWrapper):
    """The env's Dict observation reduced to its 29-D state vector (SAC's MlpPolicy input)."""

    def __init__(self, env: PandaPickPlaceEnv):
        super().__init__(env)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(STATE_DIM,), dtype=np.float32)

    def observation(self, observation: dict) -> np.ndarray:
        return np.asarray(observation["state"], dtype=np.float32)


def reward_config(reward: str) -> RewardConfig:
    if reward == "shaped":
        return RewardConfig()
    if reward == "reach":
        return RewardConfig(reach=1.0, grasp=0.0, lift=0.0, place=0.0, success=0.0)
    raise ValueError(f"unknown reward {reward!r}")


def sac_env_config(reward: str) -> EnvConfig:
    return EnvConfig(obs_mode="state", reward=reward_config(reward))


def make_sac_env(cfg: EnvConfig):
    from stable_baselines3.common.monitor import Monitor

    return Monitor(StateObservation(PandaPickPlaceEnv(cfg)), info_keywords=("success",))


def env_config_differences(cfg: EnvConfig, paths: Paths, dataset: str = "scripted_v1") -> list[str]:
    """Fields in which the SAC env differs from the one the IL data was recorded with
    (obs_mode, reward and the recording cap are expected to differ)."""
    directory = paths.datasets / dataset
    if not (directory / "metadata.json").exists():
        return [f"dataset {dataset} not found; comparison skipped"]
    recorded = asdict(env_config_from_metadata(Dataset(directory).metadata))
    ours = asdict(cfg)
    ignored = {"obs_mode", "reward", "max_steps"}
    return [k for k in ours if k not in ignored and ours[k] != recorded.get(k)]


def train(cfg: SACConfig) -> Path:
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import DummyVecEnv

    from panda_il.eval.evaluate import Evaluator

    torch.set_num_threads(cfg.threads)
    paths = Paths.resolve(cfg.data_root)
    paths.make_dirs()
    env_cfg = sac_env_config(cfg.reward)
    run_dir = paths.runs / (cfg.run_name or f"sac/{cfg.reward}/seed{cfg.seed}")
    run_dir.mkdir(parents=True, exist_ok=True)
    differences = env_config_differences(env_cfg, paths)
    if differences:
        print(f"WARNING: SAC env differs from the scripted_v1 recording env in {differences}")
    else:
        print("SAC env config matches the scripted_v1 recording env (apart from reward, limit)")

    if cfg.check_env:
        from stable_baselines3.common.env_checker import check_env

        check_env(make_sac_env(env_cfg), warn=True)
        print("SB3 env checker: ok")

    vec_env = DummyVecEnv([lambda: make_sac_env(env_cfg) for _ in range(cfg.n_envs)])
    vec_env.seed(cfg.seed)
    model = SAC(
        "MlpPolicy",
        vec_env,
        learning_rate=cfg.lr,
        buffer_size=cfg.buffer_size,
        learning_starts=cfg.learning_starts,
        batch_size=cfg.batch_size,
        tau=cfg.tau,
        gamma=cfg.gamma,
        train_freq=cfg.train_freq,
        gradient_steps=cfg.gradient_steps,
        ent_coef=cfg.ent_coef,
        policy_kwargs={"net_arch": list(cfg.net)},
        seed=cfg.seed,
        device="cpu",
        verbose=0,
    )
    logger = configure(str(run_dir), ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)
    commit, dirty = git_state()
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "train": asdict(cfg),
                "env": asdict(env_cfg),
                "env_differences_vs_scripted_v1": differences,
                "git_commit": commit,
                "git_dirty": dirty,
                "sb3_policy": str(model.policy),
            },
            indent=1,
            default=str,
        ),
        encoding="utf-8",
    )

    dev_ics = load_set("dev_set_v1").ics[: cfg.eval_episodes]
    window = WindowConfig(obs_mode="state", obs_horizon=1, pred_horizon=1, action_horizon=1)
    evaluator = Evaluator(env_cfg, window, min(cfg.eval_envs, len(dev_ics)), bf16=False)
    adapter = SACPolicyAdapter(model.policy)
    dev_curve: list[tuple[int, float, float]] = []  # (env steps, success rate, mean reach m)

    class RolloutStats(BaseCallback):
        """Success rate and final reach distance of finished episodes; dev-set rollouts."""

        def __init__(self):
            super().__init__()
            self.successes: list[float] = []
            self.reach: list[float] = []
            self.next_eval = cfg.eval_every
            self.t0 = time.perf_counter()

        def _on_step(self) -> bool:
            for done, info in zip(self.locals["dones"], self.locals["infos"], strict=True):
                if done:
                    self.successes.append(float(info.get("success", False)))
                    self.reach.append(float(-info["reward_terms"]["reach"]))
            if self.successes and self.n_calls % 100 == 0:
                self.logger.record("rollout/success_rate", float(np.mean(self.successes[-100:])))
                self.logger.record("rollout/final_reach_m", float(np.mean(self.reach[-100:])))
            if self.num_timesteps >= self.next_eval:
                self.next_eval += cfg.eval_every
                self._dev_eval()
            return True

        def _dev_eval(self) -> None:
            adapter.eval()
            results, _ = evaluator.run(adapter, dev_ics, list(range(len(dev_ics))), seed=cfg.seed)
            success = float(np.mean([r.success for r in results]))
            reach = float(np.mean([r.cube_final[2] for r in results]))  # unused for reach test
            dev_curve.append((int(self.num_timesteps), success, reach))
            self.logger.record("dev/success_rate", success)
            self.logger.record("time/wall_minutes", (time.perf_counter() - self.t0) / 60)
            print(
                f"  {self.num_timesteps} env steps, {(time.perf_counter() - self.t0) / 60:.1f} min: "
                f"dev-set success {success:.0%}",
                flush=True,
            )

    checkpoints = CheckpointCallback(
        save_freq=max(1, cfg.checkpoint_every // cfg.n_envs),
        save_path=str(run_dir),
        name_prefix="sac",
    )
    stats = RolloutStats()
    print(
        f"SAC ({cfg.reward} reward), seed {cfg.seed}: {cfg.total_steps} env steps on {cfg.n_envs} envs"
    )
    print(f"run dir: {run_dir}")
    t0 = time.perf_counter()
    model.learn(total_timesteps=cfg.total_steps, callback=[checkpoints, stats], log_interval=10)
    elapsed = time.perf_counter() - t0
    model.save(run_dir / "final.zip")
    if not dev_curve or dev_curve[-1][0] < model.num_timesteps:
        stats._dev_eval()
    evaluator.close()
    vec_env.close()
    summary: dict[str, Any] = {
        "total_steps": int(model.num_timesteps),
        "wall_seconds": elapsed,
        "env_steps_per_second": model.num_timesteps / elapsed,
        "episodes": len(stats.successes),
        "train_success_rate_last100": float(np.mean(stats.successes[-100:]))
        if stats.successes
        else None,
        "train_final_reach_m_last100": float(np.mean(stats.reach[-100:])) if stats.reach else None,
        "dev_curve": [
            {"env_steps": s, "success_rate": r, "cube_final_z": z} for s, r, z in dev_curve
        ],
        "final_dev_success_rate": dev_curve[-1][1] if dev_curve else None,
        "threads": cfg.threads,
        "git_commit": commit,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(
        f"done: {model.num_timesteps} env steps in {elapsed / 60:.1f} min "
        f"({summary['env_steps_per_second']:.0f} steps/s), final dev success "
        f"{summary['final_dev_success_rate']}"
    )
    return run_dir


def main() -> None:
    train(tyro.cli(SACConfig))


if __name__ == "__main__":
    main()
