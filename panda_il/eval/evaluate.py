"""Closed-loop evaluation of a checkpoint on a committed initial-condition set.

``n_envs`` environments run in lockstep so one batched network call serves all
of them (single-sample diffusion inference on the iGPU is launch-bound). Each
episode is seeded from (eval seed, IC index) and its diffusion noise is drawn
from that seed alone, so a result does not depend on batch composition. The
policy sees the last ``obs_horizon`` observations (edge padded after reset, as
the training windows are), predicts a chunk and executes its first
``action_horizon`` steps: the windows in ``data/dataset.py`` align the chunk to
the current observation, so the executed slice is ``[0:Ta]``, the same actions
Chi et al. take as ``[To-1 : To-1+Ta]`` under their alignment.

    uv run python -m panda_il.eval.evaluate --checkpoint <run>/last.pt --ic-set dev_set_v1
    uv run python -m panda_il.eval.evaluate --checkpoint <run>/last.pt --ic-set eval_set_v1 --gifs 4

Every evaluation on a held-out set (eval_set_v1, robust_set_v1, robust_set_v1b)
appends a line to ``docs/eval_ledger.jsonl``.
"""

from __future__ import annotations

import json
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro

from panda_il.config import EnvConfig
from panda_il.data.dataset import WindowConfig
from panda_il.data.recorder import env_config_from_metadata
from panda_il.data.render import upscale, write_gif
from panda_il.data.schema import Dataset, git_state, utc_now
from panda_il.device import device_summary, resolve_device
from panda_il.env.panda_env import PandaPickPlaceEnv, make_env
from panda_il.env.sets import load_set
from panda_il.env.task import InitialCondition
from panda_il.eval.failures import EpisodeLog, label_episode
from panda_il.paths import DOCS_DIR, Paths
from panda_il.policies.checkpoint import load_checkpoint
from panda_il.policies.diffusion.policy import DiffusionPolicy
from panda_il.policies.diffusion.schedulers import DDIMScheduler

HELD_OUT_SETS = ("eval_set_v1", "robust_set_v1", "robust_set_v1b")
LEDGER_PATH = DOCS_DIR / "eval_ledger.jsonl"


@dataclass
class EvalConfig:
    checkpoint: str
    ic_set: str = "dev_set_v1"
    episodes: int | None = None  # first n ICs of the set; None = the whole set
    n_envs: int = 10
    seed: int = 0
    device: str = "auto"
    use_ema: bool = True
    bf16: bool = True
    sampler: str | None = None  # override the checkpoint's sampler: "ddim" or "ddpm"
    num_inference_steps: int | None = None  # override the DDIM step count
    gifs: int = 0  # render the first n episodes to GIFs
    gif_dir: str | None = None  # default: <run dir>/eval/gifs
    out: str | None = None  # report path; default <run dir>/eval/<ic_set>_s<seed>.json
    ledger: bool = True  # append held-out evaluations to docs/eval_ledger.jsonl
    max_steps: int | None = None  # override the episode limit the dataset was recorded with
    data_root: str | None = None


def episode_seed(seed: int, ic_index: int) -> int:
    return int(np.random.SeedSequence([seed, ic_index]).generate_state(1)[0])


def env_config_for(
    dataset: Dataset | None, obs_mode: str, max_steps: int | None = None
) -> EnvConfig:
    """The env a policy is evaluated in: the one its training data was recorded with, except
    for the episode limit, which is the study-wide evaluation limit (``EnvConfig.max_steps``,
    decision 6) rather than the looser cap the recording ran with."""
    cfg = env_config_from_metadata(dataset.metadata) if dataset is not None else EnvConfig()
    cfg.obs_mode = obs_mode
    cfg.max_steps = EnvConfig().max_steps if max_steps is None else max_steps
    return cfg


@dataclass
class EpisodeResult:
    ic_index: int
    ic: dict
    seed: int
    success: bool
    steps: int
    label: str
    grasp_step: int | None
    release_step: int | None
    resync_count: int
    cube_final: list[float]
    gif: str | None = None


class Evaluator:
    """Batched lockstep rollouts; it keeps its environments so repeated calls are cheap."""

    def __init__(self, env_cfg: EnvConfig, window: WindowConfig, n_envs: int, bf16: bool = True):
        self.env_cfg = env_cfg
        self.window = window
        self.envs: list[PandaPickPlaceEnv] = [make_env(env_cfg) for _ in range(n_envs)]
        self.lowdim_key = "state" if window.obs_mode == "state" else "proprio"
        self.cameras = tuple(window.cameras) if window.obs_mode == "image" else ()
        self.bf16 = bf16
        self.infer_seconds = 0.0
        self.infer_calls = 0
        self.env_seconds = 0.0
        self.env_steps = 0

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def _collate(self, histories, device: torch.device) -> dict[str, torch.Tensor]:
        batch = {
            "lowdim": np.stack([np.stack([o[self.lowdim_key] for o in h]) for h in histories]),
        }
        for cam in self.cameras:
            batch[f"image_{cam}"] = np.stack(
                [np.stack([o[f"image_{cam}"] for o in h]) for h in histories]
            )
        return {k: torch.from_numpy(v).to(device) for k, v in batch.items()}

    def frame(self, env: PandaPickPlaceEnv, obs: dict) -> np.ndarray:
        """[third-person | what the policy sees (or the cameras, for state policies)]."""
        panels = [env.render_view("viewer")]
        if self.cameras:
            panels += [upscale(obs[f"image_{cam}"], 4) for cam in self.cameras]
        else:
            panels += [env.render_view(cam) for cam in self.env_cfg.camera.names]
        height = min(p.shape[0] for p in panels)
        pad = np.zeros((height, 8, 3), dtype=np.uint8)
        row: list[np.ndarray] = []
        for i, panel in enumerate(panels):
            if i:
                row.append(pad)
            row.append(panel[:height])
        return np.concatenate(row, axis=1)

    @torch.no_grad()
    def run(
        self,
        policy,
        ics: list[InitialCondition],
        indices: list[int],
        seed: int,
        record: set[int] | None = None,
    ) -> tuple[list[EpisodeResult], dict[int, list[np.ndarray]]]:
        """Roll out ``policy`` on ``ics`` (identified by ``indices``). Returns results in
        input order and, for the IC indices in ``record``, the rendered frames."""
        device = next(policy.parameters()).device
        autocast = torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=self.bf16 and device.type in ("xpu", "cuda")
        )
        obs_horizon, action_horizon = self.window.obs_horizon, self.window.action_horizon
        record = set(record or ())
        results: list[EpisodeResult] = []
        frames: dict[int, list[np.ndarray]] = {}
        n = len(self.envs)
        for start in range(0, len(ics), n):
            batch_ics, batch_idx = ics[start : start + n], indices[start : start + n]
            envs = self.envs[: len(batch_ics)]
            histories, logs, seeds = [], [], []
            done = [False] * len(envs)
            steps = [0] * len(envs)
            for env, ic, idx in zip(envs, batch_ics, batch_idx, strict=True):
                s = episode_seed(seed, idx)
                obs, info = env.reset(seed=s, options={"ic": ic})
                histories.append(deque([obs] * obs_horizon, maxlen=obs_horizon))
                log = EpisodeLog()
                log.append(info)
                logs.append(log)
                seeds.append(s)
                if idx in record:
                    frames[idx] = [self.frame(env, obs)]
            generators = [torch.Generator().manual_seed(s) for s in seeds]
            while not all(done):
                batch = self._collate(histories, device)
                t0 = time.perf_counter()
                with autocast:
                    if isinstance(policy, DiffusionPolicy):
                        chunk = policy.predict(batch, generator=generators)
                    else:
                        chunk = policy.predict(batch)
                actions = chunk[:, :action_horizon].float().cpu().numpy().clip(-1.0, 1.0)
                self.infer_seconds += time.perf_counter() - t0
                self.infer_calls += 1
                for k in range(action_horizon):
                    t1 = time.perf_counter()
                    for e, env in enumerate(envs):
                        if done[e]:
                            continue
                        obs, _, terminated, truncated, info = env.step(actions[e, k])
                        steps[e] += 1
                        self.env_steps += 1
                        histories[e].append(obs)
                        logs[e].append(info)
                        if batch_idx[e] in record:
                            frames[batch_idx[e]].append(self.frame(env, obs))
                        if terminated or truncated:
                            done[e] = True
                    self.env_seconds += time.perf_counter() - t1
                    if all(done):
                        break
            for e, (ic, idx) in enumerate(zip(batch_ics, batch_idx, strict=True)):
                verdict = label_episode(logs[e])
                results.append(
                    EpisodeResult(
                        ic_index=idx,
                        ic=ic.to_dict(),
                        seed=seeds[e],
                        success=verdict["label"] == "success",
                        steps=steps[e],
                        label=verdict["label"],
                        grasp_step=verdict["grasp_step"],
                        release_step=verdict["release_step"],
                        resync_count=int(logs[e].resync_count[-1]),
                        cube_final=verdict["cube_final"],
                    )
                )
        return results, frames


def window_from_checkpoint(info: dict[str, Any]) -> WindowConfig:
    policy_cfg = info["policy_config"]
    encoder = policy_cfg["encoder"]
    return WindowConfig(
        obs_mode=encoder["obs_mode"],
        obs_horizon=encoder["obs_horizon"],
        pred_horizon=max(1, policy_cfg["pred_horizon"]),
        action_horizon=min(policy_cfg["action_horizon"], max(1, policy_cfg["pred_horizon"])),
        cameras=tuple(encoder["cameras"]),
    )


def make_dev_eval(train_cfg, window: WindowConfig):
    """Callback for the training loop: success rate on the first dev-set ICs."""
    paths = Paths.resolve(train_cfg.data_root)
    dataset = Dataset.open(paths, train_cfg.dataset)
    env_cfg = env_config_for(dataset, window.obs_mode)
    ics = load_set("dev_set_v1").ics[: train_cfg.dev_eval_episodes]
    state: dict[str, Evaluator] = {}

    def dev_eval(policy, step: int) -> float:
        if "evaluator" not in state:
            state["evaluator"] = Evaluator(env_cfg, window, min(train_cfg.dev_eval_envs, len(ics)))
        results, _ = state["evaluator"].run(policy, ics, list(range(len(ics))), seed=train_cfg.seed)
        return float(np.mean([r.success for r in results]))

    return dev_eval


def evaluate(cfg: EvalConfig) -> dict[str, Any]:
    paths = Paths.resolve(cfg.data_root)
    device = resolve_device(cfg.device, paths)
    policy, info = load_checkpoint(cfg.checkpoint, device, use_ema=cfg.use_ema)
    train_cfg = info["train_config"]
    sampler = num_inference_steps = None
    if isinstance(policy, DiffusionPolicy):
        if cfg.sampler:
            policy.cfg.sampler = cfg.sampler
        if cfg.num_inference_steps:
            policy.cfg.num_inference_steps = cfg.num_inference_steps
            policy.ddim = DDIMScheduler(policy.cfg.num_train_timesteps, cfg.num_inference_steps)
            policy.to(device)
        sampler = policy.cfg.sampler
        num_inference_steps = (
            policy.cfg.num_inference_steps if sampler == "ddim" else policy.cfg.num_train_timesteps
        )
    window = window_from_checkpoint(info)
    dataset_dir = paths.datasets / train_cfg["dataset"]
    dataset = Dataset(dataset_dir) if (dataset_dir / "metadata.json").exists() else None
    env_cfg = env_config_for(dataset, window.obs_mode, cfg.max_steps)
    ic_set = load_set(cfg.ic_set)
    ics = ic_set.ics[: cfg.episodes] if cfg.episodes else list(ic_set.ics)
    indices = list(range(len(ics)))

    run_dir = Path(cfg.checkpoint).resolve().parent
    out = Path(cfg.out) if cfg.out else run_dir / "eval" / f"{cfg.ic_set}_s{cfg.seed}.json"
    gif_dir = Path(cfg.gif_dir) if cfg.gif_dir else run_dir / "eval" / "gifs"
    print(
        f"{info['method']} step {info['step']} ({info['weights']} weights) on {cfg.ic_set}: "
        f"{len(ics)} episodes, {min(cfg.n_envs, len(ics))} envs, {device_summary(device)}, "
        f"sampler {sampler} x {num_inference_steps}, limit {env_cfg.max_steps} steps"
    )
    evaluator = Evaluator(env_cfg, window, min(cfg.n_envs, len(ics)), bf16=cfg.bf16)
    t0 = time.perf_counter()
    results, frames = evaluator.run(policy, ics, indices, cfg.seed, record=set(range(cfg.gifs)))
    wall = time.perf_counter() - t0
    evaluator.close()
    for idx, episode_frames in frames.items():
        result = results[idx]
        path = gif_dir / f"{cfg.ic_set}_ic{idx:03d}_{result.label.replace(' ', '_')}.gif"
        write_gif(episode_frames, path)
        result.gif = str(path)

    labels = Counter(r.label for r in results)
    successes = sum(r.success for r in results)
    commit, dirty = git_state()
    report: dict[str, Any] = {
        "checkpoint": str(Path(cfg.checkpoint).resolve()),
        "method": info["method"],
        "step": info["step"],
        "weights": info["weights"],
        "train": {k: train_cfg.get(k) for k in ("dataset", "subset", "seed", "obs_mode", "steps")},
        "dataset_source": info.get("dataset_source"),
        "ic_set": cfg.ic_set,
        "n_episodes": len(results),
        "eval_seed": cfg.seed,
        "n_envs": min(cfg.n_envs, len(ics)),
        "sampler": sampler,
        "num_inference_steps": num_inference_steps,
        "max_steps": env_cfg.max_steps,
        "device": device_summary(device),
        "bf16": cfg.bf16 and device.type in ("xpu", "cuda"),
        "success_rate": successes / max(1, len(results)),
        "successes": successes,
        "mean_steps": float(np.mean([r.steps for r in results])),
        "mean_steps_success": float(np.mean([r.steps for r in results if r.success] or [np.nan])),
        "labels": dict(sorted(labels.items(), key=lambda kv: -kv[1])),
        "resync_episodes": sum(r.resync_count > 0 for r in results),
        "inference_ms_per_call": 1000 * evaluator.infer_seconds / max(1, evaluator.infer_calls),
        "env_ms_per_step": 1000 * evaluator.env_seconds / max(1, evaluator.env_steps),
        "wall_seconds": wall,
        "git_commit": commit,
        "git_dirty": dirty,
        "utc": utc_now(),
        "episodes": [asdict(r) for r in results],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    if cfg.ledger and cfg.ic_set in HELD_OUT_SETS:
        entry = {k: report[k] for k in ("utc", "ic_set", "method", "step", "weights", "sampler")}
        entry.update(
            train=report["train"],
            n_episodes=report["n_episodes"],
            eval_seed=cfg.seed,
            success_rate=report["success_rate"],
            checkpoint=report["checkpoint"],
            report=str(out),
            git_commit=commit,
        )
        with LEDGER_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    print(
        f"success {successes}/{len(results)} = {report['success_rate']:.0%}  labels {report['labels']}"
    )
    print(
        f"{wall / 60:.1f} min, inference {report['inference_ms_per_call']:.0f} ms/call, "
        f"env {report['env_ms_per_step']:.1f} ms/step -> {out}"
    )
    return report


def main() -> None:
    evaluate(tyro.cli(EvalConfig))


if __name__ == "__main__":
    main()
