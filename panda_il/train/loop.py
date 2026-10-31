"""The training loop shared by behavior cloning and Diffusion Policy.

One protocol per observation modality, applied to every method (DESIGN.md,
decision 13): AdamW (lr 1e-4, weight decay 1e-6, betas 0.95 / 0.999), cosine
learning rate with linear warm-up, EMA weights, the same windows, batch size
and number of gradient steps, random crops in training. The dataset is held in
RAM; batches are built in the main process (Windows).

A run directory holds ``config.json``, TensorBoard events, checkpoints
(``step_<n>.pt`` plus ``last.pt``) and ``summary.json`` with timing.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from panda_il.data.dataset import WindowConfig, WindowDataset, load_episodes
from panda_il.data.schema import Dataset, git_state
from panda_il.data.splits import load_splits, subset_ids
from panda_il.device import device_summary, resolve_device
from panda_il.paths import Paths
from panda_il.policies.diffusion.ema import EMAModel


@dataclass
class TrainConfig:
    dataset: str = "scripted_v1"
    subset: int | None = None  # number of training demonstrations (nested subset), None = all
    seed: int = 0
    obs_mode: str = "state"  # "state" or "image"
    steps: int = 20000
    batch_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 1e-6
    betas: tuple[float, float] = (0.95, 0.999)
    warmup_steps: int = 500
    ema: bool = True
    ema_power: float = 0.75
    ema_max_decay: float = 0.9999
    bf16: bool = True  # autocast on the accelerator (ignored on CPU)
    device: str = "auto"
    log_every: int = 50
    val_every: int = 1000
    val_batches: int = 20
    checkpoint_every: int = 5000
    dev_eval_every: int = 0  # 0 disables in-training rollouts on the dev set
    dev_eval_episodes: int = 20
    dev_eval_envs: int = 10
    run_name: str | None = None  # default: <method>/<dataset>_<subset>/seed<seed>
    data_root: str | None = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - legacy global RNG, seeded for third-party code
    torch.manual_seed(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)


def cosine_with_warmup(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def fused_adamw_works(device: torch.device) -> bool:
    """The fused AdamW kernel is a speed-up on accelerators; fall back silently if absent."""
    if device.type not in ("xpu", "cuda"):
        return False
    try:
        p = torch.nn.Parameter(torch.zeros(4, device=device))
        opt = torch.optim.AdamW([p], lr=1e-3, fused=True)
        p.grad = torch.ones_like(p)
        opt.step()
        return True
    except (RuntimeError, ValueError, NotImplementedError):
        return False


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def run_dir_for(paths: Paths, method: str, cfg: TrainConfig) -> Path:
    if cfg.run_name:
        return paths.runs / cfg.run_name
    subset = "all" if cfg.subset is None else str(cfg.subset)
    return paths.runs / method / f"{cfg.dataset}_{cfg.obs_mode}_{subset}" / f"seed{cfg.seed}"


def save_checkpoint(
    path: Path,
    method: str,
    policy,
    ema: EMAModel | None,
    policy_cfg,
    train_cfg: TrainConfig,
    step: int,
    extra: dict,
) -> None:
    payload = {
        "method": method,
        "policy_config": asdict(policy_cfg),
        "train_config": asdict(train_cfg),
        "state_dict": policy.state_dict(),
        "ema_state_dict": ema.model.state_dict() if ema is not None else None,
        "step": step,
        **extra,
    }
    torch.save(payload, path)


def train(
    method: str, policy, policy_cfg, cfg: TrainConfig, window_cfg: WindowConfig, dev_eval_fn=None
) -> Path:
    """Train ``policy`` (a BCPolicy or DiffusionPolicy) and return the run directory."""
    seed_everything(cfg.seed)
    paths = Paths.resolve(cfg.data_root)
    dataset = Dataset.open(paths, cfg.dataset)
    splits = load_splits(dataset)
    train_ids = subset_ids(splits.train, cfg.subset, cfg.seed)
    val_ids = list(splits.val)

    started = time.perf_counter()
    train_set = WindowDataset(load_episodes(dataset, train_ids, window_cfg), window_cfg)
    val_set = WindowDataset(load_episodes(dataset, val_ids, window_cfg), window_cfg)
    load_seconds = time.perf_counter() - started
    policy.fit_normalizers(train_set.all_lowdim(), train_set.all_actions())

    device = resolve_device(cfg.device, paths)
    policy.to(device)
    use_autocast = cfg.bf16 and device.type in ("xpu", "cuda")
    ema = EMAModel(policy, power=cfg.ema_power, max_decay=cfg.ema_max_decay) if cfg.ema else None
    fused = fused_adamw_works(device)
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas, fused=fused
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: cosine_with_warmup(s, cfg.warmup_steps, cfg.steps)
    )

    run_dir = run_dir_for(paths, method, cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    commit, dirty = git_state()
    run_info: dict[str, Any] = {
        "method": method,
        "dataset": cfg.dataset,
        "dataset_source": dataset.metadata.source,
        "train_episodes": train_ids,
        "val_episodes": val_ids,
        "n_train_windows": len(train_set),
        "device": device_summary(device),
        "git_commit": commit,
        "git_dirty": dirty,
        "parameters": sum(p.numel() for p in policy.parameters()),
    }
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "train": asdict(cfg),
                "policy": asdict(policy_cfg),
                "window": asdict(window_cfg),
                **run_info,
            },
            indent=1,
            default=str,
        ),
        encoding="utf-8",
    )
    writer = SummaryWriter(str(run_dir / "tb"))
    print(
        f"{method}: {len(train_ids)} train episodes ({len(train_set)} windows), {len(val_ids)} val, {run_info['parameters'] / 1e6:.1f} M params, {run_info['device']}, loaded in {load_seconds:.0f} s"
    )
    print(f"run dir: {run_dir}")

    generator = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=generator,
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=0
    )
    iterator = iter(loader)
    policy.train()
    losses: list[float] = []
    t_train = time.perf_counter()
    t_log = t_train
    for step in range(1, cfg.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = to_device(batch, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_autocast):
            loss = policy.compute_loss(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        decay = ema.step(policy) if ema is not None else 0.0
        losses.append(loss.item())

        if step % cfg.log_every == 0 or step == cfg.steps:
            now = time.perf_counter()
            ms = (now - t_log) * 1000 / cfg.log_every
            t_log = now
            mean_loss = float(np.mean(losses[-cfg.log_every :]))
            writer.add_scalar("train/loss", mean_loss, step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
            writer.add_scalar("train/ms_per_step", ms, step)
            if ema is not None:
                writer.add_scalar("train/ema_decay", decay, step)
            if step % (cfg.log_every * 10) == 0 or step == cfg.steps:
                print(
                    f"  step {step}/{cfg.steps} loss {mean_loss:.4f} lr {scheduler.get_last_lr()[0]:.2e} {ms:.0f} ms/step",
                    flush=True,
                )

        if cfg.val_every and (step % cfg.val_every == 0 or step == cfg.steps):
            eval_policy = ema.model if ema is not None else policy
            eval_policy.eval()
            with torch.no_grad():
                val_losses = []
                for i, val_batch in enumerate(val_loader):
                    if i >= cfg.val_batches:
                        break
                    val_batch = to_device(val_batch, device)
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_autocast):
                        val_losses.append(float(eval_policy.compute_loss(val_batch)))
            writer.add_scalar("val/loss", float(np.mean(val_losses)), step)
            policy.train()

        if (
            cfg.dev_eval_every
            and dev_eval_fn is not None
            and (step % cfg.dev_eval_every == 0 or step == cfg.steps)
        ):
            eval_policy = ema.model if ema is not None else policy
            eval_policy.eval()
            success_rate = dev_eval_fn(eval_policy, step)
            writer.add_scalar("dev/success_rate", success_rate, step)
            print(f"  step {step}: dev-set success {success_rate:.0%}", flush=True)
            policy.train()

        if step % cfg.checkpoint_every == 0 or step == cfg.steps:
            extra = {**run_info, "elapsed_s": time.perf_counter() - t_train}
            save_checkpoint(
                run_dir / f"step_{step}.pt", method, policy, ema, policy_cfg, cfg, step, extra
            )
            save_checkpoint(run_dir / "last.pt", method, policy, ema, policy_cfg, cfg, step, extra)

    elapsed = time.perf_counter() - t_train
    summary = {
        **run_info,
        "steps": cfg.steps,
        "final_train_loss": float(np.mean(losses[-cfg.log_every :])),
        "train_seconds": elapsed,
        "ms_per_step": elapsed * 1000 / cfg.steps,
        "load_seconds": load_seconds,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=1, default=str), encoding="utf-8"
    )
    writer.close()
    print(
        f"done: {cfg.steps} steps in {elapsed / 60:.1f} min ({summary['ms_per_step']:.0f} ms/step), final loss {summary['final_train_loss']:.4f}"
    )
    return run_dir
