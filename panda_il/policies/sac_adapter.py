"""Make a Stable-Baselines3 SAC model look like one of our policies to the evaluator.

The evaluator feeds a window ``{"lowdim": (B, To, 29)}`` and expects an action
chunk ``(B, Tp, 7)``; SAC is a single-step state policy, so ``To = Tp = Ta = 1``
and the adapter returns the deterministic action for the latest state.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
from torch import nn


class SACPolicyAdapter(nn.Module):
    def __init__(self, sb3_policy: nn.Module):
        super().__init__()
        self.policy = sb3_policy

    @torch.no_grad()
    def predict(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        obs = batch["lowdim"][:, -1].float().cpu().numpy()
        actions, _ = self.policy.predict(obs, deterministic=True)
        return torch.as_tensor(actions, dtype=torch.float32)[:, None, :]


def load_sac(path: str | Path, device: torch.device) -> tuple[SACPolicyAdapter, dict[str, Any]]:
    """Load ``<run dir>/<name>.zip`` written by ``train_sac``; the run's ``config.json`` and
    ``summary.json`` (if present) supply the provenance the evaluator reports."""
    from stable_baselines3 import SAC

    path = Path(path)
    model = SAC.load(path, device="cpu")
    run_dir = path.parent
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    match = re.search(r"_(\d+)_steps", path.stem)
    step = int(match.group(1)) if match else int(summary.get("total_steps", model.num_timesteps))
    adapter = SACPolicyAdapter(model.policy).to(device).eval()
    info = {
        "method": "sac",
        "step": step,
        "weights": "raw",
        "policy_config": {
            "encoder": {"obs_mode": "state", "obs_horizon": 1, "cameras": []},
            "pred_horizon": 1,
            "action_horizon": 1,
        },
        "train_config": {
            "dataset": None,
            "subset": None,
            "seed": config["train"]["seed"],
            "obs_mode": "state",
            "steps": config["train"]["total_steps"],
            "reward": config["train"]["reward"],
        },
        "dataset_source": None,
        "git_commit": config.get("git_commit"),
    }
    return adapter, info
