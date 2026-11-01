"""Load a policy from a checkpoint written by the training loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from panda_il.policies.bc import BCConfig, BCPolicy
from panda_il.policies.diffusion.policy import DiffusionConfig, DiffusionPolicy
from panda_il.policies.obs_encoder import ObsEncoderConfig


def policy_config_from_dict(method: str, d: dict[str, Any]):
    d = dict(d)
    encoder_dict = dict(d.pop("encoder"))
    encoder_dict["cameras"] = tuple(encoder_dict["cameras"])
    encoder = ObsEncoderConfig(**encoder_dict)
    if method == "bc":
        return BCConfig(encoder=encoder, **d)
    if method == "diffusion":
        d["down_dims"] = tuple(d["down_dims"])
        return DiffusionConfig(encoder=encoder, **d)
    raise ValueError(f"unknown method {method!r}")


def build_policy(method: str, cfg):
    if method == "bc":
        return BCPolicy(cfg)
    if method == "diffusion":
        return DiffusionPolicy(cfg)
    raise ValueError(method)


def load_checkpoint(path: str | Path, device: torch.device, use_ema: bool = True):
    """Returns (policy in eval mode on ``device``, checkpoint payload without the weights).
    A ``.zip`` is a Stable-Baselines3 SAC model (``policies/sac_adapter.py``)."""
    if str(path).endswith(".zip"):
        from panda_il.policies.sac_adapter import load_sac

        return load_sac(path, device)
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    method = payload["method"]
    cfg = policy_config_from_dict(method, payload["policy_config"])
    policy = build_policy(method, cfg)
    state = (
        payload["ema_state_dict"]
        if use_ema and payload.get("ema_state_dict") is not None
        else payload["state_dict"]
    )
    policy.load_state_dict(state)
    policy.to(device).eval()
    info = {k: v for k, v in payload.items() if k not in ("state_dict", "ema_state_dict")}
    info["weights"] = "ema" if use_ema and payload.get("ema_state_dict") is not None else "raw"
    return policy, info
