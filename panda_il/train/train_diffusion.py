"""Train Diffusion Policy (state or image observations).

uv run python -m panda_il.train.train_diffusion --train.dataset scripted_v1 --train.obs-mode state --train.seed 0
uv run python -m panda_il.train.train_diffusion --train.obs-mode image --train.batch-size 64
uv run python -m panda_il.train.train_diffusion --policy.down-dims 256 512 1024 --policy.diffusion-step-embed-dim 256   # paper-size UNet
"""

from __future__ import annotations

from dataclasses import dataclass, field

import tyro

from panda_il.data.dataset import WindowConfig
from panda_il.policies.diffusion.policy import DiffusionConfig, DiffusionPolicy
from panda_il.train.loop import TrainConfig, train


@dataclass
class Args:
    train: TrainConfig = field(default_factory=TrainConfig)
    policy: DiffusionConfig = field(default_factory=DiffusionConfig)


def main(args: Args) -> None:
    args.policy.encoder.obs_mode = args.train.obs_mode
    args.policy.encoder.lowdim_dim = 29 if args.train.obs_mode == "state" else 22
    window = WindowConfig(
        obs_mode=args.train.obs_mode,
        obs_horizon=args.policy.encoder.obs_horizon,
        pred_horizon=args.policy.pred_horizon,
        action_horizon=args.policy.action_horizon,
        cameras=args.policy.encoder.cameras,
    )
    dev_eval = None
    if args.train.dev_eval_every:
        from panda_il.eval.evaluate import make_dev_eval

        dev_eval = make_dev_eval(args.train, window)
    train("diffusion", DiffusionPolicy(args.policy), args.policy, args.train, window, dev_eval)


if __name__ == "__main__":
    main(tyro.cli(Args))
