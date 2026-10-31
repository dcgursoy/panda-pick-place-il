"""Train behavior cloning (chunked by default; --policy.pred-horizon 1 for the single-step ablation).

uv run python -m panda_il.train.train_bc --train.dataset scripted_v1 --train.obs-mode state --train.seed 0
uv run python -m panda_il.train.train_bc --train.obs-mode image --train.batch-size 64 --train.subset 50
"""

from __future__ import annotations

from dataclasses import dataclass, field

import tyro

from panda_il.data.dataset import WindowConfig
from panda_il.policies.bc import BCConfig, BCPolicy
from panda_il.train.loop import TrainConfig, train


@dataclass
class Args:
    train: TrainConfig = field(default_factory=TrainConfig)
    policy: BCConfig = field(default_factory=BCConfig)


def main(args: Args) -> None:
    args.policy.encoder.obs_mode = args.train.obs_mode
    args.policy.encoder.lowdim_dim = 29 if args.train.obs_mode == "state" else 22
    window = WindowConfig(
        obs_mode=args.train.obs_mode,
        obs_horizon=args.policy.encoder.obs_horizon,
        pred_horizon=max(args.policy.pred_horizon, 1),
        action_horizon=min(args.policy.action_horizon, max(args.policy.pred_horizon, 1)),
        cameras=args.policy.encoder.cameras,
    )
    dev_eval = None
    if args.train.dev_eval_every:
        from panda_il.eval.evaluate import make_dev_eval

        dev_eval = make_dev_eval(args.train, window)
    train("bc", BCPolicy(args.policy), args.policy, args.train, window, dev_eval)


if __name__ == "__main__":
    main(tyro.cli(Args))
