"""Replay logged actions from the logged initial state and report whether the outcome reproduces.

    uv run python -m panda_il.data.replay --dataset scripted_v1 [--episodes 0 1 2]

MuJoCo is deterministic within one version on one machine, and the controller
(IK reference, commanded pose) is a pure function of its inputs, so re-running
the synced actions from the same reset must reproduce the privileged simulator
state at every tick bit for bit. The report gives the first tick at which the
replay diverges (if any), the largest state difference, and whether the success
flag and final cube pose match.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np

from panda_il.data.recorder import env_config_from_metadata
from panda_il.data.schema import Dataset, EpisodeRecord
from panda_il.env.panda_env import PandaPickPlaceEnv
from panda_il.env.task import InitialCondition
from panda_il.paths import Paths


@dataclass
class ReplayReport:
    episode_id: int
    steps: int
    exact: bool
    first_divergence_step: int | None
    max_state_abs_diff: float
    success_logged: bool
    success_replayed: bool
    final_cube_distance_m: float

    def __str__(self) -> str:
        verdict = (
            "EXACT"
            if self.exact
            else f"diverges at step {self.first_divergence_step} (max |diff| {self.max_state_abs_diff:.3e})"
        )
        return (
            f"episode {self.episode_id}: {self.steps} steps, {verdict}; success logged={self.success_logged} "
            f"replayed={self.success_replayed}; final cube distance {self.final_cube_distance_m * 1000:.2f} mm"
        )


def replay_episode(record: EpisodeRecord, env: PandaPickPlaceEnv) -> ReplayReport:
    attrs = record.attrs
    ic = InitialCondition.from_dict(attrs["ic"])
    env.reset(seed=int(attrs["seed"]), options={"ic": ic})
    logged_states = np.asarray(record.raw["priv"]["sim_state"])
    actions = np.asarray(record.synced["action"])
    n_steps = int(attrs["length"])

    first_divergence = None
    max_diff = float(np.max(np.abs(env.get_sim_state() - logged_states[0])))
    if max_diff > 0:
        first_divergence = 0
    info = None
    for k in range(n_steps):
        _, _, terminated, truncated, info = env.step(actions[k])
        diff = float(np.max(np.abs(env.get_sim_state() - logged_states[k + 1])))
        max_diff = max(max_diff, diff)
        if diff > 0 and first_divergence is None:
            first_divergence = k + 1
        if (terminated or truncated) and k + 1 < n_steps:
            first_divergence = first_divergence if first_divergence is not None else k + 1
            break
    success_replayed = bool(info["success"]) if info is not None else False
    cube_logged = np.asarray(record.raw["priv"]["cube_pos"])[-1]
    cube_replayed = np.asarray(info["cube_pos"]) if info is not None else cube_logged
    return ReplayReport(
        episode_id=int(attrs.get("episode_id", -1)),
        steps=n_steps,
        exact=first_divergence is None,
        first_divergence_step=first_divergence,
        max_state_abs_diff=max_diff,
        success_logged=bool(attrs["success"]),
        success_replayed=success_replayed,
        final_cube_distance_m=float(np.linalg.norm(cube_logged - cube_replayed)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--episodes", type=int, nargs="*", default=None, help="default: all")
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    dataset = Dataset.open(Paths.resolve(args.data_root), args.dataset)
    cfg = env_config_from_metadata(dataset.metadata)
    cfg.obs_mode = "state"
    env = PandaPickPlaceEnv(cfg)
    ids = args.episodes if args.episodes else dataset.episode_ids()
    exact = 0
    for episode_id in ids:
        report = replay_episode(dataset.read_episode(episode_id, load_images=False), env)
        print(report)
        exact += int(report.exact and report.success_logged == report.success_replayed)
    env.close()
    print(f"{exact}/{len(ids)} episodes reproduced exactly with the same outcome")
    return 0 if exact == len(ids) else 1


if __name__ == "__main__":
    sys.exit(main())
