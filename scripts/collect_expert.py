"""Evaluate the scripted expert, or record a scripted dataset with it.

    uv run python scripts/collect_expert.py --evaluate 50                 # success rate on 50 random cube poses
    uv run python scripts/collect_expert.py --evaluate-set eval_set_v1    # on a committed held-out set
    uv run python scripts/collect_expert.py --name scripted_v1 --n 220    # record a dataset (source=scripted)
    uv run python scripts/collect_expert.py --name scripted_kb_v1 --n 220 --quantize keyboard

Seeds for a recorded dataset come from the named RNG stream
default_rng([20261031, <stream>]), separate from the held-out sets' streams.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from panda_il.config import EnvConfig
from panda_il.data import EpisodeRecorder, new_dataset
from panda_il.data.validate import validate_dataset
from panda_il.env import make_env, sets
from panda_il.env.task import InitialCondition
from panda_il.paths import Paths
from panda_il.teleop.expert import ExpertConfig, ScriptedExpert


def run_episode(
    env,
    expert: ScriptedExpert,
    seed: int,
    ic: InitialCondition | None = None,
    recorder: EpisodeRecorder | None = None,
):
    if recorder is not None:
        obs, info = recorder.start(seed=seed, ic=ic)
    else:
        obs, info = env.reset(seed=seed, options={"ic": ic} if ic is not None else None)
    expert.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        action = expert.act_from_env(obs, info)
        if recorder is not None:
            obs, _, terminated, truncated, info = recorder.step(action)
        else:
            obs, _, terminated, truncated, info = env.step(action)
    return bool(info["success"]), env.step_count, expert.phase_log


def evaluate(args) -> int:
    env = make_env(EnvConfig(obs_mode="state"))
    expert = ScriptedExpert(env.cfg, ExpertConfig(quantize=args.quantize))
    if args.evaluate_set:
        ic_set = sets.load_set(args.evaluate_set)
        cases = [(1000 + i, ic) for i, ic in enumerate(ic_set.ics)]
        label = args.evaluate_set
    else:
        rng = np.random.default_rng([20261031, 99])
        cases = [(int(rng.integers(0, 2**31 - 1)), None) for _ in range(args.evaluate)]
        label = f"{args.evaluate} random poses"
    successes, lengths, failures = 0, [], []
    started = time.perf_counter()
    for seed, ic in cases:
        ok, steps, phases = run_episode(env, expert, seed, ic)
        successes += int(ok)
        lengths.append(steps)
        if not ok:
            failures.append((seed, ic, phases[-1][0], steps))
    env.close()
    rate = successes / len(cases)
    print(
        f"expert{' (keyboard-quantised)' if args.quantize else ''} on {label}: {successes}/{len(cases)} = {rate:.1%} success, "
        f"length mean {np.mean(lengths):.1f} min {min(lengths)} max {max(lengths)} steps, {time.perf_counter() - started:.1f} s"
    )
    for seed, ic, phase, steps in failures[:10]:
        print(f"  failure: seed {seed} ic {ic} stuck in phase {phase} after {steps} steps")
    return 0 if rate >= 0.9 else 1


def record(args) -> int:
    paths = Paths.resolve(args.data_root)
    paths.make_dirs()
    env = make_env(EnvConfig(obs_mode="both"))
    expert = ScriptedExpert(env.cfg, ExpertConfig(quantize=args.quantize))
    dataset = new_dataset(
        paths,
        args.name,
        env,
        source="scripted",
        operator=f"ScriptedExpert{'-keyboard' if args.quantize else ''}",
        input_device="script",
        notes=args.notes,
    )
    recorder = EpisodeRecorder(
        env,
        "scripted",
        dataset.metadata.operator,
        "script",
        dataset=dataset,
        rng_stream=f"scripted:{args.seed_stream}",
    )
    rng = np.random.default_rng([20261031, args.seed_stream])
    successes, lengths = 0, []
    started = time.perf_counter()
    for i in range(args.n):
        seed = int(rng.integers(0, 2**31 - 1))
        ok, steps, phases = run_episode(env, expert, seed, recorder=recorder)
        record = recorder.finish(
            extra_attrs={"expert_phases": phases, "expert_quantize": args.quantize or "none"}
        )
        successes += int(ok)
        lengths.append(steps)
        if (i + 1) % 10 == 0 or i + 1 == args.n:
            elapsed = time.perf_counter() - started
            print(
                f"  {i + 1}/{args.n} episodes, {successes} successful, mean length {np.mean(lengths):.1f}, "
                f"{elapsed / (i + 1):.1f} s/episode, episode {record.attrs['episode_id']} max align "
                f"{max(record.attrs['max_alignment_error_s'].values()) * 1000:.0f} ms",
                flush=True,
            )
    env.close()
    print(
        f"recorded {args.n} episodes into {dataset.directory}: {successes} successful ({successes / args.n:.1%}), "
        f"lengths {min(lengths)}-{max(lengths)} (mean {np.mean(lengths):.1f}), {time.perf_counter() - started:.0f} s"
    )
    report = validate_dataset(dataset, check_held_out=True)
    for issue in report.issues[:20]:
        print(issue)
    print(report.summary())
    return 0 if report.ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--evaluate", type=int, default=0, help="evaluate on N random cube poses")
    parser.add_argument(
        "--evaluate-set", default=None, help="evaluate on a committed IC set, e.g. eval_set_v1"
    )
    parser.add_argument("--name", default=None, help="record a dataset with this name")
    parser.add_argument("--n", type=int, default=220)
    parser.add_argument(
        "--seed-stream", type=int, default=10, help="RNG stream id for the episode seeds"
    )
    parser.add_argument("--quantize", choices=[None, "keyboard"], default=None)
    parser.add_argument("--notes", default="")
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    if args.name:
        return record(args)
    if args.evaluate or args.evaluate_set:
        return evaluate(args)
    parser.error("choose --evaluate N, --evaluate-set NAME or --name NAME")
    return 2


if __name__ == "__main__":
    sys.exit(main())
