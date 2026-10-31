import numpy as np
import pytest

from panda_il.config import EnvConfig
from panda_il.env import make_env
from panda_il.teleop.expert import (
    PHASES,
    ExpertConfig,
    ScriptedExpert,
    hand_yaw,
    wrap_to_quarter_turn,
)


def run(env, expert, seed):
    obs, info = env.reset(seed=seed)
    expert.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        obs, _, terminated, truncated, info = env.step(expert.act_from_env(obs, info))
    return bool(info["success"]), env.step_count, expert.phase_log


@pytest.mark.parametrize("quantize", [None, "keyboard"])
def test_expert_solves_random_poses(quantize):
    env = make_env(EnvConfig(obs_mode="state"))
    expert = ScriptedExpert(env.cfg, ExpertConfig(quantize=quantize))
    for seed in range(5):
        success, steps, phases = run(env, expert, seed)
        assert success, (seed, phases)
        assert steps < 200
        names = [name for name, _ in phases]
        # phases in order, each entered once; the episode usually terminates (cube at rest)
        # during the retreat, before the expert's own "done" phase
        assert names == list(PHASES)[: len(names)] and len(names) >= 8, names
    env.close()


def test_keyboard_quantisation_only_emits_the_three_levels():
    env = make_env(EnvConfig(obs_mode="state"))
    expert = ScriptedExpert(env.cfg, ExpertConfig(quantize="keyboard"))
    obs, info = env.reset(seed=1)
    expert.reset()
    for _ in range(40):
        action = expert.act_from_env(obs, info)
        assert set(np.unique(action[:6])) <= {-1.0, 0.0, 1.0}
        assert action[6] in (-1.0, 1.0)
        obs, _, _, _, info = env.step(action)
    env.close()


def test_yaw_helpers():
    assert wrap_to_quarter_turn(np.pi / 2 + 0.1) == pytest.approx(0.1)
    assert wrap_to_quarter_turn(-np.pi / 4) == pytest.approx(-np.pi / 4)
    # a frame rotated by 180 degrees about x (gripper pointing down) has its y axis along -y: yaw -90 degrees
    assert hand_yaw(np.array([0.0, 1.0, 0.0, 0.0])) == pytest.approx(-np.pi / 2)
