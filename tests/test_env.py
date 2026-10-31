import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from panda_il.config import EnvConfig
from panda_il.env import PROPRIO_DIM, STATE_DIM, InitialCondition, make_env
from panda_il.env.scene import (
    BIN_CENTER,
    BIN_FLOOR_TOP,
    CUBE_HALF,
    GRIPPER_CLOSED_CTRL,
    GRIPPER_OPEN_CTRL,
)


def random_actions(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-1, 1, size=(n, 7))


def rollout(env, seed: int, actions: np.ndarray):
    obs, info = env.reset(seed=seed)
    observations = [obs]
    infos = [info]
    for action in actions:
        obs, _, terminated, truncated, info = env.step(action)
        observations.append(obs)
        infos.append(info)
        if terminated or truncated:
            break
    return observations, infos


def test_state_space_and_gymnasium_checker():
    env = make_env(EnvConfig(obs_mode="state"))
    check_env(env, skip_render_check=True)
    obs, info = env.reset(seed=0)
    assert obs["state"].shape == (STATE_DIM,) and obs["state"].dtype == np.float32
    assert set(obs) == {"state"}
    assert info["sim_state"].shape == (env.state_size,)
    env.close()


def test_same_seed_and_actions_give_bitwise_identical_rollouts():
    actions = random_actions(60)
    env_a, env_b = make_env(EnvConfig()), make_env(EnvConfig())
    obs_a, infos_a = rollout(env_a, 123, actions)
    obs_b, infos_b = rollout(env_b, 123, actions)
    assert len(obs_a) == len(obs_b) == 61
    for a, b in zip(obs_a, obs_b, strict=True):
        assert np.array_equal(a["state"], b["state"])
    assert np.array_equal(infos_a[-1]["sim_state"], infos_b[-1]["sim_state"])
    assert infos_a[-1]["ic"] == infos_b[-1]["ic"]
    env_a.close()
    env_b.close()


def test_different_seeds_give_different_initial_conditions():
    env = make_env(EnvConfig())
    _, info_a = env.reset(seed=1)
    _, info_b = env.reset(seed=2)
    assert info_a["ic"] != info_b["ic"]
    env.close()


def test_explicit_initial_condition_is_applied():
    env = make_env(EnvConfig())
    ic = InitialCondition(0.45, -0.2, 0.3)
    obs, info = env.reset(seed=0, options={"ic": ic})
    assert info["ic"] == ic.to_dict()
    cube_pos = obs["state"][PROPRIO_DIM : PROPRIO_DIM + 3]
    assert np.allclose(cube_pos, [0.45, -0.2, CUBE_HALF], atol=1e-6)
    obs2, _ = env.reset(seed=0, options={"ic": ic.to_dict()})
    assert np.array_equal(obs["state"], obs2["state"])
    env.close()


def test_observations_are_fresh_arrays():
    env = make_env(EnvConfig())
    obs1, _ = env.reset(seed=0)
    obs2, *_ = env.step(np.zeros(7))
    obs3, *_ = env.step(np.zeros(7))
    assert obs1["state"] is not obs2["state"] and obs2["state"] is not obs3["state"]
    env.close()


def test_truncated_at_the_step_limit_and_not_terminated():
    env = make_env(EnvConfig(max_steps=12))
    env.reset(seed=0)
    outcome = None
    for step in range(1, 13):
        _, _, terminated, truncated, info = env.step(np.zeros(7))
        assert not terminated
        outcome = (step, truncated)
    assert outcome == (12, True)
    assert not info["success"]
    env.close()


def test_terminated_after_the_cube_rests_in_the_bin():
    cfg = EnvConfig(success_steps=3, max_steps=100)
    env = make_env(cfg)
    env.reset(seed=0)
    # Privileged: drop the cube just above the bin floor and let it settle.
    start = env.ids.cube_qpos
    env.data.qpos[start : start + 7] = [
        BIN_CENTER[0],
        BIN_CENTER[1],
        BIN_FLOOR_TOP + CUBE_HALF + 0.002,
        1,
        0,
        0,
        0,
    ]
    env.data.qvel[env.ids.cube_dof : env.ids.cube_dof + 6] = 0
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        _, reward, terminated, truncated, info = env.step(np.zeros(7))
        steps += 1
    assert terminated and not truncated
    assert (
        info["success"]
        and info["success_detail"]["on_floor"]
        and not info["success_detail"]["held"]
    )
    assert steps <= 30
    assert reward > cfg.reward.success * 0.5  # the sparse bonus is paid on the terminal step
    env.close()


def test_actions_are_clipped_and_the_workspace_is_clamped():
    cfg = EnvConfig()
    env = make_env(cfg)
    env.reset(seed=0)
    for _ in range(60):
        _, _, _, _, info = env.step(np.array([10.0, 0, 0, 0, 0, 0, 0]))
    assert info["tcp_target"][0] <= cfg.workspace_max[0] + 1e-9
    assert info["tcp_target"][0] > 0.7
    env.close()


def test_gripper_command_maps_to_actuator_ctrl():
    env = make_env(EnvConfig())
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.array([0, 0, 0, 0, 0, 0, 1.0]))
    assert info["ctrl"][env.ids.gripper_actuator] == GRIPPER_CLOSED_CTRL and info["gripper_closed"]
    _, _, _, _, info = env.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))
    assert (
        info["ctrl"][env.ids.gripper_actuator] == GRIPPER_OPEN_CTRL and not info["gripper_closed"]
    )
    env.close()


def test_snapshot_and_restore_reproduce_the_trajectory_exactly():
    env = make_env(EnvConfig())
    actions = random_actions(30, seed=3)
    env.reset(seed=7)
    for action in actions[:10]:
        env.step(action)
    snap = env.snapshot()
    continuation = [env.step(a)[0]["state"] for a in actions[10:]]
    env.restore(snap)
    replay = [env.step(a)[0]["state"] for a in actions[10:]]
    for a, b in zip(continuation, replay, strict=True):
        assert np.array_equal(a, b)
    env.close()


@pytest.mark.render
def test_image_mode_spaces_and_frame_ages():
    cfg = EnvConfig(obs_mode="image")
    env = make_env(cfg)
    obs, info = env.reset(seed=0)
    assert set(obs) == {"proprio", "image_front", "image_wrist"}
    assert obs["proprio"].shape == (PROPRIO_DIM,)
    for name in ("front", "wrist"):
        image = obs[f"image_{name}"]
        assert image.shape == (96, 96, 3) and image.dtype == np.uint8
        assert info["frame_age_substeps"][name] == 0
    first = obs["image_front"].copy()
    for _ in range(5):
        obs, _, _, _, info = env.step(np.array([1.0, 0, 0, 0, 0, 0, 0]))
        assert info["frame_age_substeps"] == {"front": 10, "wrist": 10}
    assert not np.array_equal(first, obs["image_front"])
    assert not np.array_equal(obs["image_front"], obs["image_wrist"])
    env.close()


@pytest.mark.render
def test_image_rollouts_are_deterministic():
    actions = random_actions(20, seed=5)
    env_a, env_b = make_env(EnvConfig(obs_mode="image")), make_env(EnvConfig(obs_mode="image"))
    obs_a, infos_a = rollout(env_a, 11, actions)
    obs_b, infos_b = rollout(env_b, 11, actions)
    # The simulator state is bitwise identical; the rendered pixels are too, except that this
    # iGPU occasionally differs in a handful of edge pixels (measured: 3 of 9216 pixels in 1 of
    # 21 wrist frames while another process was rendering). Allow that, nothing more.
    assert np.array_equal(infos_a[-1]["sim_state"], infos_b[-1]["sim_state"])
    for a, b in zip(obs_a, obs_b, strict=True):
        for key in ("image_front", "image_wrist"):
            differing = np.any(a[key] != b[key], axis=-1).mean()
            assert differing <= 1e-3, (key, differing)
    env_a.close()
    env_b.close()


@pytest.mark.render
def test_render_returns_a_third_person_frame():
    env = make_env(EnvConfig(render_width=320, render_height=240), render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame.shape == (240, 320, 3) and frame.dtype == np.uint8
    env.close()
