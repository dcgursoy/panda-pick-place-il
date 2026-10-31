import mujoco
import numpy as np

from panda_il.env import sets
from panda_il.env.scene import BIN_CENTER, BIN_FLOOR_TOP, CUBE_HALF, compile_scene_model, scene_ids
from panda_il.env.task import (
    InitialCondition,
    cube_resting_in_bin,
    ic_distance,
    sample_in_band,
    sample_in_square,
)


def test_square_and_band_sampling_stay_in_their_regions():
    rng = np.random.default_rng(0)
    center, half = (0.5, -0.15), 0.08
    for _ in range(500):
        ic = sample_in_square(rng, center, half)
        assert abs(ic.x - center[0]) <= half and abs(ic.y - center[1]) <= half
        assert -np.pi <= ic.yaw <= np.pi
    for _ in range(500):
        ic = sample_in_band(rng, center, half, half + 0.04)
        d = max(abs(ic.x - center[0]), abs(ic.y - center[1]))
        assert half < d <= half + 0.04


def test_ic_distance_respects_the_cube_symmetry():
    a = InitialCondition(0.5, -0.1, 0.1)
    b = InitialCondition(0.5, -0.1, 0.1 + np.pi / 2)
    assert ic_distance(a, b) == (0.0, 0.0) or abs(ic_distance(a, b)[1]) < 1e-12
    c = InitialCondition(0.53, -0.14, 0.1 + 0.2)
    dpos, dyaw = ic_distance(a, c)
    assert abs(dpos - 0.05) < 1e-12 and abs(dyaw - 0.2) < 1e-12


def _settle(model, data, substeps=100):
    for _ in range(substeps):
        mujoco.mj_step(model, data)


def test_success_predicate_cases():
    model = compile_scene_model()
    data = mujoco.MjData(model)
    ids = scene_ids(model)
    start = ids.cube_qpos

    mujoco.mj_resetDataKeyframe(model, data, ids.home_key)
    data.ctrl[ids.arm_actuators] = data.qpos[ids.arm_qpos]
    data.ctrl[ids.gripper_actuator] = 255
    _settle(model, data)
    ok, detail = cube_resting_in_bin(model, data, ids)
    assert not ok and not detail["inside_xy"]  # on the table, far from the bin

    data.qpos[start : start + 7] = [
        BIN_CENTER[0],
        BIN_CENTER[1],
        BIN_FLOOR_TOP + CUBE_HALF + 0.05,
        1,
        0,
        0,
        0,
    ]
    data.qvel[ids.cube_dof : ids.cube_dof + 6] = 0
    mujoco.mj_forward(model, data)
    ok, detail = cube_resting_in_bin(model, data, ids)
    assert not ok and detail["inside_xy"] and not detail["on_floor"]  # hovering above the floor

    _settle(model, data, 300)
    ok, detail = cube_resting_in_bin(model, data, ids)
    assert ok, detail  # dropped, settled on the bin floor, not held, still


def test_eval_sets_are_deterministic_disjoint_and_round_trip(tmp_path):
    first = sets.generate_set("eval_set_v1")
    second = sets.generate_set("eval_set_v1")
    assert first.ics == second.ics and len(first.ics) == 100
    dev = sets.generate_set("dev_set_v1")
    assert len(dev.ics) == 20
    dpos, _, _ = sets.nearest_in_sets(dev.ics[0], [first])
    assert dpos > 0.0
    band = sets.generate_set("robust_set_v1")
    for ic in band.ics:
        d = max(abs(ic.x - band.center[0]), abs(ic.y - band.center[1]))
        assert band.inner_half < d <= band.outer_half
    path = sets.save_set(first, tmp_path)
    loaded = sets.load_set("eval_set_v1", tmp_path)
    assert path.exists() and loaded.ics == first.ics and loaded.stream == first.stream
