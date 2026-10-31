import mujoco
import numpy as np
import pytest

from panda_il.config import CameraConfig, EnvConfig
from panda_il.env import make_env
from panda_il.env.cameras import CameraClock, CameraSet, camera_intrinsics, project_point
from panda_il.env.scene import compile_scene_model


def test_clock_schedule_in_integer_substeps():
    clock = CameraClock(period=25, capture_offset=-10, latency=8)
    assert clock.capture_substep(1) == 15 and clock.arrival_substep(1) == 23
    assert clock.capture_substep(2) == 40 and clock.arrival_substep(2) == 48
    assert clock.delivered_frame_age() == 10
    for tick in range(1, 50):
        assert clock.arrival_substep(tick) < tick * clock.period


def test_config_rejects_timing_ties_and_bad_offsets():
    with pytest.raises(ValueError):
        CameraConfig(capture_offset_substeps=-8, latency_substeps=8)
    with pytest.raises(ValueError):
        CameraConfig(capture_offset_substeps=-25, latency_substeps=0)
    with pytest.raises(ValueError):
        EnvConfig(substeps=20)  # camera period no longer matches the control period


def test_intrinsics_follow_the_fovy_formula():
    model = compile_scene_model()
    k = camera_intrinsics(model, "front", 96, 96)
    fovy = float(model.cam_fovy[model.camera("front").id])
    assert np.isclose(k["fx"], 48 / np.tan(np.deg2rad(fovy) / 2))
    assert k["fx"] == k["fy"] and k["cx"] == 48 and k["cy"] == 48


def test_projection_of_a_point_in_front_of_the_camera():
    model = compile_scene_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    u, v, depth = project_point(model, data, "front", [0.5, 0.0, 0.02], 96, 96)
    assert depth > 0
    assert 0 <= u <= 96 and 0 <= v <= 96  # the look-at target is inside the image
    assert abs(u - 48) < 3 and abs(v - 48) < 3


@pytest.mark.render
def test_frames_arrive_on_schedule_and_never_early():
    model = compile_scene_model()
    data = mujoco.MjData(model)
    cameras = CameraSet(model, CameraConfig())
    cameras.reset(data)
    assert cameras.ages(0) == {"front": 0, "wrist": 0}
    rendered_at = []
    for substep in range(1, 51):
        mujoco.mj_step(model, data)
        if cameras.maybe_capture(data, substep):
            rendered_at.append(substep)
        if substep == 22:
            assert (
                cameras.deliver(22)["front"].capture_substep == 0
            )  # captured at 15, arrives at 23
        if substep == 23:
            assert cameras.deliver(23)["front"].capture_substep == 15
    assert rendered_at == [15, 40]
    assert cameras.deliver(50)["front"].capture_substep == 40
    assert cameras.ages(50) == {"front": 10, "wrist": 10}
    cameras.close()


@pytest.mark.render
def test_delivered_frame_age_is_identical_across_reruns():
    env = make_env(EnvConfig(obs_mode="image"))
    ages = set()
    for seed in range(3):
        _, info = env.reset(seed=seed)
        assert info["frame_age_substeps"] == {"front": 0, "wrist": 0}
        for _ in range(30):
            _, _, _, _, info = env.step(np.zeros(7))
            ages.add(tuple(sorted(info["frame_age_substeps"].items())))
    assert ages == {(("front", 10), ("wrist", 10))}
    env.close()
