import mink
import numpy as np
import pytest

from panda_il.config import EnvConfig, IKConfig
from panda_il.env import make_env
from panda_il.env.ik import apply_delta, make_ik, pose_error, pose_from_vector, pose_to_vector
from panda_il.env.scene import compile_arm_model

SQUARE = ([1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0])


@pytest.mark.parametrize("backend", ["mink", "dls"])
def test_kinematic_accuracy_along_a_square(backend):
    """The IK reference itself must land within 0.5 mm / 2 mrad of every 1 cm target."""
    ik = make_ik(compile_arm_model(), IKConfig(backend=backend))
    ik.reset(np.array([0, 0, 0, -1.5708, 0, 1.5708, -0.7853, 0.04, 0.04]))
    target = ik.tcp_pose()
    worst_pos = worst_rot = 0.0
    for direction in SQUARE:
        for _ in range(10):
            target = mink.SE3.from_translation(0.01 * np.asarray(direction, dtype=float)) @ target
            ik.solve(target, 0.05)
            pos_err, rot_err = ik.error(target)
            worst_pos, worst_rot = max(worst_pos, pos_err), max(worst_rot, rot_err)
    assert worst_pos < 5e-4, worst_pos
    assert worst_rot < 2e-3, worst_rot


@pytest.mark.parametrize("backend", ["mink", "dls"])
def test_physical_tracking_along_a_square(backend):
    """After five hold steps at each corner the measured TCP is within 2 mm of the command,
    and the lag while moving at full speed stays well below the resync threshold."""
    cfg = EnvConfig()
    cfg.ik.backend = backend
    env = make_env(cfg)
    env.reset(seed=0)
    worst_settled = worst_moving = 0.0
    for direction in SQUARE + ([0, 0, -1], [0, 0, 1]):
        for _ in range(10):
            action = np.zeros(7)
            action[:3] = direction
            _, _, _, _, info = env.step(action)
            worst_moving = max(worst_moving, info["tracking_lag"][0])
        for _ in range(5):
            _, _, _, _, info = env.step(np.zeros(7))
        settled = np.linalg.norm(info["tcp_measured"][:3] - info["tcp_target"][:3])
        worst_settled = max(worst_settled, settled)
    assert worst_settled < 2e-3, worst_settled
    assert worst_moving < cfg.ik.resync_position_m / 2, worst_moving
    assert info["resync_count"] == 0
    env.close()


def test_reference_resyncs_when_the_arm_is_blocked():
    env = make_env(EnvConfig())
    env.reset(seed=0)
    env.step(np.zeros(7))
    # Simulate a blocked arm: push the open-loop reference 10 cm away from where the arm is.
    blocked_reference = mink.SE3.from_translation(np.array([0.0, 0.0, -0.10])) @ env.ik.tcp_pose()
    env.ik.solve(blocked_reference, 0.05)
    env.ik.solve(blocked_reference, 0.05)
    _, _, _, _, info = env.step(np.zeros(7))
    assert info["resync_count"] == 1
    # After the resync the commanded pose sits on the measured one.
    assert np.linalg.norm(info["tcp_target"][:3] - info["tcp_measured"][:3]) < 0.02
    env.close()


def test_apply_delta_translates_in_world_and_rotates_locally():
    pose = mink.SE3.from_rotation_and_translation(
        mink.SO3.from_x_radians(np.pi), np.array([0.5, 0.0, 0.5])
    )
    moved = apply_delta(pose, [0.01, 0.02, -0.03], [0, 0, 0], (-1, -1, -1), (1, 1, 1))
    assert np.allclose(moved.translation(), [0.51, 0.02, 0.47])
    assert np.allclose(moved.rotation().wxyz, pose.rotation().wxyz)
    rotated = apply_delta(pose, [0, 0, 0], [0, 0, 0.1], (-1, -1, -1), (1, 1, 1))
    expected = pose.rotation() @ mink.SO3.from_z_radians(0.1)
    assert np.allclose(rotated.rotation().wxyz, expected.wxyz)
    clamped = apply_delta(pose, [1.0, 0, 0], [0, 0, 0], (-1, -1, -1), (0.75, 1, 1))
    assert clamped.translation()[0] == 0.75


def test_pose_vector_round_trip_and_error():
    pose = mink.SE3.from_rotation_and_translation(
        mink.SO3.from_rpy_radians(0.1, -0.2, 0.3), np.array([0.4, 0.1, 0.3])
    )
    vector = pose_to_vector(pose)
    assert vector.shape == (7,)
    back = pose_from_vector(vector)
    assert np.allclose(back.translation(), pose.translation()) and np.allclose(
        back.rotation().wxyz, pose.rotation().wxyz
    )
    dpos, drot = pose_error(
        pose,
        pose.translation() + [0.003, 0.004, 0.0],
        (pose.rotation() @ mink.SO3.from_z_radians(0.05)).wxyz,
    )
    assert abs(dpos - 0.005) < 1e-9 and abs(drot - 0.05) < 1e-6
