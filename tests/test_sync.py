import numpy as np
import pytest

from panda_il.data.sync import latest_index_at_or_before, sync_episode
from tests import synthetic as syn

N = 12
TICKS = np.arange(N + 1) * syn.SUBSTEPS


def build(proprio_offset=0, cam_offset=-10, latency=8, drop_ticks=(), command_substeps=None):
    raw = {
        "proprio": syn.proprio_stream(N, offset=proprio_offset),
        "action": syn.action_stream(N, command_substeps),
    }
    raw["obs"], raw["priv"] = syn.tick_streams(N)
    for cam in syn.CAMS:
        raw[f"cam_{cam}"] = syn.camera_stream(N, cam_offset, latency, drop_ticks)
    return raw


def test_samples_exactly_at_ticks_are_reproduced_exactly():
    result = sync_episode(build(), N, syn.SUBSTEPS, syn.PHYSICS_DT, syn.CAMS)
    expected = np.concatenate(
        [
            syn.truth_qpos(TICKS),
            syn.truth_qvel(TICKS),
            syn.truth_gripper(TICKS)[:, None],
            syn.truth_tcp_pos(TICKS),
            syn.truth_tcp_quat(TICKS),
        ],
        axis=1,
    )
    assert np.allclose(result.arrays["proprio"], expected, atol=1e-12)
    assert result.alignment["proprio"]["max_substeps"] == 0
    assert result.arrays["state"].shape == (N + 1, 29)
    assert np.allclose(result.arrays["state"][:, 22:25], syn.truth_cube(TICKS)[0])


def test_offset_proprio_is_interpolated_with_a_known_error():
    # samples at substeps 2, 7, 12, ...: every tick is 2 or 3 substeps from a sample
    result = sync_episode(build(proprio_offset=2), N, syn.SUBSTEPS, syn.PHYSICS_DT, syn.CAMS)
    expected = syn.truth_qpos(TICKS)
    error = np.abs(result.arrays["proprio"][:, :7] - expected).max()
    # linear interpolation error bound for sin(0.004 s): (h^2 / 8) * max|f''| with h = 5 substeps
    bound = (5**2 / 8) * (0.004**2) * 0.5 + 1e-12
    assert error <= bound, (error, bound)
    assert result.alignment["proprio"]["max_substeps"] == 2
    assert result.alignment["proprio"]["max_s"] == pytest.approx(2 * syn.PHYSICS_DT)
    quats = result.arrays["proprio"][:, 18:22]
    assert np.allclose(np.linalg.norm(quats, axis=1), 1.0)


def test_cameras_deliver_the_latest_arrived_frame_and_report_its_age():
    result = sync_episode(build(), N, syn.SUBSTEPS, syn.PHYSICS_DT, syn.CAMS)
    for cam in syn.CAMS:
        assert np.array_equal(
            result.arrays[f"image_index_{cam}"], np.arange(N + 1)
        )  # frame k at tick k
        assert np.array_equal(
            result.arrays[f"frame_age_{cam}"], np.concatenate([[0], np.full(N, 10)])
        )
        assert result.alignment[f"cam_{cam}"]["max_s"] == pytest.approx(10 * syn.PHYSICS_DT)


def test_a_frame_arriving_after_the_tick_is_not_delivered_until_the_next_tick():
    # captured 10 substeps before the tick but arriving 12 later: 2 substeps after the tick
    result = sync_episode(build(latency=12), N, syn.SUBSTEPS, syn.PHYSICS_DT, syn.CAMS)
    index = result.arrays["image_index_front"]
    assert np.array_equal(index, np.concatenate([[0], np.arange(N)]))  # one frame behind
    assert result.arrays["frame_age_front"][1] == 25  # frame 0 (substep 0) at tick 25
    assert result.arrays["frame_age_front"][2] == 35  # frame 1 captured at 15, seen at tick 50


def test_arrival_exactly_at_the_tick_counts_as_arrived():
    result = sync_episode(
        build(cam_offset=-8, latency=8), N, syn.SUBSTEPS, syn.PHYSICS_DT, syn.CAMS
    )
    assert np.array_equal(result.arrays["image_index_front"], np.arange(N + 1))
    assert np.all(result.arrays["frame_age_front"][1:] == 8)


def test_dropped_camera_frame_holds_the_previous_one():
    result = sync_episode(build(drop_ticks=(5,)), N, syn.SUBSTEPS, syn.PHYSICS_DT, syn.CAMS)
    index = result.arrays["image_index_front"]
    assert index[5] == index[4]
    assert result.arrays["frame_age_front"][5] == 35  # the tick-4 frame, 25 substeps older
    assert index[6] == index[4] + 1  # the stream continues (its indices shifted by the drop)
    assert result.alignment["cam_front"]["max_substeps"] == 35


def test_actions_are_zero_order_held_from_irregular_commands():
    n = 4
    commands = np.array([0, 3, 30, 49, 50, 77])
    raw = {"proprio": syn.proprio_stream(n), "action": syn.action_stream(n, commands)}
    raw["obs"], raw["priv"] = syn.tick_streams(n)
    for cam in syn.CAMS:
        raw[f"cam_{cam}"] = syn.camera_stream(n)
    result = sync_episode(raw, n, syn.SUBSTEPS, syn.PHYSICS_DT, syn.CAMS)
    # tick 0 -> command 0 (substep 0); tick 25 -> command 1 (substep 3); tick 50 -> command 4 (50); tick 75 -> command 4 (77 is later)
    assert np.array_equal(result.arrays["action_index"], [0, 1, 4, 4])
    assert np.array_equal(result.arrays["action"], raw["action"]["action"][[0, 1, 4, 4]])
    assert (
        result.alignment["action"]["max_substeps"] == 25
    )  # tick 75 held the command from substep 50


def test_missing_command_before_the_first_step_is_an_error():
    raw = build(command_substeps=np.array([5, 30]))
    with pytest.raises(ValueError):
        sync_episode(raw, N, syn.SUBSTEPS, syn.PHYSICS_DT, syn.CAMS)


def test_latest_index_helper():
    samples = np.array([0, 10, 20, 30])
    assert np.array_equal(
        latest_index_at_or_before(samples, np.array([0, 5, 10, 25, 31])), [0, 0, 1, 2, 3]
    )
    assert latest_index_at_or_before(samples, np.array([-1]))[0] == -1


def test_quaternion_sync_keeps_the_raw_sign_at_ticks():
    """MuJoCo's quaternion is not sign-continuous; the synced value at a tick must equal
    the logged one exactly (what the policy sees), while between samples the lerp stays
    on the raw sample's side."""
    from panda_il.data.sync import interpolate_quaternion

    samples = np.arange(0, 101, 5)
    angles = np.linspace(0.0, 2.5, len(samples))
    quats = np.stack(
        [np.cos(angles / 2), np.sin(angles / 2), np.zeros_like(angles), np.zeros_like(angles)], 1
    )
    quats[7:12] *= -1  # a run of flipped signs, as the simulator produces them
    quats[15] *= -1
    ticks = np.array([0, 25, 35, 50, 60, 75, 100])
    out = interpolate_quaternion(samples, quats, ticks)
    on_samples = np.isin(ticks, samples)
    assert np.allclose(
        out[on_samples], quats[np.searchsorted(samples, ticks[on_samples])], atol=1e-12
    )
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
    ref = quats[np.searchsorted(samples, ticks, side="right") - 1]
    assert np.all(np.einsum("ij,ij->i", out, ref) > 0)
