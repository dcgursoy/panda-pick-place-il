"""The recorder against the real environment (needs rendering, so local only)."""

import numpy as np
import pytest

from panda_il.config import EnvConfig
from panda_il.data import Dataset, EpisodeRecorder, env_config_from_metadata, new_dataset
from panda_il.data.render import episode_frames
from panda_il.data.replay import replay_episode
from panda_il.data.validate import validate_dataset
from panda_il.env import make_env
from panda_il.paths import Paths

pytestmark = pytest.mark.render


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("data")
    paths = Paths(tmp)
    paths.make_dirs()
    env = make_env(EnvConfig(obs_mode="both", max_steps=30))
    dataset = new_dataset(
        paths, "rec", env, source="scripted", operator="pytest", input_device="script"
    )
    recorder = EpisodeRecorder(env, "scripted", "pytest", "script", dataset=dataset)
    rng = np.random.default_rng(0)
    records = []
    for seed in (10, 11):
        recorder.start(seed=seed)
        for _ in range(30):
            _, _, terminated, truncated, _ = recorder.step(rng.uniform(-1, 1, 7))
            if terminated or truncated:
                break
        records.append(recorder.finish())
    recorder.start(seed=12)
    recorder.step(np.zeros(7))
    assert recorder.finish(discard="test") is None
    env.close()
    return paths, dataset, records


def test_stream_counts_follow_the_rates(recorded):
    _, _, records = recorded
    for record in records:
        n = record.length
        lengths = record.attrs["stream_lengths"]
        assert lengths["proprio"] == 5 * n + 1
        assert lengths["obs"] == n + 1 and lengths["priv"] == n + 1
        assert lengths["action"] == n
        assert lengths["cam_front"] == n + 1 and lengths["cam_wrist"] == n + 1
        assert np.all(np.diff(record.raw["proprio"]["substep"]) == 5)
        assert np.all(np.diff(record.raw["proprio"]["t_wall"]) >= 0)


def test_alignment_errors_are_the_designed_ones(recorded):
    _, _, records = recorded
    for record in records:
        errors = record.attrs["max_alignment_error_s"]
        assert errors["cam_front"] == pytest.approx(0.02) and errors["cam_wrist"] == pytest.approx(
            0.02
        )
        assert errors["proprio"] == 0 and errors["action"] == 0 and errors["priv"] == 0


def test_logged_images_are_the_ones_the_policy_saw(recorded):
    _, dataset, _ = recorded
    record = dataset.read_episode(0)
    index = record.synced["image_index_front"]
    logged_capture = record.raw["cam_front"]["capture_substep"][index]
    assert np.array_equal(logged_capture, record.raw["obs"]["frame_substep_front"])


def test_dataset_counts_and_validator(recorded):
    _, dataset, _ = recorded
    reopened = Dataset(dataset.directory)
    assert reopened.episode_ids() == [0, 1]
    assert reopened.metadata.counts == {"episodes": 2, "successes": 0, "discarded": 1}
    report = validate_dataset(reopened, check_held_out=True)
    assert report.ok, [str(i) for i in report.issues]
    assert env_config_from_metadata(reopened.metadata).max_steps == 30


def test_replay_reproduces_the_episode_bit_for_bit(recorded):
    _, dataset, _ = recorded
    cfg = env_config_from_metadata(dataset.metadata)
    cfg.obs_mode = "state"
    env = make_env(cfg)
    for episode_id in dataset.episode_ids():
        report = replay_episode(dataset.read_episode(episode_id, load_images=False), env)
        assert report.exact and report.first_divergence_step is None, str(report)
        assert report.success_logged == report.success_replayed
    env.close()


def test_render_frames_from_the_store(recorded):
    _, dataset, _ = recorded
    frames = episode_frames(dataset.read_episode(0), None, ("front", "wrist"))
    assert len(frames) == 31
    assert frames[0].shape == (384, 776, 3) and frames[0].dtype == np.uint8
