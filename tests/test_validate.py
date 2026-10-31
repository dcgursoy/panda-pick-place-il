import copy

import numpy as np

from panda_il.data.schema import Dataset
from panda_il.data.validate import validate_dataset, validate_episode
from panda_il.env import sets
from tests import synthetic as syn

META = syn.synthetic_metadata()


def issues_of(record, held_out=None):
    issues = validate_episode(record, META, held_out)
    return [str(i) for i in issues]


def errors_of(record, held_out=None):
    return [m for m in issues_of(record, held_out) if m.startswith("[error]")]


def test_valid_synthetic_episode_passes():
    record = syn.synthetic_record(n_steps=20, success=True)
    assert issues_of(record) == []


def test_nan_is_reported():
    record = syn.synthetic_record()
    record.raw["proprio"]["qpos"][3, 2] = np.nan
    messages = errors_of(record)
    assert any("raw/proprio/qpos" in m and "NaN" in m for m in messages)


def test_dropped_camera_frame_is_reported_as_a_gap():
    record = syn.synthetic_record(n_steps=12)
    cam = record.raw["cam_front"]
    keep = np.arange(len(cam["substep"])) != 5
    for name in cam:
        cam[name] = cam[name][keep]
    messages = errors_of(record)
    assert any("raw/cam_front" in m and "dropped frames" in m for m in messages)


def test_non_monotonic_timestamps_are_reported():
    record = syn.synthetic_record()
    record.raw["proprio"]["substep"][4] = record.raw["proprio"]["substep"][3]
    messages = errors_of(record)
    assert any("raw/proprio" in m and "strictly increasing" in m for m in messages)


def test_action_out_of_bounds_is_reported():
    record = syn.synthetic_record()
    record.raw["action"]["action"][2, 0] = 1.5
    assert any("outside [-1, 1]" in m for m in errors_of(record))


def test_source_mismatch_is_reported():
    record = syn.synthetic_record(source="human")
    assert any("differs from dataset source" in m for m in errors_of(record))


def test_tampered_synced_array_is_reported():
    record = syn.synthetic_record()
    record.synced["action"][0, 0] += 0.1
    assert any("synced/action" in m and "fresh sync" in m for m in errors_of(record))


def test_observation_disagreeing_with_raw_streams_is_reported():
    record = syn.synthetic_record()
    record.raw["obs"]["proprio"][5, 0] += 0.01
    assert any("sync/proprio" in m for m in errors_of(record))


def test_executed_action_disagreeing_with_commands_is_reported():
    record = syn.synthetic_record()
    record.raw["priv"]["action_executed"][3] = 0.0
    assert any("sync/action" in m for m in errors_of(record))


def test_missing_stream_and_wrong_schema_version():
    record = syn.synthetic_record()
    del record.raw["cam_wrist"]
    record.attrs["schema_version"] = 99
    messages = errors_of(record)
    assert any("missing stream 'cam_wrist'" in m for m in messages)
    assert any("schema_version 99" in m for m in messages)


def test_success_flag_must_match_the_final_streak():
    record = syn.synthetic_record(success=False)
    record.attrs["success"] = True
    assert any("attrs/success" in m for m in errors_of(record))


def test_held_out_proximity_warns():
    eval_set = sets.generate_set("eval_set_v1")
    near = eval_set.ics[0]
    record = syn.synthetic_record(ic=(near.x + 0.001, near.y, near.yaw + 0.01))
    far = syn.synthetic_record(ic=(near.x + 0.05, near.y, near.yaw))
    assert any("[warning]" in m and "eval_set_v1" in m for m in issues_of(record, [eval_set]))
    assert issues_of(far, [eval_set]) == []


def test_dataset_level_report(tmp_path):
    dataset = Dataset.create(tmp_path / "ds", syn.synthetic_metadata())
    dataset.write_episode(syn.synthetic_record(n_steps=20, seed=0, success=True))
    dataset.write_episode(syn.synthetic_record(n_steps=15, seed=1))
    report = validate_dataset(Dataset(tmp_path / "ds"), check_held_out=False)
    assert report.ok and report.episodes == 2 and report.successes == 1
    assert report.lengths == [20, 15]
    assert report.max_alignment_s["cam_front"] == 10 * syn.PHYSICS_DT
    # corrupt one episode on disk and check the report catches it
    bad = copy.deepcopy(dataset.read_episode(1))
    bad.raw["action"]["action"][0, 0] = 7.0
    corrupted = Dataset.create(tmp_path / "bad", syn.synthetic_metadata())
    corrupted.write_episode(bad)
    report = validate_dataset(Dataset(tmp_path / "bad"), check_held_out=False)
    assert not report.ok and report.errors >= 1
