import numpy as np
import pytest

from panda_il.data.schema import IMAGE_CHUNK_FRAMES, SCHEMA_VERSION, Dataset, DatasetMetadata
from tests import synthetic as syn


def test_write_and_read_round_trip(tmp_path):
    dataset = Dataset.create(tmp_path / "ds", syn.synthetic_metadata())
    record = syn.synthetic_record(n_steps=40, seed=3, success=True)
    episode_id = dataset.write_episode(record)
    assert episode_id == 0
    assert dataset.episode_ids() == [0]
    assert dataset.metadata.counts == {"episodes": 1, "successes": 1, "discarded": 0}

    back = dataset.read_episode(0)
    assert back.attrs["episode_id"] == 0 and back.attrs["schema_version"] == SCHEMA_VERSION
    assert back.attrs["success"] is True and back.attrs["seed"] == 3
    assert back.attrs["ic"] == record.attrs["ic"]
    for stream, fields in record.raw.items():
        assert (
            back.raw_attrs[stream]["nominal_period_substeps"]
            == record.raw_attrs[stream]["nominal_period_substeps"]
        )
        for name, array in fields.items():
            assert np.array_equal(back.raw[stream][name], np.asarray(array)), (stream, name)
            assert back.raw[stream][name].dtype == np.asarray(array).dtype
    for name, array in record.synced.items():
        assert np.array_equal(back.synced[name], array), name
    for name, array in record.init.items():
        assert np.array_equal(back.init[name], array)


def test_images_are_chunked_by_frames_and_can_be_read_by_index(tmp_path):
    dataset = Dataset.create(tmp_path / "ds", syn.synthetic_metadata())
    dataset.write_episode(syn.synthetic_record(n_steps=50))
    array = dataset.root["ep_000000"]["raw"]["cam_front"]["image"]
    assert array.chunks == (IMAGE_CHUNK_FRAMES, syn.IMAGE_SIZE, syn.IMAGE_SIZE, 3)
    picked = dataset.read_images(0, "front", np.array([0, 7, 50]))
    assert picked.shape == (3, syn.IMAGE_SIZE, syn.IMAGE_SIZE, 3)
    assert picked[1, 0, 0, 0] == 7  # frame index stored in the red channel


def test_ids_increase_and_read_only_mode_refuses_writes(tmp_path):
    dataset = Dataset.create(tmp_path / "ds", syn.synthetic_metadata())
    dataset.write_episode(syn.synthetic_record())
    dataset.write_episode(syn.synthetic_record(seed=1))
    assert dataset.episode_ids() == [0, 1]
    reopened = Dataset(tmp_path / "ds")  # read-only
    assert reopened.episode_ids() == [0, 1]
    with pytest.raises(PermissionError):
        reopened.write_episode(syn.synthetic_record())
    with pytest.raises(FileExistsError):
        Dataset.create(tmp_path / "ds", syn.synthetic_metadata())


def test_metadata_round_trip(tmp_path):
    metadata = syn.synthetic_metadata(name="meta")
    text = metadata.to_json()
    back = DatasetMetadata.from_json(text)
    assert back == metadata
    Dataset.create(tmp_path / "meta", metadata)
    assert (tmp_path / "meta" / "metadata.json").exists()
    assert Dataset(tmp_path / "meta").metadata.name == "meta"
