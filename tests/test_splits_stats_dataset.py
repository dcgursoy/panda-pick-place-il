import numpy as np
import pytest

from panda_il.data import splits as splits_mod
from panda_il.data.dataset import WindowConfig, WindowDataset, load_episodes
from panda_il.data.schema import Dataset
from panda_il.data.stats import compute_stats
from tests import synthetic as syn


@pytest.fixture
def dataset(tmp_path):
    ds = Dataset.create(tmp_path / "ds", syn.synthetic_metadata())
    for i in range(12):
        ds.write_episode(syn.synthetic_record(n_steps=20 + i, seed=i, success=(i % 4 != 3)))
    return Dataset(tmp_path / "ds")


def test_splits_use_only_successful_episodes_and_are_reproducible(dataset):
    a = splits_mod.make_splits(dataset, val_size=3, seed=0)
    b = splits_mod.make_splits(dataset, val_size=3, seed=0)
    assert a == b
    assert a.failed == [3, 7, 11]
    assert len(a.val) == 3 and len(a.train) == 6
    assert not set(a.train) & set(a.val) and not set(a.train + a.val) & set(a.failed)
    assert splits_mod.make_splits(dataset, val_size=3, seed=1).val != a.val
    path = splits_mod.save_splits(dataset, a)
    assert path.exists() and splits_mod.load_splits(dataset) == a
    with pytest.raises(ValueError):
        splits_mod.make_splits(dataset, val_size=9)


def test_subsets_are_nested_and_seed_dependent():
    train = list(range(100))
    small = splits_mod.subset_ids(train, 10, seed=0)
    large = splits_mod.subset_ids(train, 25, seed=0)
    assert set(small) <= set(large) and len(small) == 10 and len(large) == 25
    assert splits_mod.subset_ids(train, 10, seed=1) != small
    assert splits_mod.subset_ids(train, 500, seed=0) == train


def test_stats_shapes_and_values(dataset):
    stats = compute_stats(dataset, [0, 1, 2], image_episodes=2)
    assert len(stats["action"]["min"]) == 7 and len(stats["proprio"]["mean"]) == 22
    assert stats["steps"] == 20 + 21 + 22
    assert all(-1 <= v <= 1 for v in stats["action"]["min"] + stats["action"]["max"])
    assert len(stats["action_saturated_fraction"]) == 6
    assert set(stats["images"]) == {"front", "wrist"} and len(stats["images"]["front"]["mean"]) == 3


def test_window_dataset_padding_and_shapes(dataset):
    cfg = WindowConfig(obs_mode="image", obs_horizon=2, pred_horizon=16, action_horizon=8)
    episodes = load_episodes(dataset, [0, 1], cfg)
    assert episodes[0].lowdim.shape == (21, 22) and episodes[0].action.shape == (20, 7)
    assert episodes[0].images["front"].shape == (21, syn.IMAGE_SIZE, syn.IMAGE_SIZE, 3)
    ds = WindowDataset(episodes, cfg)
    # per episode: starts from -1 to T - 16 + 7 = T - 9 inclusive -> T - 7 windows
    assert len(ds) == (20 - 7) + (21 - 7)
    first = ds[0]
    assert first["lowdim"].shape == (2, 22) and first["action"].shape == (16, 7)
    assert first["image_front"].shape == (2, syn.IMAGE_SIZE, syn.IMAGE_SIZE, 3)
    assert np.array_equal(first["lowdim"][0], first["lowdim"][1])  # pad_before repeats frame 0
    last = ds[(20 - 7) - 1]  # last window of episode 0 starts at T - 9 = 11
    assert np.array_equal(
        last["action"][-7:], np.repeat(last["action"][-8:-7], 7, axis=0)
    )  # pad_after repeats the last action
    state_cfg = WindowConfig(obs_mode="state")
    state_ds = WindowDataset(load_episodes(dataset, [0], state_cfg), state_cfg)
    assert state_ds[0]["lowdim"].shape == (2, 29) and "image_front" not in state_ds[0]
