"""Train / validation splits by episode, and nested subsets for the demo-count curve.

    uv run python -m panda_il.data.splits --dataset scripted_v1 [--val 20] [--seed 0]

Only successful episodes are used for training or validation (DESIGN.md,
decision 8); failed ones are listed so they are visibly excluded. The validation
set is a fixed number of episodes drawn once with a seed and written to
``splits.json``. Demo-count subsets are nested prefixes of a shuffle of the
training episodes with the *training seed*, so different seeds also vary which
demonstrations are drawn (``subset_ids``).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from panda_il.data.schema import Dataset
from panda_il.paths import Paths


@dataclass
class Splits:
    train: list[int]
    val: list[int]
    failed: list[int]
    seed: int
    val_size: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)

    @classmethod
    def from_json(cls, text: str) -> Splits:
        return cls(**json.loads(text))


def splits_path(dataset: Dataset) -> Path:
    return dataset.directory / "splits.json"


def make_splits(dataset: Dataset, val_size: int = 20, seed: int = 0) -> Splits:
    successful, failed = [], []
    for episode_id in dataset.episode_ids():
        (successful if dataset.read_attrs(episode_id).get("success") else failed).append(episode_id)
    if len(successful) <= val_size:
        raise ValueError(f"only {len(successful)} successful episodes, cannot hold out {val_size}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(successful)
    val = sorted(int(i) for i in order[:val_size])
    train = sorted(int(i) for i in order[val_size:])
    return Splits(train=train, val=val, failed=failed, seed=seed, val_size=val_size)


def save_splits(dataset: Dataset, splits: Splits) -> Path:
    path = splits_path(dataset)
    path.write_text(splits.to_json(), encoding="utf-8")
    return path


def load_splits(dataset: Dataset) -> Splits:
    path = splits_path(dataset)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing; run `python -m panda_il.data.splits --dataset {dataset.name}`"
        )
    return Splits.from_json(path.read_text(encoding="utf-8"))


def subset_ids(train_ids: list[int], n_demos: int | None, seed: int) -> list[int]:
    """The first ``n_demos`` of a seeded shuffle of the training episodes (nested across sizes)."""
    ids = list(train_ids)
    if n_demos is None or n_demos >= len(ids):
        return sorted(ids)
    order = np.random.default_rng([seed, 7]).permutation(ids)
    return sorted(int(i) for i in order[:n_demos])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--val", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    dataset = Dataset.open(Paths.resolve(args.data_root), args.dataset)
    splits = make_splits(dataset, args.val, args.seed)
    path = save_splits(dataset, splits)
    print(
        f"{dataset.name}: {len(splits.train)} train, {len(splits.val)} val, {len(splits.failed)} failed (excluded) -> {path}"
    )
    for n in (10, 25, 50, 100, 200):
        if n < len(splits.train):
            print(f"  subset {n} (seed 0): {subset_ids(splits.train, n, 0)[:8]} ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
