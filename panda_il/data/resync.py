"""Rebuild the synced arrays of recorded episodes from their raw streams.

    uv run python -m panda_il.data.resync --dataset human_v1
    uv run python -m panda_il.data.resync --dataset human_v1 --episodes 0 3

The raw streams are the record; ``synced/`` is derived from them by
``sync.sync_episode`` at save time. When the sync code changes (the quaternion
sign fix after the first human episode), this recomputes ``synced/`` and the
alignment attributes in place so the validator's "stored equals fresh" check
holds again. Nothing in ``raw/`` or ``init/`` is touched.
"""

from __future__ import annotations

import argparse
import sys

from panda_il.data.schema import Dataset, _write_array
from panda_il.data.sync import sync_episode
from panda_il.paths import Paths


def resync_episode(dataset: Dataset, episode_id: int) -> dict[str, float]:
    record = dataset.read_episode(episode_id, load_images=False)
    metadata = dataset.metadata
    n_steps = int(record.attrs["length"])
    result = sync_episode(
        record.raw,
        n_steps,
        metadata.substeps,
        1.0 / metadata.physics_hz,
        tuple(metadata.cameras),
    )
    group = dataset.root[dataset.episode_key(episode_id)]
    del group["synced"]
    synced = group.create_group("synced")
    for name, array in result.arrays.items():
        _write_array(synced, name, array)
    max_err = {name: a["max_s"] for name, a in result.alignment.items()}
    group.attrs.update({"alignment": result.alignment, "max_alignment_error_s": max_err})
    return max_err


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--episodes", type=int, nargs="*", default=None, help="default: all")
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    dataset = Dataset.open(Paths.resolve(args.data_root), args.dataset, mode="a")
    ids = args.episodes if args.episodes else dataset.episode_ids()
    for episode_id in ids:
        max_err = resync_episode(dataset, episode_id)
        errors = ", ".join(f"{k} {v * 1000:.1f} ms" for k, v in max_err.items())
        print(f"episode {episode_id}: synced arrays rebuilt (max alignment error {errors})")
    print(f"{len(ids)} episode(s) re-synced; run the validator to confirm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
