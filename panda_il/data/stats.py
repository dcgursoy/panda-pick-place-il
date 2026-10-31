"""Normalisation statistics and action statistics for a dataset (or a subset of its episodes).

    uv run python -m panda_il.data.stats --dataset scripted_v1 [--split train]

Writes ``stats.json`` next to the store: per-dimension min / max / mean / std of
the synced proprio, state and action arrays, per-channel image mean / std, the
per-dimension action histogram and the fraction of saturated (|a| = 1) actions,
plus which episodes were used. Policies fit their normalisers on exactly the
episodes they train on (``compute_stats`` with an explicit id list), so the
file is documentation and a sanity check rather than something the training
code reads blindly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from panda_il.data.schema import Dataset
from panda_il.data.splits import load_splits
from panda_il.paths import Paths

HIST_BINS = np.linspace(-1.0, 1.0, 21)


def _dim_stats(array: np.ndarray) -> dict[str, list[float]]:
    array = np.asarray(array, dtype=np.float64)
    return {
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
    }


def compute_stats(dataset: Dataset, episode_ids: list[int], image_episodes: int = 20) -> dict:
    proprio, state, action, lengths = [], [], [], []
    image_sums: dict[str, np.ndarray] = {}
    image_sq: dict[str, np.ndarray] = {}
    image_count = 0
    cameras = list(dataset.metadata.cameras)
    for n, episode_id in enumerate(episode_ids):
        record = dataset.read_episode(episode_id, load_images=n < image_episodes)
        proprio.append(record.synced["proprio"])
        state.append(record.synced["state"])
        action.append(record.synced["action"])
        lengths.append(record.length)
        if n < image_episodes:
            for cam in cameras:
                images = np.asarray(record.raw[f"cam_{cam}"]["image"], dtype=np.float64) / 255.0
                image_sums[cam] = image_sums.get(cam, 0) + images.sum(axis=(0, 1, 2))
                image_sq[cam] = image_sq.get(cam, 0) + (images**2).sum(axis=(0, 1, 2))
                if cam == cameras[0]:
                    image_count += images.shape[0] * images.shape[1] * images.shape[2]
    proprio = np.concatenate(proprio)
    state = np.concatenate(state)
    action = np.concatenate(action)
    hist = {}
    for d in range(action.shape[1]):
        counts, _ = np.histogram(action[:, d], bins=HIST_BINS)
        hist[d] = counts.tolist()
    saturated = (np.abs(action[:, :6]) >= 0.999).mean(axis=0).tolist()
    stats = {
        "episodes": [int(i) for i in episode_ids],
        "steps": int(action.shape[0]),
        "episode_length": {
            "min": int(min(lengths)),
            "max": int(max(lengths)),
            "mean": float(np.mean(lengths)),
        },
        "proprio": _dim_stats(proprio),
        "state": _dim_stats(state),
        "action": _dim_stats(action),
        "action_histogram": {"bins": HIST_BINS.tolist(), "counts": hist},
        "action_saturated_fraction": saturated,
        "gripper_closed_fraction": float((action[:, 6] > 0).mean()),
        "images": {
            cam: {
                "mean": (image_sums[cam] / image_count).tolist(),
                "std": np.sqrt(
                    np.maximum(
                        image_sq[cam] / image_count - (image_sums[cam] / image_count) ** 2, 0
                    )
                ).tolist(),
                "sampled_episodes": min(image_episodes, len(episode_ids)),
            }
            for cam in cameras
            if cam in image_sums
        },
    }
    return stats


def stats_path(dataset: Dataset) -> Path:
    return dataset.directory / "stats.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", choices=["train", "val", "all"], default="train")
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    dataset = Dataset.open(Paths.resolve(args.data_root), args.dataset)
    if args.split == "all":
        ids = dataset.episode_ids()
    else:
        splits = load_splits(dataset)
        ids = splits.train if args.split == "train" else splits.val
    stats = compute_stats(dataset, ids)
    stats["split"] = args.split
    path = stats_path(dataset)
    path.write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(
        f"{dataset.name} ({args.split}, {len(ids)} episodes, {stats['steps']} steps): wrote {path}"
    )
    names = dataset.metadata.action["names"]
    print("action dims: min / max / mean / std / saturated")
    for d, name in enumerate(names):
        sat = stats["action_saturated_fraction"][d] if d < 6 else float("nan")
        print(
            f"  {name:8s} {stats['action']['min'][d]:+.3f} / {stats['action']['max'][d]:+.3f} / {stats['action']['mean'][d]:+.3f} / {stats['action']['std'][d]:.3f} / {sat:.2f}"
        )
    print(f"gripper closed fraction: {stats['gripper_closed_fraction']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
