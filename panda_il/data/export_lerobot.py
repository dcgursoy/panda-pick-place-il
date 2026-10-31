"""Export a dataset to the LeRobotDataset v3 format (run inside the LeRobot environment).

    uv run --project tools/lerobot_env python -m panda_il.data.export_lerobot --dataset scripted_v1 --variant image --images png
    uv run --project tools/lerobot_env python -m panda_il.data.export_lerobot --dataset scripted_v1 --variant state

Variants:
  image   observation.state = 22-D proprio, observation.images.<camera>, action; the cube pose is
          stored under ``privileged.cube_pose``, which is not an ``observation.*`` key, so LeRobot
          policies cannot condition on it (verified by the cross-check script).
  state   observation.state = 22-D proprio, observation.environment_state = 7-D cube pose, action.

Images are stored as PNG inside the parquet files by default (``--images png``,
LeRobot dtype "image"), so the AV1 codec of the default video path is not a
confound in the cross-check; ``--images video`` uses LeRobot's video encoder.
Only successful episodes from the train and validation splits are exported, in
split order, and the mapping from LeRobot episode index to our episode id is
written next to the export as ``export_map.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from panda_il.data.recorder import ACTION_NAMES, PROPRIO_NAMES
from panda_il.data.schema import Dataset
from panda_il.data.splits import load_splits
from panda_il.paths import Paths

TASK = "pick up the red cube and place it in the blue bin"
CUBE_POSE_NAMES = ["cube_x", "cube_y", "cube_z", "cube_qw", "cube_qx", "cube_qy", "cube_qz"]


def lerobot_features(metadata, variant: str, images: str) -> dict:
    features = {
        "observation.state": {"dtype": "float32", "shape": (22,), "names": list(PROPRIO_NAMES)},
        "action": {"dtype": "float32", "shape": (7,), "names": list(ACTION_NAMES)},
    }
    if variant == "image":
        for cam, spec in metadata.cameras.items():
            features[f"observation.images.{cam}"] = {
                "dtype": "video" if images == "video" else "image",
                "shape": (spec["height"], spec["width"], 3),
                "names": ["height", "width", "channels"],
            }
        features["privileged.cube_pose"] = {
            "dtype": "float32",
            "shape": (7,),
            "names": CUBE_POSE_NAMES,
        }
    elif variant == "state":
        features["observation.environment_state"] = {
            "dtype": "float32",
            "shape": (7,),
            "names": CUBE_POSE_NAMES,
        }
    else:
        raise ValueError(variant)
    return features


def export(
    dataset: Dataset,
    out_dir: Path,
    repo_id: str,
    variant: str,
    images: str,
    limit: int | None = None,
) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    splits = load_splits(dataset)
    episode_ids = list(splits.train) + list(splits.val)
    if limit:
        episode_ids = episode_ids[:limit]
    features = lerobot_features(dataset.metadata, variant, images)
    lerobot_ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(round(dataset.metadata.control_hz)),
        features=features,
        root=str(out_dir),
        robot_type="franka_panda_mujoco",
        use_videos=(variant == "image" and images == "video"),
        image_writer_threads=4 if variant == "image" else 0,
    )
    export_map = []
    total_frames = 0
    for lerobot_index, episode_id in enumerate(episode_ids):
        record = dataset.read_episode(episode_id, load_images=(variant == "image"))
        proprio = record.synced["proprio"].astype(np.float32)
        state = record.synced["state"].astype(np.float32)
        action = record.synced["action"].astype(np.float32)
        n_steps = len(action)
        cams = {}
        if variant == "image":
            for cam in dataset.metadata.cameras:
                index = record.synced[f"image_index_{cam}"]
                cams[cam] = np.asarray(record.raw[f"cam_{cam}"]["image"])[index]
        for k in range(n_steps):
            frame = {"observation.state": proprio[k], "action": action[k], "task": TASK}
            if variant == "image":
                for cam, frames in cams.items():
                    frame[f"observation.images.{cam}"] = np.ascontiguousarray(frames[k])
                frame["privileged.cube_pose"] = state[k, 22:29]
            else:
                frame["observation.environment_state"] = state[k, 22:29]
            lerobot_ds.add_frame(frame)
        lerobot_ds.save_episode()
        export_map.append(
            {
                "lerobot_episode": lerobot_index,
                "episode_id": int(episode_id),
                "frames": n_steps,
                "split": "train" if episode_id in splits.train else "val",
            }
        )
        total_frames += n_steps
        if (lerobot_index + 1) % 20 == 0:
            print(
                f"  {lerobot_index + 1}/{len(episode_ids)} episodes, {total_frames} frames",
                flush=True,
            )
    lerobot_ds.finalize()
    summary = {
        "repo_id": repo_id,
        "root": str(out_dir),
        "variant": variant,
        "images": images if variant == "image" else "none",
        "source_dataset": dataset.name,
        "episodes": len(episode_ids),
        "frames": total_frames,
        "features": {
            k: {"dtype": v["dtype"], "shape": list(v["shape"])} for k, v in features.items()
        },
        "map": export_map,
    }
    (out_dir / "export_map.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--variant", choices=["image", "state"], default="image")
    parser.add_argument("--images", choices=["png", "video"], default="png")
    parser.add_argument(
        "--limit", type=int, default=None, help="export only the first N episodes (smoke test)"
    )
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    paths = Paths.resolve(args.data_root)
    dataset = Dataset.open(paths, args.dataset)
    name = f"{args.dataset}_{args.variant}" + (
        "_video" if args.variant == "image" and args.images == "video" else ""
    )
    out_dir = paths.lerobot / name
    if out_dir.exists():
        print(f"{out_dir} exists; delete it first")
        return 1
    summary = export(
        dataset,
        out_dir,
        repo_id=f"dcgursoy/{name}",
        variant=args.variant,
        images=args.images,
        limit=args.limit,
    )
    print(f"exported {summary['episodes']} episodes / {summary['frames']} frames to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
