"""Cross-check an exported LeRobotDataset against our Zarr dataset (run inside the LeRobot environment).

    uv run --project tools/lerobot_env python tools/lerobot_env/crosscheck_dataset.py --dataset scripted_v1 --variant image

Loads the export with LeRobot's own loader, compares frame counts, states,
actions and images against the source episodes, loads a Diffusion-Policy-style
window with delta_timestamps, and confirms that LeRobot's policy input features
for the image variant do not include the cube pose.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from panda_il.data.schema import Dataset
from panda_il.paths import Paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--variant", choices=["image", "state"], default="image")
    parser.add_argument("--video", action="store_true", help="the export used video encoding")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    paths = Paths.resolve(args.data_root)
    name = f"{args.dataset}_{args.variant}" + (
        "_video" if args.variant == "image" and args.video else ""
    )
    root = paths.lerobot / name
    summary = json.loads((root / "export_map.json").read_text(encoding="utf-8"))
    source = Dataset.open(paths, args.dataset)

    lerobot_ds = LeRobotDataset(summary["repo_id"], root=str(root), video_backend="pyav")
    print(
        f"LeRobot loaded {lerobot_ds.num_episodes} episodes, {lerobot_ds.num_frames} frames, fps {lerobot_ds.fps}"
    )
    print(f"features: {sorted(lerobot_ds.features)}")
    ok = True
    if lerobot_ds.num_frames != summary["frames"] or lerobot_ds.num_episodes != summary["episodes"]:
        print("MISMATCH: frame or episode counts differ from the export map")
        ok = False

    # frame offsets per LeRobot episode (episodes are exported in map order)
    offsets = np.cumsum([0] + [m["frames"] for m in summary["map"]])
    rng = np.random.default_rng(0)
    picks = rng.integers(0, lerobot_ds.num_frames, size=min(args.samples, lerobot_ds.num_frames))
    max_state = max_action = max_image = max_cube = 0.0
    cache: dict[int, object] = {}
    for global_index in sorted(int(p) for p in picks):
        episode_pos = int(np.searchsorted(offsets, global_index, side="right") - 1)
        k = global_index - offsets[episode_pos]
        episode_id = summary["map"][episode_pos]["episode_id"]
        if episode_id not in cache:
            cache = {
                episode_id: source.read_episode(episode_id, load_images=(args.variant == "image"))
            }
        record = cache[episode_id]
        item = lerobot_ds[global_index]
        assert int(item["episode_index"]) == episode_pos and int(item["frame_index"]) == k, (
            global_index,
            episode_pos,
            k,
        )
        state = item["observation.state"].numpy()
        max_state = max(
            max_state, float(np.abs(state - record.synced["proprio"][k].astype(np.float32)).max())
        )
        action = item["action"].numpy()
        max_action = max(
            max_action, float(np.abs(action - record.synced["action"][k].astype(np.float32)).max())
        )
        cube = record.synced["state"][k, 22:29].astype(np.float32)
        if args.variant == "image":
            max_cube = max(
                max_cube, float(np.abs(item["privileged.cube_pose"].numpy() - cube).max())
            )
            for cam in source.metadata.cameras:
                image = item[f"observation.images.{cam}"]  # float32 CHW in [0, 1]
                image = (image.permute(1, 2, 0).numpy() * 255).round().astype(np.int16)
                index = record.synced[f"image_index_{cam}"][k]
                ours = np.asarray(record.raw[f"cam_{cam}"]["image"])[index].astype(np.int16)
                max_image = max(max_image, float(np.abs(image - ours).max()))
        else:
            max_cube = max(
                max_cube, float(np.abs(item["observation.environment_state"].numpy() - cube).max())
            )
    print(
        f"checked {len(picks)} random frames: max |state diff| {max_state:.2e}, max |action diff| {max_action:.2e}, "
        f"max |cube pose diff| {max_cube:.2e}"
        + (f", max |image diff| {max_image:.0f}/255" if args.variant == "image" else "")
    )
    if max_state > 1e-6 or max_action > 1e-6 or max_cube > 1e-6:
        print("MISMATCH: states, actions or cube poses differ")
        ok = False
    if args.variant == "image" and max_image > (3 if args.video else 0):
        print("MISMATCH: images differ more than expected")
        ok = False

    # a Diffusion-Policy-style window: 2 observation steps, 16 actions
    fps = lerobot_ds.fps
    delta = {"observation.state": [-1 / fps, 0.0], "action": [i / fps for i in range(-1, 15)]}
    if args.variant == "image":
        for cam in source.metadata.cameras:
            delta[f"observation.images.{cam}"] = [-1 / fps, 0.0]
    windowed = LeRobotDataset(
        summary["repo_id"], root=str(root), delta_timestamps=delta, video_backend="pyav"
    )
    item = windowed[5]
    print(
        f"window shapes: state {tuple(item['observation.state'].shape)}, action {tuple(item['action'].shape)}"
        + (
            f", image {tuple(item['observation.images.front'].shape)}"
            if args.variant == "image"
            else ""
        )
    )
    assert tuple(item["observation.state"].shape) == (2, 22) and tuple(item["action"].shape) == (
        16,
        7,
    )

    # LeRobot derives policy inputs from observation.* keys: the cube pose must not be among them
    from lerobot.datasets.utils import dataset_to_policy_features

    policy_features = dataset_to_policy_features(lerobot_ds.features)
    inputs = sorted(k for k in policy_features if k != "action")
    print(f"policy input features LeRobot would use: {inputs}")
    if args.variant == "image" and any("cube" in k for k in inputs):
        print("MISMATCH: the cube pose leaked into the policy inputs")
        ok = False
    print("CROSS-CHECK OK" if ok else "CROSS-CHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
