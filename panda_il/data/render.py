"""Render a logged episode to MP4: the frames the policy saw, plus a third-person view
re-rendered from the logged simulator state.

    uv run python -m panda_il.data.render --dataset scripted_v1 --episode 0 [--out file.mp4] [--gif]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import imageio.v3 as iio
import numpy as np

from panda_il.data.recorder import env_config_from_metadata
from panda_il.data.schema import Dataset, EpisodeRecord
from panda_il.env.panda_env import PandaPickPlaceEnv
from panda_il.paths import Paths


def upscale(image: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(image, factor, axis=0), factor, axis=1)


def episode_frames(
    record: EpisodeRecord,
    env: PandaPickPlaceEnv | None,
    camera_names: tuple[str, ...],
    scale: int = 4,
    start: int = 0,
    end: int | None = None,
) -> list[np.ndarray]:
    """One composite frame per control tick in [start, end): [third-person | cameras...]."""
    n = len(record.synced["substep"])
    sim_states = np.asarray(record.raw["priv"]["sim_state"])
    frames = []
    for k in range(start, n if end is None else min(end, n)):
        panels = []
        if env is not None:
            env.set_sim_state(sim_states[k])
            panels.append(env.render_view("viewer"))
        for cam in camera_names:
            index = int(record.synced[f"image_index_{cam}"][k])
            image = upscale(np.asarray(record.raw[f"cam_{cam}"]["image"])[index], scale)
            panels.append(image)
        height = min(p.shape[0] for p in panels)
        pad = np.zeros((height, 8, 3), dtype=np.uint8)
        row = []
        for i, panel in enumerate(panels):
            if i:
                row.append(pad)
            row.append(panel[:height])
        frames.append(np.concatenate(row, axis=1))
    return frames


def write_video(frames: list[np.ndarray], path: Path, fps: int = 20) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(
        str(path),
        frames,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=16,
    )


def write_gif(
    frames: list[np.ndarray],
    path: Path,
    fps: int = 10,
    downscale: int = 2,
    stride: int | None = None,
) -> None:
    """Every ``stride``-th frame (default: real time at ``fps``) at ``1/downscale`` size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stride = max(1, 20 // fps) if stride is None else max(1, stride)
    small = [f[::downscale, ::downscale] for f in frames[::stride]]
    iio.imwrite(path, np.stack(small), extension=".gif", duration=int(1000 / fps), loop=0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument(
        "--out", default=None, help="default: DATA_ROOT/videos/<dataset>_ep<id>.mp4"
    )
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--gif-fps", type=int, default=10)
    parser.add_argument("--gif-downscale", type=int, default=2)
    parser.add_argument(
        "--gif-stride", type=int, default=None, help="keep every n-th frame (default: real time)"
    )
    parser.add_argument("--start", type=int, default=0, help="first control step to render")
    parser.add_argument("--end", type=int, default=None, help="one past the last step")
    parser.add_argument(
        "--no-viewer", action="store_true", help="skip the re-rendered third-person view"
    )
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    paths = Paths.resolve(args.data_root)
    dataset = Dataset.open(paths, args.dataset)
    record = dataset.read_episode(args.episode)
    env = None
    if not args.no_viewer:
        cfg = env_config_from_metadata(dataset.metadata)
        cfg.obs_mode = "state"
        cfg.render_width, cfg.render_height = 480, 384
        env = PandaPickPlaceEnv(cfg)
        env.reset(seed=int(record.attrs["seed"]), options={"ic": record.attrs["ic"]})
    frames = episode_frames(
        record, env, tuple(dataset.metadata.cameras), start=args.start, end=args.end
    )
    out = Path(args.out) if args.out else paths.videos / f"{dataset.name}_ep{args.episode:06d}.mp4"
    write_video(frames, out)
    print(f"wrote {out} ({len(frames)} frames, success={record.attrs['success']})")
    if args.gif:
        gif = out.with_suffix(".gif")
        write_gif(
            frames, gif, fps=args.gif_fps, downscale=args.gif_downscale, stride=args.gif_stride
        )
        print(f"wrote {gif}")
    if env is not None:
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
