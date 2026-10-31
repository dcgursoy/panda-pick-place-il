"""Milestone 1 evidence: a scripted pick-and-place through the public environment API,
rendered as a video showing both observation cameras and a third-person view.

    uv run python scripts/env_demo.py [--seed 0] [--gif]

The controller is deliberately crude (proportional moves of the commanded TCP
toward privileged targets); the proper scripted expert arrives in milestone 3.
Writes DATA_ROOT/videos/m1_env_demo_seed<seed>.mp4 and, with --gif, docs/gifs/m1_env_demo.gif.
"""

from __future__ import annotations

import argparse
import sys

import imageio.v2 as imageio
import imageio.v3 as iio
import numpy as np

from panda_il.config import EnvConfig
from panda_il.env import make_env
from panda_il.env.scene import BIN_CENTER
from panda_il.paths import DOCS_DIR, Paths


def wrap_to_quarter_turn(angle: float) -> float:
    """Map an angle to [-pi/4, pi/4): the cube's yaw modulo its 90-degree symmetry."""
    return (angle + np.pi / 4) % (np.pi / 2) - np.pi / 4


def hand_yaw(tcp_quat_wxyz: np.ndarray) -> float:
    """World yaw of the finger-closing axis (the TCP frame's y axis)."""
    w, x, y, z = tcp_quat_wxyz
    # second column of the rotation matrix
    yx = 2 * (x * y - w * z)
    yy = 1 - 2 * (x * x + z * z)
    return float(np.arctan2(yy, yx))


def move_action(cfg: EnvConfig, tcp_pos, target_pos, yaw_error: float, close: bool) -> np.ndarray:
    action = np.zeros(7)
    action[:3] = np.clip((np.asarray(target_pos) - tcp_pos) / cfg.max_delta_pos, -1, 1)
    # The TCP's z axis points down, so a positive rotation about it is a negative world yaw.
    action[5] = np.clip(-yaw_error / cfg.max_delta_rot, -1, 1)
    action[6] = 1.0 if close else -1.0
    return action


def scripted_episode(env, seed: int, max_steps: int = 300):
    cfg = env.cfg
    obs, info = env.reset(seed=seed)
    cube_yaw = info["ic"]["yaw"]
    frames = []
    phase = "align"
    hold = 0
    outcome = {"success": False, "steps": 0, "phases": []}

    def capture():
        front = np.repeat(np.repeat(obs["image_front"], 4, axis=0), 4, axis=1)  # 96 -> 384
        wrist = np.repeat(np.repeat(obs["image_wrist"], 4, axis=0), 4, axis=1)
        viewer = env.render_view("viewer")  # render_height x render_width
        pad = np.zeros((viewer.shape[0], 8, 3), dtype=np.uint8)
        frames.append(
            np.concatenate(
                [viewer, pad, front[: viewer.shape[0]], pad, wrist[: viewer.shape[0]]], axis=1
            )
        )

    capture()
    for step in range(max_steps):
        tcp_pos = info["tcp_measured"][:3]
        cube = obs["state"][22:25]
        yaw_error = wrap_to_quarter_turn(cube_yaw - hand_yaw(info["tcp_measured"][3:7]))
        close = phase in ("grasp", "lift", "carry", "lower")
        if phase == "align":
            target = cube + [0, 0, 0.12]
            if np.linalg.norm(tcp_pos - target) < 0.006 and abs(yaw_error) < 0.02:
                phase = "descend"
        elif phase == "descend":
            target = cube + [0, 0, 0.004]
            yaw_error = 0.0
            if np.linalg.norm(tcp_pos - target) < 0.006:
                phase, hold = "grasp", 0
        elif phase == "grasp":
            target = cube + [0, 0, 0.004]
            yaw_error = 0.0
            hold += 1
            if hold >= 12:
                phase = "lift"
        elif phase == "lift":
            target = [tcp_pos[0], tcp_pos[1], 0.15]
            yaw_error = 0.0
            if tcp_pos[2] > 0.14:
                phase = "carry"
        elif phase == "carry":
            target = [BIN_CENTER[0], BIN_CENTER[1], 0.15]
            yaw_error = 0.0
            if np.linalg.norm(tcp_pos[:2] - BIN_CENTER) < 0.008:
                phase = "lower"
        elif phase == "lower":
            target = [BIN_CENTER[0], BIN_CENTER[1], 0.09]
            yaw_error = 0.0
            if tcp_pos[2] < 0.095:
                phase, hold = "release", 0
        elif phase == "release":
            target = [BIN_CENTER[0], BIN_CENTER[1], 0.09]
            yaw_error = 0.0
            hold += 1
            if hold >= 10:
                phase = "retreat"
        else:  # retreat
            target = [BIN_CENTER[0], BIN_CENTER[1], 0.22]
            yaw_error = 0.0
        if not outcome["phases"] or outcome["phases"][-1][0] != phase:
            outcome["phases"].append((phase, step))
        obs, reward, terminated, truncated, info = env.step(
            move_action(cfg, tcp_pos, target, yaw_error, close)
        )
        capture()
        if terminated or truncated:
            break
    outcome["success"] = bool(info["success"])
    outcome["steps"] = env.step_count
    outcome["cube_final"] = obs["state"][22:25].round(3).tolist()
    outcome["resyncs"] = info["resync_count"]
    return frames, outcome


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gif", action="store_true", help="also write docs/gifs/m1_env_demo.gif")
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    paths = Paths.resolve(args.data_root)
    paths.make_dirs()

    env = make_env(EnvConfig(obs_mode="both", render_width=480, render_height=384))
    frames, outcome = scripted_episode(env, args.seed)
    env.close()
    print(
        f"seed {args.seed}: success={outcome['success']} steps={outcome['steps']} cube_final={outcome['cube_final']} resyncs={outcome['resyncs']}"
    )
    print("phases:", ", ".join(f"{name}@{step}" for name, step in outcome["phases"]))

    video = paths.videos / f"m1_env_demo_seed{args.seed}.mp4"
    imageio.mimwrite(
        str(video),
        frames,
        fps=20,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=16,
    )
    print(f"wrote {video} ({len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]})")
    if args.gif:
        small = [f[::2, ::2] for f in frames[::2]]  # 10 fps, half size
        gif = DOCS_DIR / "gifs" / "m1_env_demo.gif"
        iio.imwrite(gif, np.stack(small), extension=".gif", duration=100, loop=0)
        print(f"wrote {gif} ({gif.stat().st_size / 1e6:.2f} MB)")
    return 0 if outcome["success"] else 2


if __name__ == "__main__":
    sys.exit(main())
