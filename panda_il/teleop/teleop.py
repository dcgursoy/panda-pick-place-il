"""Teleoperation with the live MuJoCo viewer, recording demonstrations into a dataset.

    uv run python -m panda_il.teleop.teleop --dataset human_v1 --operator "Deniz"
    uv run python -m panda_il.teleop.teleop --calibrate            # print gamepad axes and buttons live
    uv run python -m panda_il.teleop.teleop --dataset test --autopilot --no-viewer --episodes 3   # pipeline test

Keys (keyboard) or buttons (gamepad, configs/gamepad.yaml):
    N  start a new episode (resets the scene to a fresh random cube pose and begins recording)
    K  keep the episode (saves it, prints the running count)
    X  discard the episode
    G  toggle the gripper        R  reset the scene without recording        Esc  quit

The loop runs at the 20 Hz control rate, paced by the wall clock, and polls
the input at 50 Hz; every polled command is logged at its own time, and the
command in force at each control tick is what the environment executes. The
gamepad is used when one is connected (auto-detected), otherwise the keyboard.
"""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import nullcontext

import numpy as np

from panda_il.config import EnvConfig
from panda_il.data import Dataset, EpisodeRecorder, new_dataset
from panda_il.env import make_env
from panda_il.env.panda_env import PandaPickPlaceEnv
from panda_il.paths import Paths
from panda_il.teleop.expert import ScriptedExpert
from panda_il.teleop.gamepad import GamepadInput, save_default_mapping
from panda_il.teleop.keyboard import KeyboardInput

POLL_HZ = 50


class AutopilotInput:
    """The scripted expert through the input interface, for testing the session without a human."""

    def __init__(self, env: PandaPickPlaceEnv, episodes: int):
        self.expert = ScriptedExpert(env.cfg)
        self.env = env
        self.episodes_left = episodes
        self.obs = None
        self.info = None
        self.recording = False
        self.idle_ticks = 0

    def observe(self, obs, info) -> None:
        self.obs, self.info = obs, info

    def poll(self) -> tuple[np.ndarray, list[str]]:
        if not self.recording:
            if self.episodes_left <= 0:
                return np.array([0, 0, 0, 0, 0, 0, -1.0]), ["quit"]
            self.episodes_left -= 1
            self.recording = True
            self.expert.reset()
            return np.array([0, 0, 0, 0, 0, 0, -1.0]), ["start"]
        if self.obs is None:
            return np.array([0, 0, 0, 0, 0, 0, -1.0]), []
        if self.info["success"] or self.env.step_count >= self.env.cfg.max_steps:
            self.recording = False
            return np.array([0, 0, 0, 0, 0, 0, -1.0]), ["keep"]
        return self.expert.act_from_env(self.obs, self.info), []


class TeleopSession:
    def __init__(
        self,
        env: PandaPickPlaceEnv,
        recorder: EpisodeRecorder | None,
        inputs,
        use_viewer: bool,
        seed_start: int,
    ):
        self.env = env
        self.recorder = recorder
        self.inputs = inputs
        self.use_viewer = use_viewer
        self.next_seed = seed_start
        self.saved = (
            recorder.dataset.metadata.counts["episodes"] if recorder and recorder.dataset else 0
        )
        self.obs = None
        self.info = None
        self.profile: dict[str, list[float]] = {"step": [], "sync": [], "tick": []}

    def _print(self, message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    def print_profile(self) -> None:
        if len(self.profile["tick"]) < 10:
            return
        step = np.asarray(self.profile["step"]) * 1000
        sync = np.asarray(self.profile["sync"]) * 1000
        ticks = np.asarray(self.profile["tick"]) * 1000
        self._print(
            f"loop timing over {len(ticks)} ticks (ms, median / 90th pct): "
            f"env step {np.median(step):.1f} / {np.percentile(step, 90):.1f}, "
            f"viewer sync {np.median(sync):.1f} / {np.percentile(sync, 90):.1f}, "
            f"tick period {np.median(ticks):.1f} / {np.percentile(ticks, 90):.1f} "
            f"(target {self.env.cfg.control_dt * 1000:.0f})"
        )

    def _reset(self, record: bool) -> None:
        seed = self.next_seed
        self.next_seed += 1
        if record and self.recorder is not None:
            self.obs, self.info = self.recorder.start(seed=seed)
            self._print(f"episode started (seed {seed}); K keeps, X discards")
        else:
            self.obs, self.info = self.env.reset(seed=seed)
            self._print(f"scene reset (seed {seed}), not recording; press N to start an episode")
        if hasattr(self.inputs, "gripper_closed"):
            self.inputs.gripper_closed = False

    def _finish(self, keep: bool) -> None:
        if self.recorder is None or not self.recorder.recording:
            return
        if keep:
            record = self.recorder.finish()
            self.saved += 1
            outcome = "SUCCESS" if record.attrs["success"] else "no success"
            self._print(
                f"kept episode {record.attrs['episode_id']} ({record.length} steps, {outcome}); "
                f"{self.saved} demos saved in total; press N for the next one"
            )
        else:
            self.recorder.finish(discard="operator")
            self._print("episode discarded; press N to start again")

    def run(self) -> int:
        env = self.env
        viewer = None
        if self.use_viewer:
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(
                env.model, env.data, show_left_ui=False, show_right_ui=False
            )
        self._reset(record=False)
        control_dt = env.cfg.control_dt
        poll_dt = 1.0 / POLL_HZ
        latest = np.array([0, 0, 0, 0, 0, 0, -1.0])
        next_tick = time.perf_counter()
        tick_start = next_tick
        try:
            while viewer is None or viewer.is_running():
                # poll the input until the next control tick, logging each command at its own time
                while True:
                    if hasattr(self.inputs, "observe"):
                        self.inputs.observe(self.obs, self.info)
                    action, commands = self.inputs.poll()
                    latest = action
                    for command in commands:
                        if command == "start":
                            if self.recorder is not None and self.recorder.recording:
                                self._finish(keep=False)
                            self._reset(record=True)
                        elif command == "keep":
                            self._finish(keep=True)
                            self._reset(record=False)
                        elif command == "discard":
                            self._finish(keep=False)
                            self._reset(record=False)
                        elif command == "reset":
                            self._finish(keep=False)
                            self._reset(record=False)
                        elif command == "quit":
                            self._finish(keep=False)
                            return 0
                    if self.recorder is not None and self.recorder.recording:
                        self.recorder.log_command(latest)
                    now = time.perf_counter()
                    if now >= next_tick:
                        break
                    time.sleep(min(poll_dt, next_tick - now))
                next_tick += control_dt
                t_step = time.perf_counter()
                # The passive viewer's render thread competes with the offscreen camera renders
                # for the iGPU (two 96x96 renders take ~110 ms instead of ~22 ms while it draws);
                # holding its lock for the step confines the viewer's drawing to the idle gap.
                with viewer.lock() if viewer is not None else nullcontext():
                    if self.recorder is not None and self.recorder.recording:
                        self.obs, _, terminated, truncated, self.info = self.recorder.step(
                            latest, log_command=False
                        )
                    else:
                        self.obs, _, terminated, truncated, self.info = env.step(latest)
                if self.recorder is not None and self.recorder.recording:
                    if terminated:
                        self._print(
                            "the cube rests in the bin: episode complete, press K to keep it"
                        )
                elif terminated or truncated:
                    self._reset(record=False)
                t_sync = time.perf_counter()
                if viewer is not None:
                    viewer.sync()
                t_end = time.perf_counter()
                self.profile["step"].append(t_sync - t_step)
                self.profile["sync"].append(t_end - t_sync)
                self.profile["tick"].append(t_end - tick_start)
                tick_start = t_end
                if t_end > next_tick + control_dt:
                    next_tick = t_end  # we fell behind (e.g. a window drag); do not try to catch up
        finally:
            self.print_profile()
            if viewer is not None:
                viewer.close()
            if self.recorder is not None and self.recorder.recording:
                self.recorder.finish(discard="session ended")
        return 0


def calibrate() -> int:
    pad = GamepadInput()
    name = pad.start()
    print(
        f"gamepad: {name}. Move each stick and press each button; Ctrl+C to stop. Mapping file: configs/gamepad.yaml"
    )
    try:
        while True:
            axes, buttons = pad.raw()
            axes_text = " ".join(f"{i}:{v:+.2f}" for i, v in enumerate(axes))
            pressed = [i for i, b in enumerate(buttons) if b]
            print(f"\raxes {axes_text}   buttons pressed {pressed}      ", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()
    pad.stop()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset", default=None, help="dataset name under DATA_ROOT/datasets (created if missing)"
    )
    parser.add_argument("--operator", default="unknown")
    parser.add_argument("--input", choices=["auto", "keyboard", "gamepad"], default="auto")
    parser.add_argument(
        "--seed-start",
        type=int,
        default=None,
        help="first episode seed (default: continue from the dataset's count)",
    )
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument(
        "--autopilot",
        action="store_true",
        help="drive with the scripted expert instead of a human (pipeline test)",
    )
    parser.add_argument("--episodes", type=int, default=3, help="autopilot episodes")
    parser.add_argument(
        "--calibrate", action="store_true", help="print gamepad axes and buttons live"
    )
    parser.add_argument("--write-default-mapping", action="store_true")
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    if args.write_default_mapping:
        print(f"wrote {save_default_mapping()}")
        return 0
    if args.calibrate:
        return calibrate()

    paths = Paths.resolve(args.data_root)
    paths.make_dirs()
    env = make_env(EnvConfig(obs_mode="both"))

    if args.autopilot:
        inputs = AutopilotInput(env, args.episodes)
        device = "autopilot"
    elif args.input == "gamepad" or (args.input == "auto" and GamepadInput.available()):
        inputs = GamepadInput()
        device = "gamepad:" + inputs.start()
    else:
        inputs = KeyboardInput()
        inputs.start()
        device = "keyboard"
    print(f"input device: {device}")

    recorder = None
    if args.dataset:
        directory = paths.datasets / args.dataset
        source = "scripted" if args.autopilot else "human"
        if directory.exists():
            dataset = Dataset(directory, mode="a")
            if dataset.metadata.source != source:
                print(
                    f"dataset {args.dataset} holds {dataset.metadata.source} data; refusing to mix in {source} episodes"
                )
                return 1
            print(
                f"continuing dataset {args.dataset}: {dataset.metadata.counts['episodes']} episodes so far"
            )
        else:
            dataset = new_dataset(
                paths, args.dataset, env, source=source, operator=args.operator, input_device=device
            )
            print(f"created dataset {dataset.directory}")
        recorder = EpisodeRecorder(
            env, source, args.operator, device, dataset=dataset, rng_stream="teleop"
        )
        seed_start = (
            args.seed_start
            if args.seed_start is not None
            else 100_000
            + dataset.metadata.counts["episodes"]
            + dataset.metadata.counts.get("discarded", 0)
        )
    else:
        seed_start = args.seed_start or 0
        print("no --dataset given: driving only, nothing is recorded")

    session = TeleopSession(
        env, recorder, inputs, use_viewer=not args.no_viewer, seed_start=seed_start
    )
    try:
        return session.run()
    finally:
        if hasattr(inputs, "stop"):
            inputs.stop()
        env.close()


if __name__ == "__main__":
    sys.exit(main())
