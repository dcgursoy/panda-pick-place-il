"""Time the pieces of one control step and the full environment step.

    uv run python scripts/time_env.py

Prints a markdown table and saves DATA_ROOT/bench/env_timing.json.
"""

from __future__ import annotations

import json
import sys
import time

import mink
import mujoco
import numpy as np

from panda_il.config import EnvConfig, IKConfig
from panda_il.env import make_env
from panda_il.env.cameras import CameraSet, ViewRenderer
from panda_il.env.ik import make_ik
from panda_il.env.scene import compile_arm_model, compile_scene_model, scene_ids
from panda_il.paths import Paths


def median_ms(fn, repeats: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


def make_ik_solver(ik):
    state = {"target": ik.tcp_pose(), "sign": 1.0}

    def solve():
        state["target"] = (
            mink.SE3.from_translation(np.array([0.005 * state["sign"], 0, 0])) @ state["target"]
        )
        state["sign"] *= -1
        ik.solve(state["target"], 0.05)

    return solve


def make_env_stepper(env):
    action = np.zeros(7)
    counter = {"n": 0}

    def step():
        counter["n"] += 1
        action[0] = 1.0 if (counter["n"] // 20) % 2 == 0 else -1.0
        env.step(action)

    return step


def main() -> int:
    paths = Paths.resolve()
    paths.make_dirs()
    results: dict[str, float] = {}

    model = compile_scene_model()
    data = mujoco.MjData(model)
    ids = scene_ids(model)
    mujoco.mj_resetDataKeyframe(model, data, ids.home_key)
    data.ctrl[ids.arm_actuators] = data.qpos[ids.arm_qpos]
    data.ctrl[ids.gripper_actuator] = 255

    def physics_25():
        for _ in range(25):
            mujoco.mj_step(model, data)

    results["physics: 25 substeps"] = median_ms(physics_25, 200)

    arm = compile_arm_model()
    home = np.array([0, 0, 0, -1.5708, 0, 1.5708, -0.7853, 0.04, 0.04])
    for backend in ("mink", "dls"):
        ik = make_ik(arm, IKConfig(backend=backend))
        ik.reset(home)
        results[f"ik solve ({backend}), 5 mm target step"] = median_ms(make_ik_solver(ik), 200)

    cameras = CameraSet(model, EnvConfig().camera)
    cameras.reset(data)
    mujoco.mj_forward(model, data)
    results["render front 96x96"] = median_ms(lambda: cameras._render(data, "front"), 100)
    results["render wrist 96x96"] = median_ms(lambda: cameras._render(data, "wrist"), 100)
    cameras.close()
    viewer = ViewRenderer(model, 480, 384)
    results["render viewer 480x384"] = median_ms(lambda: viewer.render(data), 50)
    viewer.close()

    for mode in ("state", "image", "both"):
        env = make_env(EnvConfig(obs_mode=mode))
        env.reset(seed=0)
        results[f"env.step obs_mode={mode}"] = median_ms(make_env_stepper(env), 100)
        env.close()

    print("| what | ms (median) |")
    print("|---|---:|")
    for name, ms in results.items():
        print(f"| {name} | {ms:.2f} |")
    budget = 1000 / EnvConfig().control_hz
    print(
        f"\ncontrol period budget: {budget:.0f} ms; env.step with both cameras uses {results['env.step obs_mode=both'] / budget * 100:.0f}% of it"
    )
    out = paths.bench / "env_timing.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
