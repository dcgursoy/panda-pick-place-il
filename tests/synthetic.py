"""Synthetic episodes with known timing, for the sync, schema and validator tests.

Everything is generated from closed-form functions of the physics substep, so
the tests know exactly what the synchronised arrays must contain.
"""

from __future__ import annotations

import numpy as np

from panda_il.data.schema import SCHEMA_VERSION, DatasetMetadata, EpisodeRecord, utc_now
from panda_il.data.sync import sync_episode

SUBSTEPS = 25
PHYSICS_DT = 0.002
PROPRIO_PERIOD = 5
CAMS = ("front", "wrist")
IMAGE_SIZE = 8
STATE_SIZE = 6


def truth_qpos(substep):
    s = np.asarray(substep, dtype=np.float64)[:, None]
    return np.sin(0.004 * s + np.arange(7)[None, :]) * 0.5


def truth_qvel(substep):
    s = np.asarray(substep, dtype=np.float64)[:, None]
    return np.cos(0.004 * s + np.arange(7)[None, :]) * 0.1


def truth_gripper(substep):
    return 0.04 + 0.0 * np.asarray(substep, dtype=np.float64)


def truth_tcp_pos(substep):
    s = np.asarray(substep, dtype=np.float64)[:, None]
    return np.array([0.5, 0.0, 0.5])[None, :] + 1e-4 * s * np.array([1.0, -0.5, 0.2])[None, :]


def truth_tcp_quat(substep):
    angle = 0.002 * np.asarray(substep, dtype=np.float64)
    return np.stack(
        [np.cos(angle / 2), np.zeros_like(angle), np.zeros_like(angle), np.sin(angle / 2)], axis=1
    )


def truth_cube(substep):
    s = np.asarray(substep, dtype=np.float64)[:, None]
    pos = np.array([0.5, -0.15, 0.02])[None, :] + 2e-5 * s * np.array([1.0, 1.0, 0.0])[None, :]
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(s), 1))
    return pos, quat


def truth_action(step_index):
    k = np.asarray(step_index, dtype=np.float64)[:, None]
    return np.clip(np.sin(0.3 * k + np.arange(7)[None, :]), -1, 1)


def proprio_stream(
    n_steps: int, offset: int = 0, period: int = PROPRIO_PERIOD
) -> dict[str, np.ndarray]:
    last = n_steps * SUBSTEPS
    substeps = np.arange(offset, last + 1, period, dtype=np.int64)
    if offset != 0:
        # a sample at reset is always logged, and the stream runs past the last tick
        substeps = np.concatenate([[0], np.arange(offset, last + period, period, dtype=np.int64)])
    return {
        "substep": substeps,
        "t_sim": substeps * PHYSICS_DT,
        "t_wall": substeps * PHYSICS_DT * 1.01,
        "qpos": truth_qpos(substeps),
        "qvel": truth_qvel(substeps),
        "gripper_width": truth_gripper(substeps),
        "tcp_pos": truth_tcp_pos(substeps),
        "tcp_quat": truth_tcp_quat(substeps),
    }


def camera_stream(
    n_steps: int, capture_offset: int = -10, latency: int = 8, drop_ticks: tuple[int, ...] = ()
) -> dict[str, np.ndarray]:
    """Frame 0 at reset (zero latency), then one frame per tick k >= 1 captured at k*25 + offset."""
    captures = [0]
    arrivals = [0]
    for k in range(1, n_steps + 1):
        if k in drop_ticks:
            continue
        captures.append(k * SUBSTEPS + capture_offset)
        arrivals.append(k * SUBSTEPS + capture_offset + latency)
    captures = np.asarray(captures, dtype=np.int64)
    arrivals = np.asarray(arrivals, dtype=np.int64)
    images = np.zeros((len(captures), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    images[:, :, :, 0] = (np.arange(len(captures)) % 256)[
        :, None, None
    ]  # frame index in the red channel
    return {
        "substep": captures,
        "t_sim": captures * PHYSICS_DT,
        "t_wall": arrivals * PHYSICS_DT * 1.01,
        "capture_substep": captures,
        "arrival_substep": arrivals,
        "image": images,
    }


def action_stream(
    n_steps: int, command_substeps: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    if command_substeps is None:
        command_substeps = np.arange(n_steps, dtype=np.int64) * SUBSTEPS
    command_substeps = np.asarray(command_substeps, dtype=np.int64)
    return {
        "substep": command_substeps,
        "t_sim": command_substeps * PHYSICS_DT,
        "t_wall": np.arange(len(command_substeps)) * 0.02,
        "action": truth_action(np.arange(len(command_substeps))),
    }


def tick_streams(
    n_steps: int, success: bool = False
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """The 20 Hz ``obs`` and ``priv`` streams consistent with the ground truth and with
    the default camera timing (delivered frame k at tick k)."""
    ticks = np.arange(n_steps + 1, dtype=np.int64) * SUBSTEPS
    proprio = np.concatenate(
        [
            truth_qpos(ticks),
            truth_qvel(ticks),
            truth_gripper(ticks)[:, None],
            truth_tcp_pos(ticks),
            truth_tcp_quat(ticks),
        ],
        axis=1,
    )
    cube_pos, cube_quat = truth_cube(ticks)
    state = np.concatenate([proprio, cube_pos, cube_quat], axis=1)
    frame_substeps = np.concatenate([[0], ticks[1:] - 10])
    obs = {
        "substep": ticks,
        "t_sim": ticks * PHYSICS_DT,
        "t_wall": ticks * PHYSICS_DT * 1.01,
        "proprio": proprio.astype(np.float32),
        "state": state.astype(np.float32),
    }
    for cam in CAMS:
        obs[f"frame_substep_{cam}"] = frame_substeps
    executed = np.concatenate([np.zeros((1, 7)), truth_action(np.arange(n_steps))], axis=0)
    streak = np.zeros(n_steps + 1, dtype=np.int64)
    if success:
        streak[-10:] = np.arange(1, 11)
    priv = {
        "substep": ticks,
        "t_sim": ticks * PHYSICS_DT,
        "t_wall": ticks * PHYSICS_DT * 1.01,
        "sim_state": np.tile(np.arange(STATE_SIZE, dtype=np.float64), (n_steps + 1, 1))
        + ticks[:, None] * 1e-3,
        "ctrl": np.zeros((n_steps + 1, 8)),
        "q_target": np.zeros((n_steps + 1, 7)),
        "tcp_target": np.zeros((n_steps + 1, 7)),
        "tcp_measured": np.zeros((n_steps + 1, 7)),
        "cube_pos": cube_pos,
        "cube_quat": cube_quat,
        "cube_linvel": np.zeros((n_steps + 1, 3)),
        "gripper_closed": np.zeros(n_steps + 1, dtype=bool),
        "action_executed": executed,
        "success_now": streak > 0,
        "success_streak": streak,
        "contacts": np.zeros((n_steps + 1, 5), dtype=bool),
        "reward_terms": np.zeros((n_steps + 1, 5)),
        "tracking_lag": np.zeros((n_steps + 1, 2)),
        "resync_count": np.zeros(n_steps + 1, dtype=np.int64),
    }
    return obs, priv


def synthetic_metadata(name: str = "synthetic", source: str = "scripted") -> DatasetMetadata:
    cameras = {
        cam: {
            "fovy_deg": 45.0,
            "fx": 10.0,
            "fy": 10.0,
            "cx": 4.0,
            "cy": 4.0,
            "width": IMAGE_SIZE,
            "height": IMAGE_SIZE,
            "period_substeps": SUBSTEPS,
            "capture_offset_substeps": -10,
            "latency_substeps": 8,
        }
        for cam in CAMS
    }
    streams = {
        "proprio": {
            "nominal_period_substeps": PROPRIO_PERIOD,
            "fields": ["qpos", "qvel", "gripper_width", "tcp_pos", "tcp_quat"],
        },
        "obs": {
            "nominal_period_substeps": SUBSTEPS,
            "fields": ["proprio", "state"] + [f"frame_substep_{c}" for c in CAMS],
        },
        "priv": {
            "nominal_period_substeps": SUBSTEPS,
            "fields": ["sim_state", "cube_pos", "cube_quat", "action_executed", "success_streak"],
        },
        "action": {"nominal_period_substeps": 0, "fields": ["action"]},
    }
    for cam in CAMS:
        streams[f"cam_{cam}"] = {
            "nominal_period_substeps": SUBSTEPS,
            "fields": ["capture_substep", "arrival_substep", "image"],
        }
    return DatasetMetadata(
        name=name,
        source=source,
        operator="synthetic",
        input_device="none",
        env_version="test",
        env_config={"success_steps": 10, "max_steps": 300},
        control_hz=1.0 / (SUBSTEPS * PHYSICS_DT),
        physics_hz=1.0 / PHYSICS_DT,
        substeps=SUBSTEPS,
        cameras=cameras,
        action={"dim": 7},
        observation={"proprio_dim": 22, "state_dim": 29},
        streams=streams,
        mujoco_state_spec="synthetic",
        mujoco_state_size=STATE_SIZE,
        versions={},
        git_commit="test",
        git_dirty=False,
        created_at=utc_now(),
    )


def synthetic_record(
    n_steps: int = 12,
    seed: int = 0,
    success: bool = False,
    ic=(0.5, -0.15, 0.3),
    source: str = "scripted",
) -> EpisodeRecord:
    raw = {"proprio": proprio_stream(n_steps), "action": action_stream(n_steps)}
    obs, priv = tick_streams(n_steps, success=success)
    raw["obs"], raw["priv"] = obs, priv
    for cam in CAMS:
        raw[f"cam_{cam}"] = camera_stream(n_steps)
    raw_attrs = {
        name: {
            "nominal_period_substeps": synthetic_metadata().streams[name]["nominal_period_substeps"]
        }
        for name in raw
    }
    result = sync_episode(raw, n_steps, SUBSTEPS, PHYSICS_DT, CAMS)
    attrs = {
        "source": source,
        "operator": "synthetic",
        "input_device": "none",
        "seed": seed,
        "ic": {"x": ic[0], "y": ic[1], "yaw": ic[2]},
        "rng_stream": "test",
        "t_wall_start": utc_now(),
        "env_version": "test",
        "time_limit_steps": 300,
        "git_commit": "test",
        "git_dirty": False,
        "length": n_steps,
        "success": success,
        "alignment": result.alignment,
        "max_alignment_error_s": {k: v["max_s"] for k, v in result.alignment.items()},
        "schema_version": SCHEMA_VERSION,
    }
    init = {
        "sim_state": priv["sim_state"][0].copy(),
        "proprio": obs["proprio"][0].copy(),
        "state": obs["state"][0].copy(),
    }
    return EpisodeRecord(attrs=attrs, init=init, raw=raw, raw_attrs=raw_attrs, synced=result.arrays)
