"""The multi-rate episode logger.

``EpisodeRecorder`` wraps a ``PandaPickPlaceEnv`` and logs every stream at its
own rate with its own timestamps (simulation time, wall-clock time, physics
substep):

    proprio      100 Hz, every 5th physics substep, from a kinematics pass
    cam_<name>   on each camera's own clock (captured frames with capture and arrival substeps)
    action       at the input rate: whenever the teleop or the expert issues a command
    obs          20 Hz, the observation the policy saw at each control tick
    priv         20 Hz, privileged simulator state, contacts, reward terms, executed action
    init         the state at reset

``finish()`` runs the sync module and returns an ``EpisodeRecord``; with a
``Dataset`` attached it also writes the episode. ``finish(discard=...)`` drops it.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

import mujoco
import numpy as np

from panda_il.config import EnvConfig
from panda_il.data.schema import (
    SCHEMA_VERSION,
    Dataset,
    DatasetMetadata,
    EpisodeRecord,
    git_state,
    library_versions,
    utc_now,
)
from panda_il.data.sync import sync_episode
from panda_il.env import ENV_VERSION, PROPRIO_DIM, STATE_DIM
from panda_il.env.cameras import Frame, camera_intrinsics
from panda_il.env.panda_env import PandaPickPlaceEnv
from panda_il.env.task import InitialCondition

PROPRIO_PERIOD_SUBSTEPS = 5

PROPRIO_NAMES = (
    [f"q{i}" for i in range(1, 8)]
    + [f"dq{i}" for i in range(1, 8)]
    + ["gripper_width", "tcp_x", "tcp_y", "tcp_z", "tcp_qw", "tcp_qx", "tcp_qy", "tcp_qz"]
)
STATE_NAMES = PROPRIO_NAMES + ["cube_x", "cube_y", "cube_z", "cube_qw", "cube_qx", "cube_qy", "cube_qz"]
ACTION_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


class StreamBuffer:
    def __init__(self, nominal_period_substeps: int, kind: str):
        self.fields: dict[str, list] = {}
        self.attrs = {"nominal_period_substeps": nominal_period_substeps, "kind": kind}

    def append(self, **values) -> None:
        for name, value in values.items():
            self.fields.setdefault(name, []).append(value)

    def __len__(self) -> int:
        return len(self.fields.get("substep", []))

    def arrays(self) -> dict[str, np.ndarray]:
        return {name: np.asarray(values) for name, values in self.fields.items()}


def build_metadata(env: PandaPickPlaceEnv, name: str, source: str, operator: str, input_device: str, notes: str = "") -> DatasetMetadata:
    cfg = env.cfg
    cameras = {}
    for cam in cfg.camera.names:
        cameras[cam] = {
            **camera_intrinsics(env.model, cam, cfg.camera.width, cfg.camera.height),
            "period_substeps": cfg.camera.period_substeps,
            "capture_offset_substeps": cfg.camera.capture_offset_substeps,
            "latency_substeps": cfg.camera.latency_substeps,
        }
    commit, dirty = git_state()
    streams = {
        "proprio": {"nominal_period_substeps": PROPRIO_PERIOD_SUBSTEPS, "fields": ["qpos", "qvel", "gripper_width", "tcp_pos", "tcp_quat"]},
        "obs": {"nominal_period_substeps": cfg.substeps, "fields": ["proprio", "state"] + [f"frame_substep_{c}" for c in cfg.camera.names]},
        "priv": {
            "nominal_period_substeps": cfg.substeps,
            "fields": [
                "sim_state", "ctrl", "q_target", "tcp_target", "tcp_measured", "cube_pos", "cube_quat", "cube_linvel",
                "gripper_closed", "action_executed", "success_now", "success_streak", "contacts", "reward_terms",
                "tracking_lag", "resync_count",
            ],
        },
        "action": {"nominal_period_substeps": 0, "fields": ["action"]},
    }
    for cam in cfg.camera.names:
        streams[f"cam_{cam}"] = {"nominal_period_substeps": cfg.camera.period_substeps, "fields": ["capture_substep", "arrival_substep", "image"]}
    return DatasetMetadata(
        name=name,
        source=source,
        operator=operator,
        input_device=input_device,
        env_version=ENV_VERSION,
        env_config=asdict(cfg),
        control_hz=cfg.control_hz,
        physics_hz=1.0 / cfg.physics_timestep,
        substeps=cfg.substeps,
        cameras=cameras,
        action={
            "dim": 7,
            "names": ACTION_NAMES,
            "range": [-1.0, 1.0],
            "max_delta_pos_m": cfg.max_delta_pos,
            "max_delta_rot_rad": cfg.max_delta_rot,
            "gripper": "negative = open, positive = close",
        },
        observation={"proprio_dim": PROPRIO_DIM, "proprio_names": PROPRIO_NAMES, "state_dim": STATE_DIM, "state_names": STATE_NAMES},
        streams=streams,
        mujoco_state_spec="mjSTATE_INTEGRATION",
        mujoco_state_size=env.state_size,
        versions=library_versions(),
        git_commit=commit,
        git_dirty=dirty,
        created_at=utc_now(),
        notes=notes,
    )


class EpisodeRecorder:
    def __init__(
        self,
        env: PandaPickPlaceEnv,
        source: str,
        operator: str,
        input_device: str,
        dataset: Dataset | None = None,
        rng_stream: str = "",
    ):
        if source not in ("human", "scripted"):
            raise ValueError("source must be 'human' or 'scripted'")
        if env.cfg.obs_mode != "both":
            raise ValueError("the recorder needs obs_mode='both' (state and images)")
        self.env = env
        self.source = source
        self.operator = operator
        self.input_device = input_device
        self.dataset = dataset
        self.rng_stream = rng_stream
        self.recording = False
        self.streams: dict[str, StreamBuffer] = {}
        self.init: dict[str, np.ndarray] = {}
        self.attrs: dict[str, Any] = {}
        self._t0_wall = 0.0

    # ------------------------------------------------------------------ timing
    def _t_wall(self) -> float:
        return time.perf_counter() - self._t0_wall

    def _t_sim(self) -> float:
        return float(self.env.data.time)

    # ---------------------------------------------------------------- listeners
    def _on_substep(self, substep: int) -> None:
        if substep % PROPRIO_PERIOD_SUBSTEPS == 0:
            self._log_proprio(substep)

    def _log_proprio(self, substep: int) -> None:
        env = self.env
        mujoco.mj_kinematics(env.model, env.data)
        pos, quat = env.tcp_pose_measured()
        self.streams["proprio"].append(
            t_sim=self._t_sim(),
            t_wall=self._t_wall(),
            substep=substep,
            qpos=env.data.qpos[env.ids.arm_qpos].copy(),
            qvel=env.data.qvel[env.ids.arm_dof].copy(),
            gripper_width=env.gripper_width(),
            tcp_pos=pos,
            tcp_quat=quat,
        )

    def _on_capture(self, name: str, frame: Frame) -> None:
        self.streams[f"cam_{name}"].append(
            t_sim=frame.capture_substep * self.env.cfg.physics_timestep,
            t_wall=self._t_wall(),
            substep=frame.capture_substep,
            capture_substep=frame.capture_substep,
            arrival_substep=frame.arrival_substep,
            image=frame.image,
        )

    def _log_tick(self, obs: dict[str, np.ndarray], info: dict[str, Any]) -> None:
        env = self.env
        substep = env.substep
        frame_substeps = {f"frame_substep_{c}": env.cameras.latest[c].capture_substep for c in env.cfg.camera.names}
        self.streams["obs"].append(
            t_sim=self._t_sim(), t_wall=self._t_wall(), substep=substep, proprio=obs["proprio"].copy(), state=obs["state"].copy(), **frame_substeps
        )
        contacts = info["contacts"]
        terms = info["reward_terms"]
        self.streams["priv"].append(
            t_sim=self._t_sim(),
            t_wall=self._t_wall(),
            substep=substep,
            sim_state=info["sim_state"].copy(),
            ctrl=info["ctrl"].copy(),
            q_target=info["q_target"].copy(),
            tcp_target=np.asarray(info["tcp_target"], dtype=np.float64),
            tcp_measured=np.asarray(info["tcp_measured"], dtype=np.float64),
            cube_pos=np.asarray(info["cube_pos"], dtype=np.float64),
            cube_quat=np.asarray(info["cube_quat"], dtype=np.float64),
            cube_linvel=np.asarray(info["cube_linvel"], dtype=np.float64),
            gripper_closed=bool(info["gripper_closed"]),
            action_executed=np.asarray(info["action_executed"], dtype=np.float64),
            success_now=bool(info["cube_in_bin_now"]),
            success_streak=int(info["success_streak"]),
            contacts=np.array([contacts[k] for k in ("floor", "wall", "table", "left_finger", "right_finger")], dtype=bool),
            reward_terms=np.array([terms[k] for k in ("reach", "grasp", "lift", "place", "success")], dtype=np.float64),
            tracking_lag=np.asarray(info["tracking_lag"], dtype=np.float64),
            resync_count=int(info["resync_count"]),
        )

    # ------------------------------------------------------------------- API
    def start(self, seed: int, ic: InitialCondition | dict | None = None, notes: str = "") -> tuple[dict, dict]:
        if self.recording:
            raise RuntimeError("already recording; call finish() first")
        env = self.env
        self.streams = {
            "proprio": StreamBuffer(PROPRIO_PERIOD_SUBSTEPS, "continuous"),
            "obs": StreamBuffer(env.cfg.substeps, "tick"),
            "priv": StreamBuffer(env.cfg.substeps, "tick"),
            "action": StreamBuffer(0, "command"),
        }
        for cam in env.cfg.camera.names:
            self.streams[f"cam_{cam}"] = StreamBuffer(env.cfg.camera.period_substeps, "camera")
        env.substep_listeners.append(self._on_substep)
        env.cameras.capture_listeners.append(self._on_capture)
        self.recording = True
        self._t0_wall = time.perf_counter()
        wall_start = utc_now()

        obs, info = env.reset(seed=seed, options={"ic": ic} if ic is not None else None)
        self._log_proprio(0)
        self._log_tick(obs, info)
        self.init = {"sim_state": info["sim_state"].copy(), "proprio": obs["proprio"].copy(), "state": obs["state"].copy()}
        commit, dirty = git_state()
        self.attrs = {
            "source": self.source,
            "operator": self.operator,
            "input_device": self.input_device,
            "seed": int(seed),
            "ic": info["ic"],
            "rng_stream": self.rng_stream,
            "t_wall_start": wall_start,
            "env_version": ENV_VERSION,
            "time_limit_steps": env.cfg.max_steps,
            "git_commit": commit,
            "git_dirty": dirty,
            "notes": notes,
        }
        return obs, info

    def log_command(self, action: np.ndarray) -> None:
        """Log a raw command at the input rate (teleop polling or the expert's decision)."""
        if not self.recording:
            raise RuntimeError("not recording")
        self.streams["action"].append(
            t_sim=self._t_sim(),
            t_wall=self._t_wall(),
            substep=self.env.substep,
            action=np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0),
        )

    def step(self, action: np.ndarray, log_command: bool = True):
        """Step the environment with ``action`` and log the tick. ``log_command=False`` when the
        caller already logged this command through ``log_command`` (teleop polling)."""
        if not self.recording:
            raise RuntimeError("not recording")
        if log_command:
            self.log_command(action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._log_tick(obs, info)
        return obs, reward, terminated, truncated, info

    def finish(self, discard: str | None = None, extra_attrs: dict | None = None) -> EpisodeRecord | None:
        """End the episode. Returns the record (and writes it if a dataset is attached), or None if discarded."""
        if not self.recording:
            raise RuntimeError("not recording")
        env = self.env
        env.substep_listeners.remove(self._on_substep)
        env.cameras.capture_listeners.remove(self._on_capture)
        self.recording = False
        if discard is not None:
            if self.dataset is not None:
                self.dataset.metadata.counts["discarded"] = self.dataset.metadata.counts.get("discarded", 0) + 1
                self.dataset.save_metadata()
            return None

        raw = {name: buf.arrays() for name, buf in self.streams.items()}
        raw_attrs = {name: dict(buf.attrs) for name, buf in self.streams.items()}
        n_steps = env.step_count
        result = sync_episode(raw, n_steps, env.cfg.substeps, env.cfg.physics_timestep, tuple(env.cfg.camera.names))
        priv = raw["priv"]
        attrs = dict(self.attrs)
        attrs.update(
            {
                "length": n_steps,
                "success": bool(priv["success_streak"][-1] >= env.cfg.success_steps),
                "terminated_early": bool(n_steps < env.cfg.max_steps),
                "wall_duration_s": self._t_wall(),
                "resync_count": int(priv["resync_count"][-1]),
                "alignment": result.alignment,
                "max_alignment_error_s": {name: a["max_s"] for name, a in result.alignment.items()},
                "stream_lengths": {name: len(buf) for name, buf in self.streams.items()},
                "schema_version": SCHEMA_VERSION,
            }
        )
        if extra_attrs:
            attrs.update(extra_attrs)
        record = EpisodeRecord(attrs=attrs, init=self.init, raw=raw, raw_attrs=raw_attrs, synced=result.arrays)
        if self.dataset is not None:
            self.dataset.write_episode(record)
        return record


def new_dataset(paths, name: str, env: PandaPickPlaceEnv, source: str, operator: str, input_device: str, notes: str = "") -> Dataset:
    """Create an empty dataset directory for this environment configuration."""
    return Dataset.create(paths.datasets / name, build_metadata(env, name, source, operator, input_device, notes))


def env_config_from_metadata(metadata: DatasetMetadata) -> EnvConfig:
    """Rebuild the EnvConfig a dataset was recorded with (nested dataclasses included)."""
    from panda_il.config import CameraConfig, IKConfig, RewardConfig

    cfg = dict(metadata.env_config)
    cfg["camera"] = CameraConfig(**{**cfg["camera"], "names": tuple(cfg["camera"]["names"])})
    cfg["ik"] = IKConfig(**{**cfg["ik"], "velocity_limits": tuple(cfg["ik"]["velocity_limits"])})
    cfg["reward"] = RewardConfig(**cfg["reward"])
    for key in ("workspace_min", "workspace_max", "cube_region_center"):
        cfg[key] = tuple(cfg[key])
    return EnvConfig(**cfg)
