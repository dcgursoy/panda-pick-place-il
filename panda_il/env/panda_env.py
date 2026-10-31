"""The gymnasium environment: a Franka Panda picking a cube and placing it in a bin.

Physics at 500 Hz, control at 20 Hz (25 substeps per step). The action is a
7-D delta TCP pose plus a gripper command in [-1, 1]; it is integrated into a
commanded TCP pose, turned into joint targets by differential IK and tracked by
the Panda's position actuators.

Observations (by ``obs_mode``):
  state  -> {"state": 29-D}          proprio + cube pose, for state policies and SAC
  image  -> {"proprio": 22-D, "image_front", "image_wrist"}   no cube pose
  both   -> all of the above

``info`` carries the privileged quantities the logger, the expert and the
evaluation need: the full simulator state, contacts, reward terms, the
commanded pose and joint targets, the success flag.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from panda_il.config import EnvConfig
from panda_il.env import task
from panda_il.env.cameras import CameraSet, ViewRenderer
from panda_il.env.ik import apply_delta, make_ik, pose_error, pose_from_vector, pose_to_vector
from panda_il.env.scene import (
    GRIPPER_CLOSED_CTRL,
    GRIPPER_OPEN_CTRL,
    SceneIds,
    compile_arm_model,
    compile_scene_model,
    scene_ids,
)
from panda_il.env.task import InitialCondition

PROPRIO_DIM = 22
STATE_DIM = 29
ACTION_DIM = 7
ENV_VERSION = (
    "1"  # bump when anything that changes recorded data changes (scene, timing, action semantics)
)

STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


class PandaPickPlaceEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, cfg: EnvConfig | None = None, render_mode: str | None = None):
        super().__init__()
        self.cfg = cfg or EnvConfig()
        self.render_mode = render_mode
        self.model = compile_scene_model()
        self.model.opt.timestep = self.cfg.physics_timestep
        self.data = mujoco.MjData(self.model)
        self.ids: SceneIds = scene_ids(self.model)
        self.arm_model = compile_arm_model()
        self.ik = make_ik(self.arm_model, self.cfg.ik)
        self.state_size = mujoco.mj_stateSize(self.model, STATE_SPEC)

        self.cameras: CameraSet | None = None
        if self.cfg.obs_mode in ("image", "both"):
            self.cameras = CameraSet(self.model, self.cfg.camera)
        self._view_renderer: ViewRenderer | None = None

        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)
        self.observation_space = self._make_observation_space()

        self.substep = 0
        self.step_count = 0
        self.success_streak = 0
        self.resync_count = 0
        self.gripper_closed = False
        self.target_pose = None
        self.tracking_lag = (0.0, 0.0)
        self.q_target = np.zeros(7)
        self.ic: InitialCondition | None = None
        self.substep_listeners: list = []  # called as f(substep) after every physics substep (the logger)

    # ------------------------------------------------------------------ spaces
    def _make_observation_space(self) -> spaces.Dict:
        parts: dict[str, spaces.Space] = {}
        if self.cfg.obs_mode in ("state", "both"):
            parts["state"] = spaces.Box(-np.inf, np.inf, shape=(STATE_DIM,), dtype=np.float32)
        if self.cfg.obs_mode in ("image", "both"):
            parts["proprio"] = spaces.Box(-np.inf, np.inf, shape=(PROPRIO_DIM,), dtype=np.float32)
            shape = (self.cfg.camera.height, self.cfg.camera.width, 3)
            for name in self.cfg.camera.names:
                parts[f"image_{name}"] = spaces.Box(0, 255, shape=shape, dtype=np.uint8)
        return spaces.Dict(parts)

    # ------------------------------------------------------------------ helpers
    def arm_qpos(self) -> np.ndarray:
        return self.data.qpos[self.ids.arm_qpos].copy()

    def arm_and_finger_qpos(self) -> np.ndarray:
        return np.concatenate(
            [self.data.qpos[self.ids.arm_qpos], self.data.qpos[self.ids.finger_qpos]]
        )

    def gripper_width(self) -> float:
        return float(self.data.qpos[self.ids.finger_qpos].sum())

    def tcp_pose_measured(self) -> tuple[np.ndarray, np.ndarray]:
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.ids.tcp_site])
        return self.data.site_xpos[self.ids.tcp_site].copy(), quat

    def proprio(self) -> np.ndarray:
        pos, quat = self.tcp_pose_measured()
        return np.concatenate(
            [
                self.data.qpos[self.ids.arm_qpos],
                self.data.qvel[self.ids.arm_dof],
                [self.gripper_width()],
                pos,
                quat,
            ]
        ).astype(np.float32)

    def state(self) -> np.ndarray:
        cube_pos, cube_quat, _ = task.cube_state(self.data, self.ids)
        return np.concatenate([self.proprio(), cube_pos, cube_quat]).astype(np.float32)

    def get_sim_state(self) -> np.ndarray:
        state = np.empty(self.state_size)
        mujoco.mj_getState(self.model, self.data, state, STATE_SPEC)
        return state

    def set_sim_state(self, state: np.ndarray) -> None:
        mujoco.mj_setState(self.model, self.data, np.asarray(state, dtype=np.float64), STATE_SPEC)
        mujoco.mj_forward(self.model, self.data)

    def snapshot(self) -> dict[str, Any]:
        """Everything needed to continue the episode bit-for-bit: simulator state plus the
        controller's own state (open-loop IK reference, commanded pose, counters)."""
        return {
            "sim_state": self.get_sim_state(),
            "ik_q": np.asarray(
                self.ik.configuration.q if hasattr(self.ik, "configuration") else self.ik.data.qpos
            ).copy(),
            "target_pose": pose_to_vector(self.target_pose),
            "q_target": self.q_target.copy(),
            "gripper_closed": self.gripper_closed,
            "substep": self.substep,
            "step_count": self.step_count,
            "success_streak": self.success_streak,
            "resync_count": self.resync_count,
            "ic": self.ic.to_dict() if self.ic is not None else None,
        }

    def restore(self, snap: dict[str, Any]) -> None:
        self.set_sim_state(snap["sim_state"])
        self.ik.sync(snap["ik_q"])
        self.target_pose = pose_from_vector(snap["target_pose"])
        self.q_target = np.asarray(snap["q_target"], dtype=float).copy()
        self.gripper_closed = bool(snap["gripper_closed"])
        self.substep = int(snap["substep"])
        self.step_count = int(snap["step_count"])
        self.success_streak = int(snap["success_streak"])
        self.resync_count = int(snap["resync_count"])
        self.ic = InitialCondition.from_dict(snap["ic"]) if snap.get("ic") else None
        if self.cameras is not None:
            self.cameras.reset(self.data)
            self.cameras.next_tick = self.substep // self.cfg.substeps + 1

    def _observation(self) -> dict[str, np.ndarray]:
        obs: dict[str, np.ndarray] = {}
        if self.cfg.obs_mode in ("state", "both"):
            obs["state"] = self.state()
        if self.cfg.obs_mode in ("image", "both"):
            obs["proprio"] = self.proprio()
            assert self.cameras is not None
            for name, image in self.cameras.deliver(self.substep).items():
                obs[f"image_{name}"] = image.image.copy()
        return obs

    def _info(self, success_now: bool, detail: dict, terms: dict[str, float]) -> dict[str, Any]:
        tcp_pos, tcp_quat = self.tcp_pose_measured()
        info: dict[str, Any] = {
            "success": bool(self.success_streak >= self.cfg.success_steps),
            "cube_in_bin_now": success_now,
            "success_detail": detail,
            "success_streak": self.success_streak,
            "reward_terms": terms,
            "contacts": task.contact_flags(self.model, self.data, self.ids),
            "substep": self.substep,
            "sim_time": float(self.data.time),
            "sim_state": self.get_sim_state(),
            "tcp_target": pose_to_vector(self.target_pose)
            if self.target_pose is not None
            else None,
            "tcp_measured": np.concatenate([tcp_pos, tcp_quat]),
            "tracking_lag": self.tracking_lag,
            "q_target": self.q_target.copy(),
            "ctrl": self.data.ctrl.copy(),
            "gripper_closed": self.gripper_closed,
            "resync_count": self.resync_count,
            "ik_iterations": getattr(self.ik, "iterations_last_solve", 0),
            "ic": self.ic.to_dict() if self.ic is not None else None,
        }
        if self.cameras is not None:
            info["frame_age_substeps"] = self.cameras.ages(self.substep)
        return info

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        ic = options.get("ic")
        if ic is None:
            ic = task.sample_in_square(
                self.np_random, self.cfg.cube_region_center, self.cfg.cube_region_half
            )
        elif isinstance(ic, dict):
            ic = InitialCondition.from_dict(ic)
        self.ic = ic

        mujoco.mj_resetDataKeyframe(self.model, self.data, self.ids.home_key)
        self.data.qpos[self.ids.cube_qpos : self.ids.cube_qpos + 7] = ic.cube_qpos()
        self.data.qvel[:] = 0.0
        self.data.ctrl[self.ids.arm_actuators] = self.data.qpos[self.ids.arm_qpos]
        self.data.ctrl[self.ids.gripper_actuator] = GRIPPER_OPEN_CTRL
        mujoco.mj_forward(self.model, self.data)

        self.ik.reset(self.arm_and_finger_qpos())
        self.target_pose = self.ik.tcp_pose()
        self.q_target = self.ik.q
        self.substep = 0
        self.step_count = 0
        self.success_streak = 0
        self.resync_count = 0
        self.tracking_lag = (0.0, 0.0)
        self.gripper_closed = False
        if self.cameras is not None:
            self.cameras.reset(self.data)

        success_now, detail = task.cube_resting_in_bin(self.model, self.data, self.ids)
        terms = task.reward_terms(self.model, self.data, self.ids, self.gripper_closed, False)
        return self._observation(), self._info(success_now, detail, terms)

    def step(self, action):
        if self.target_pose is None:
            raise RuntimeError("call reset() before step()")
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        cfg = self.cfg

        # Resync the open-loop IK reference if the arm fell too far behind it (blocked by contact).
        tcp_pos, tcp_quat = self.tcp_pose_measured()
        lag_pos, lag_rot = pose_error(self.ik.tcp_pose(), tcp_pos, tcp_quat)
        if lag_pos > cfg.ik.resync_position_m or lag_rot > cfg.ik.resync_orientation_rad:
            self.ik.sync(self.arm_and_finger_qpos())
            self.target_pose = self.ik.tcp_pose()
            self.resync_count += 1
        self.tracking_lag = (lag_pos, lag_rot)

        self.target_pose = apply_delta(
            self.target_pose,
            action[:3] * cfg.max_delta_pos,
            action[3:6] * cfg.max_delta_rot,
            cfg.workspace_min,
            cfg.workspace_max,
        )
        self.q_target = self.ik.solve(self.target_pose, cfg.control_dt)
        self.gripper_closed = bool(action[6] > 0.0)
        self.data.ctrl[self.ids.arm_actuators] = self.q_target
        self.data.ctrl[self.ids.gripper_actuator] = (
            GRIPPER_CLOSED_CTRL if self.gripper_closed else GRIPPER_OPEN_CTRL
        )

        for _ in range(cfg.substeps):
            mujoco.mj_step(self.model, self.data)
            self.substep += 1
            if self.cameras is not None:
                self.cameras.maybe_capture(self.data, self.substep)
            for listener in self.substep_listeners:
                listener(self.substep)
        self.step_count += 1

        success_now, detail = task.cube_resting_in_bin(self.model, self.data, self.ids)
        self.success_streak = self.success_streak + 1 if success_now else 0
        terminated = self.success_streak >= cfg.success_steps
        truncated = (not terminated) and self.step_count >= cfg.max_steps
        terms = task.reward_terms(self.model, self.data, self.ids, self.gripper_closed, terminated)
        reward = task.shaped_reward(terms, cfg.reward)
        return (
            self._observation(),
            float(reward),
            bool(terminated),
            bool(truncated),
            self._info(success_now, detail, terms),
        )

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        return self.render_view()

    def render_view(self, camera: str = "viewer") -> np.ndarray:
        if self._view_renderer is None:
            self._view_renderer = ViewRenderer(
                self.model, self.cfg.render_width, self.cfg.render_height
            )
        return self._view_renderer.render(self.data, camera)

    def close(self):
        if self.cameras is not None:
            self.cameras.close()
            self.cameras = None
        if self._view_renderer is not None:
            self._view_renderer.close()
            self._view_renderer = None


def make_env(cfg: EnvConfig | None = None, render_mode: str | None = None) -> PandaPickPlaceEnv:
    return PandaPickPlaceEnv(cfg, render_mode=render_mode)
