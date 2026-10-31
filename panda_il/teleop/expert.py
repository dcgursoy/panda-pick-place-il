"""The scripted expert: a privileged state machine that picks the cube and places it in the bin.

It only exists to test the pipeline and for the human-versus-scripted ablation;
every episode it records is tagged ``source="scripted"``. It reads the cube pose
from the privileged state, moves the commanded TCP proportionally toward phase
targets through the same 7-D action interface a human uses, and can quantise its
actions to the keyboard's {-1, 0, +1} levels so the ablation can match action
distributions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from panda_il.config import EnvConfig
from panda_il.env.scene import BIN_CENTER

PHASES = ("align", "descend", "grasp", "lift", "carry", "lower", "release", "retreat", "done")


def wrap_to_quarter_turn(angle: float) -> float:
    """Map an angle to [-pi/4, pi/4): the cube's yaw modulo its 90-degree symmetry."""
    return (angle + np.pi / 4) % (np.pi / 2) - np.pi / 4


def hand_yaw(tcp_quat_wxyz: np.ndarray) -> float:
    """World yaw of the finger-closing axis (the TCP frame's y axis)."""
    w, x, y, z = tcp_quat_wxyz
    yx = 2 * (x * y - w * z)
    yy = 1 - 2 * (x * x + z * z)
    return float(np.arctan2(yy, yx))


@dataclass
class ExpertConfig:
    hover_height: float = 0.12  # above the cube centre, before descending
    grasp_height: float = 0.004  # TCP above the cube centre when closing
    lift_height: float = 0.15
    lower_height: float = 0.09  # cube bottom clears the 6 cm bin walls
    retreat_height: float = 0.22
    position_tolerance: float = 0.006
    yaw_tolerance: float = 0.02
    grasp_steps: int = 12
    release_steps: int = 8
    gain: float = 1.0  # action = gain * error / max_delta (clipped), so full speed beyond 1 cm
    quantize: str | None = None  # None or "keyboard" (actions in {-1, 0, 1} with a deadband)
    deadband: float = 0.3


class ScriptedExpert:
    def __init__(self, env_cfg: EnvConfig, cfg: ExpertConfig | None = None):
        self.env_cfg = env_cfg
        self.cfg = cfg or ExpertConfig()
        self.phase = "align"
        self.hold = 0
        self.phase_log: list[tuple[str, int]] = []
        self.step_index = 0

    def reset(self) -> None:
        self.phase = "align"
        self.hold = 0
        self.phase_log = [("align", 0)]
        self.step_index = 0

    def _action(self, tcp_pos, target_pos, yaw_error: float, close: bool) -> np.ndarray:
        cfg, env_cfg = self.cfg, self.env_cfg
        action = np.zeros(7)
        action[:3] = np.clip(
            cfg.gain * (np.asarray(target_pos) - tcp_pos) / env_cfg.max_delta_pos, -1, 1
        )
        # The TCP's z axis points down, so a positive rotation about it is a negative world yaw.
        action[5] = np.clip(-cfg.gain * yaw_error / env_cfg.max_delta_rot, -1, 1)
        action[6] = 1.0 if close else -1.0
        if cfg.quantize == "keyboard":
            quantized = np.where(np.abs(action[:6]) < cfg.deadband, 0.0, np.sign(action[:6]))
            action[:6] = quantized
        return action

    def act(
        self, tcp_pos: np.ndarray, tcp_quat: np.ndarray, cube_pos: np.ndarray, cube_yaw: float
    ) -> np.ndarray:
        """One action from the measured TCP pose and the privileged cube pose."""
        cfg = self.cfg
        tcp_pos = np.asarray(tcp_pos, dtype=float)
        cube_pos = np.asarray(cube_pos, dtype=float)
        yaw_error = wrap_to_quarter_turn(cube_yaw - hand_yaw(tcp_quat))
        close = self.phase in ("grasp", "lift", "carry", "lower")
        previous = self.phase

        if self.phase == "align":
            target = cube_pos + [0, 0, cfg.hover_height]
            if (
                np.linalg.norm(tcp_pos - target) < cfg.position_tolerance
                and abs(yaw_error) < cfg.yaw_tolerance
            ):
                self.phase = "descend"
        elif self.phase == "descend":
            target = cube_pos + [0, 0, cfg.grasp_height]
            yaw_error = 0.0
            if np.linalg.norm(tcp_pos - target) < cfg.position_tolerance:
                self.phase, self.hold = "grasp", 0
        elif self.phase == "grasp":
            target = cube_pos + [0, 0, cfg.grasp_height]
            yaw_error = 0.0
            self.hold += 1
            if self.hold >= cfg.grasp_steps:
                self.phase = "lift"
        elif self.phase == "lift":
            target = [tcp_pos[0], tcp_pos[1], cfg.lift_height]
            yaw_error = 0.0
            if tcp_pos[2] > cfg.lift_height - 0.01:
                self.phase = "carry"
        elif self.phase == "carry":
            target = [BIN_CENTER[0], BIN_CENTER[1], cfg.lift_height]
            yaw_error = 0.0
            if np.linalg.norm(tcp_pos[:2] - BIN_CENTER) < 0.008:
                self.phase = "lower"
        elif self.phase == "lower":
            target = [BIN_CENTER[0], BIN_CENTER[1], cfg.lower_height]
            yaw_error = 0.0
            if tcp_pos[2] < cfg.lower_height + 0.005:
                self.phase, self.hold = "release", 0
        elif self.phase == "release":
            target = [BIN_CENTER[0], BIN_CENTER[1], cfg.lower_height]
            yaw_error = 0.0
            self.hold += 1
            if self.hold >= cfg.release_steps:
                self.phase = "retreat"
        elif self.phase == "retreat":
            target = [BIN_CENTER[0], BIN_CENTER[1], cfg.retreat_height]
            yaw_error = 0.0
            if tcp_pos[2] > cfg.retreat_height - 0.01:
                self.phase = "done"
        else:  # done: hold still and let the success streak terminate the episode
            target = tcp_pos
            yaw_error = 0.0

        if self.phase != previous:
            self.phase_log.append((self.phase, self.step_index + 1))
        self.step_index += 1
        close = self.phase in ("grasp", "lift", "carry", "lower")
        return self._action(tcp_pos, target, yaw_error, close)

    def act_from_env(self, obs: dict, info: dict) -> np.ndarray:
        """Convenience for the environment's observation/info dictionaries (obs_mode state or both)."""
        state = obs["state"]
        cube_pos = state[22:25]
        cube_quat = state[25:29]
        w, x, y, z = cube_quat
        cube_yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        return self.act(info["tcp_measured"][:3], info["tcp_measured"][3:7], cube_pos, cube_yaw)
