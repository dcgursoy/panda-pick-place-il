"""Gamepad input for teleop through pygame-ce (SDL2), which reads PlayStation and Xbox pads
without drivers. The mapping lives in configs/gamepad.yaml and is verified with
``python -m panda_il.teleop.teleop --calibrate``.

Axes give continuous actions (sticks for translation and yaw, triggers for z);
buttons give the single-key commands (gripper, start, keep, discard, reset).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import yaml

from panda_il.paths import CONFIGS_DIR

DEFAULT_MAPPING_PATH = CONFIGS_DIR / "gamepad.yaml"

# SDL2's default mapping for DualShock 4 / DualSense and Xbox controllers.
DEFAULT_MAPPING = {
    "deadzone": 0.15,
    "axes": {
        # action index: [axis index, sign]   (action: 0 x, 1 y, 2 z, 3 roll, 4 pitch, 5 yaw)
        "0": [1, -1.0],  # left stick up -> +x (toward the far edge of the table)
        "1": [0, -1.0],  # left stick left -> +y
        "5": [2, -1.0],  # right stick left -> +yaw
        "4": [3, -1.0],  # right stick up -> +pitch
    },
    "triggers": {"up": 5, "down": 4},  # R2 raises, L2 lowers (SDL reports triggers in [-1, 1])
    "buttons": {"gripper": 0, "start": 1, "keep": 3, "discard": 2, "reset": 6, "quit": 4},
    # DualShock/DualSense under SDL2: 0 cross, 1 circle, 2 square, 3 triangle, 4 share, 5 PS,
    # 6 options, 7 L3, 8 R3, 9 L1, 10 R1. Xbox: 0 A, 1 B, 2 X, 3 Y, 4 back, 6 start.
}


def load_mapping(path: Path | None = None) -> dict:
    path = path or DEFAULT_MAPPING_PATH
    if path.exists():
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    return DEFAULT_MAPPING


def save_default_mapping(path: Path | None = None) -> Path:
    path = path or DEFAULT_MAPPING_PATH
    path.write_text(yaml.safe_dump(DEFAULT_MAPPING, sort_keys=False), encoding="utf-8")
    return path


class GamepadInput:
    def __init__(self, mapping: dict | None = None):
        self.mapping = mapping or load_mapping()
        self.joystick = None
        self.gripper_closed = False
        self._previous_buttons: dict[int, bool] = {}

    @staticmethod
    def available() -> bool:
        try:
            import pygame
        except ImportError:
            return False
        os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        pygame.joystick.init()
        return pygame.joystick.get_count() > 0

    def start(self) -> str:
        import pygame

        os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("no gamepad connected")
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        return self.joystick.get_name()

    def stop(self) -> None:
        import pygame

        if self.joystick is not None:
            self.joystick.quit()
            self.joystick = None
        pygame.joystick.quit()

    def raw(self) -> tuple[list[float], list[int]]:
        import pygame

        pygame.event.pump()
        axes = [self.joystick.get_axis(i) for i in range(self.joystick.get_numaxes())]
        buttons = [self.joystick.get_button(i) for i in range(self.joystick.get_numbuttons())]
        return axes, buttons

    def poll(self) -> tuple[np.ndarray, list[str]]:
        axes, buttons = self.raw()
        deadzone = float(self.mapping.get("deadzone", 0.15))
        action = np.zeros(7)
        for action_index, (axis_index, sign) in self.mapping["axes"].items():
            if axis_index < len(axes):
                value = axes[axis_index]
                action[int(action_index)] = sign * value if abs(value) > deadzone else 0.0
        triggers = self.mapping.get("triggers", {})
        up = (
            (axes[triggers["up"]] + 1) / 2
            if "up" in triggers and triggers["up"] < len(axes)
            else 0.0
        )
        down = (
            (axes[triggers["down"]] + 1) / 2
            if "down" in triggers and triggers["down"] < len(axes)
            else 0.0
        )
        action[2] = np.clip(up - down, -1, 1) if abs(up - down) > deadzone else 0.0
        commands = []
        for command, button_index in self.mapping["buttons"].items():
            pressed = button_index < len(buttons) and bool(buttons[button_index])
            if pressed and not self._previous_buttons.get(button_index, False):
                commands.append(command)
            self._previous_buttons[button_index] = pressed
        if "gripper" in commands:
            self.gripper_closed = not self.gripper_closed
        action[6] = 1.0 if self.gripper_closed else -1.0
        return action, [c for c in commands if c != "gripper"]
