"""Keyboard input for teleop: hold-to-move jogging plus single-key episode commands.

MuJoCo's passive viewer only reports key presses (no releases) and consumes
Space, Backspace, Tab and the F-keys itself, so keys are read with a global
``pynput`` listener instead, and only while the MuJoCo viewer window is in the
foreground (Windows; elsewhere the guard is a no-op). The listener thread does
nothing but maintain the set of pressed keys; the 20 Hz loop polls it.

Bindings (letters only, none of them used by the viewer):
    W / S      +x / -x        A / D      +y / -y        Q / E      +z / -z
    Up / Down  pitch          Left / Right  roll        [ / ]      yaw
    G  toggle gripper   N  start a new episode   K  keep (save)   X  discard
    R  reset the scene without recording          Esc  quit
"""

from __future__ import annotations

import sys
import threading

import numpy as np

JOG_KEYS = {
    "w": (0, +1.0),
    "s": (0, -1.0),
    "a": (1, +1.0),
    "d": (1, -1.0),
    "q": (2, +1.0),
    "e": (2, -1.0),
    "up": (4, +1.0),
    "down": (4, -1.0),
    "left": (3, +1.0),
    "right": (3, -1.0),
    "[": (5, +1.0),
    "]": (5, -1.0),
}
COMMAND_KEYS = {
    "g": "gripper",
    "n": "start",
    "k": "keep",
    "x": "discard",
    "r": "reset",
    "esc": "quit",
}


def viewer_window_in_foreground() -> bool:
    """True when the foreground window title contains 'MuJoCo' (Windows); True elsewhere."""
    if sys.platform != "win32":
        return True
    import ctypes

    user32 = ctypes.windll.user32
    handle = user32.GetForegroundWindow()
    length = user32.GetWindowTextLengthW(handle)
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(handle, buffer, length + 1)
    return "mujoco" in buffer.value.lower()


class KeyboardInput:
    """Collects pressed keys on a listener thread; ``poll()`` turns them into an action and commands."""

    def __init__(self, require_foreground: bool = True):
        self.require_foreground = require_foreground
        self._pressed: set[str] = set()
        self._lock = threading.Lock()
        self._listener = None
        self.gripper_closed = False
        self._seen_commands: set[str] = set()

    @staticmethod
    def _key_name(key) -> str | None:
        from pynput import keyboard

        if isinstance(key, keyboard.KeyCode) and key.char:
            return key.char.lower()
        if isinstance(key, keyboard.Key):
            return key.name
        return None

    def _on_press(self, key) -> None:
        name = self._key_name(key)
        if name is not None:
            with self._lock:
                self._pressed.add(name)

    def _on_release(self, key) -> None:
        name = self._key_name(key)
        if name is not None:
            self.release(name)

    def release(self, name: str) -> None:
        """Forget a key by name (what the listener does on key release)."""
        with self._lock:
            self._pressed.discard(name)
            self._seen_commands.discard(name)

    def start(self) -> None:
        from pynput import keyboard

        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.daemon = True
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def poll(self) -> tuple[np.ndarray, list[str]]:
        """The current jog action (7-D) and the single-key commands pressed since the last poll."""
        with self._lock:
            pressed = set(self._pressed)
            fresh = [
                name for name in pressed if name in COMMAND_KEYS and name not in self._seen_commands
            ]
            self._seen_commands.update(fresh)
        if self.require_foreground and not viewer_window_in_foreground():
            return self._idle_action(), []
        action = np.zeros(7)
        for name in pressed:
            if name in JOG_KEYS:
                axis, sign = JOG_KEYS[name]
                action[axis] += sign
        action[:6] = np.clip(action[:6], -1.0, 1.0)
        commands = [COMMAND_KEYS[name] for name in fresh]
        if "gripper" in commands:
            self.gripper_closed = not self.gripper_closed
        action[6] = 1.0 if self.gripper_closed else -1.0
        return action, [c for c in commands if c != "gripper"]

    def _idle_action(self) -> np.ndarray:
        action = np.zeros(7)
        action[6] = 1.0 if self.gripper_closed else -1.0
        return action
