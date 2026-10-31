import numpy as np
import pytest

from panda_il.config import EnvConfig
from panda_il.data import Dataset, EpisodeRecorder, new_dataset
from panda_il.data.validate import validate_dataset
from panda_il.env import make_env
from panda_il.paths import Paths
from panda_il.teleop import gamepad
from panda_il.teleop.keyboard import KeyboardInput
from panda_il.teleop.teleop import AutopilotInput, TeleopSession


def test_keyboard_jog_and_commands_without_a_listener():
    keys = KeyboardInput(require_foreground=False)
    keys._pressed = {"w", "a", "q", "]"}
    action, commands = keys.poll()
    assert np.array_equal(action, [1, 1, 1, 0, 0, -1, -1])
    assert commands == []
    keys._pressed = {"g", "n"}
    action, commands = keys.poll()
    assert keys.gripper_closed and action[6] == 1.0 and commands == ["start"]
    action, commands = keys.poll()  # still held: no repeat of single-key commands
    assert commands == []
    from pynput.keyboard import KeyCode

    keys._on_release(KeyCode.from_char("n"))  # released -> can fire again
    keys._pressed = {"n", "k", "x"}
    _, commands = keys.poll()
    assert set(commands) == {"start", "keep", "discard"}


def test_opposite_keys_cancel_and_are_clipped():
    keys = KeyboardInput(require_foreground=False)
    keys._pressed = {"w", "s", "up"}
    action, _ = keys.poll()
    assert action[0] == 0 and action[4] == 1.0


def test_gamepad_mapping_defaults_and_file(tmp_path):
    path = gamepad.save_default_mapping(tmp_path / "gamepad.yaml")
    loaded = gamepad.load_mapping(path)
    assert loaded == gamepad.DEFAULT_MAPPING
    assert gamepad.load_mapping(tmp_path / "missing.yaml") == gamepad.DEFAULT_MAPPING


@pytest.mark.render
def test_autopilot_session_records_valid_episodes(tmp_path):
    paths = Paths(tmp_path)
    paths.make_dirs()
    env = make_env(EnvConfig(obs_mode="both"))
    dataset = new_dataset(
        paths, "auto", env, source="scripted", operator="autopilot", input_device="autopilot"
    )
    recorder = EpisodeRecorder(env, "scripted", "autopilot", "autopilot", dataset=dataset)
    session = TeleopSession(
        env, recorder, AutopilotInput(env, episodes=1), use_viewer=False, seed_start=5
    )
    assert session.run() == 0
    env.close()
    reopened = Dataset(dataset.directory)
    assert reopened.episode_ids() == [0]
    attrs = reopened.read_attrs(0)
    assert attrs["success"] and attrs["source"] == "scripted"
    # commands were logged at the 50 Hz poll rate, several per control step, all before their tick
    assert attrs["stream_lengths"]["action"] > attrs["length"]
    assert attrs["max_alignment_error_s"]["action"] <= 1.0 / 20
    report = validate_dataset(reopened, check_held_out=False)
    assert report.ok, [str(i) for i in report.issues]
