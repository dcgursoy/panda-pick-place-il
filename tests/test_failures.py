import numpy as np
import pytest

from panda_il.env.scene import BIN_CENTER
from panda_il.eval.failures import LABELS, EpisodeLog, label_episode

ON_TABLE = (0.5, -0.15, 0.02)
IN_BIN = (BIN_CENTER[0], BIN_CENTER[1], 0.025)


def step(cube=ON_TABLE, fingers=False, floor=False, closed=False, in_bin=False, success=False):
    return {
        "cube_pos": np.array(cube),
        "contacts": {"left_finger": fingers, "right_finger": fingers, "floor": floor},
        "gripper_closed": closed,
        "cube_in_bin_now": in_bin,
        "success": success,
        "resync_count": 0,
    }


def make_log(steps):
    log = EpisodeLog()
    for s in steps:
        log.append(s)
    return log


def carry(n=6):
    return [step(cube=(0.5, 0.0, 0.10), fingers=True, closed=True) for _ in range(n)]


def test_success_and_no_grasp_labels():
    assert label_episode(make_log([step()] * 5 + [step(success=True)]))["label"] == "success"
    assert label_episode(make_log([step()] * 5))["label"] == "never grasped"
    fell = [step()] * 3 + [step(cube=(0.5, -0.6, -0.3))]
    assert label_episode(make_log(fell))["label"] == "knocked off table"


def test_held_slipped_and_missed():
    verdict = label_episode(make_log([step()] * 2 + carry()))
    assert verdict["label"] == "held at timeout" and verdict["grasp_step"] == 2
    slipped = [step()] * 2 + carry() + [step(cube=(0.5, 0.0, 0.02), closed=True)] * 4
    verdict = label_episode(make_log(slipped))
    assert verdict["label"] == "slipped in transit" and verdict["release_step"] == 8
    missed = [step()] * 2 + carry() + [step(cube=(0.5, 0.0, 0.02))] * 4
    assert label_episode(make_log(missed))["label"] == "missed bin"


def test_contact_flicker_during_carry_is_not_a_release():
    flicker = carry(3) + [step(cube=(0.5, 0.0, 0.10), fingers=False, closed=True)] + carry(3)
    verdict = label_episode(make_log(flicker))
    assert verdict["label"] == "held at timeout" and verdict["release_step"] is None


def test_release_inside_bin_outcomes():
    inside_open = step(cube=IN_BIN, fingers=False, floor=True)
    base = [step()] * 2 + carry()
    settled = step(cube=IN_BIN, floor=True, in_bin=True)
    assert label_episode(make_log(base + [inside_open] * 4 + [settled]))["label"] == (
        "in bin at timeout"
    )
    out = step(cube=(BIN_CENTER[0] + 0.2, BIN_CENTER[1], 0.02))
    assert label_episode(make_log(base + [inside_open] * 4 + [out]))["label"] == "knocked out"
    rim = step(cube=(BIN_CENTER[0], BIN_CENTER[1], 0.08), floor=False)
    assert label_episode(make_log(base + [inside_open] * 4 + [rim]))["label"] == "on rim"
    moving = step(cube=IN_BIN, floor=True)
    assert label_episode(make_log(base + [inside_open] * 4 + [moving]))["label"] == "other"


def test_all_labels_known():
    assert set(LABELS) >= {"success", "never grasped", "held at timeout", "slipped in transit"}
    with pytest.raises((IndexError, ValueError)):
        label_episode(make_log([]))
