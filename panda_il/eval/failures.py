"""Failure labels for evaluation episodes, from the privileged per-step log.

An ordered decision tree (DESIGN.md, section 7.6) over what the simulator knew
at every control step: the cube position, which of the fingers, the bin floor
and the table it touched, whether the gripper was commanded closed, and the
success predicate. A *grasp* is the cube lifted more than ``LIFT_HEIGHT`` above
its resting height while both fingers touch it; a *release* is the first loss
of two-finger contact after the last grasp that lasts ``RELEASE_STEPS`` steps
(single-step contact flicker during a carry is ignored).

    success            the env's success predicate held for its settle window
    never grasped      no grasp event; the cube is still on the table
    knocked off table  no grasp event; the cube fell below the table top
    held at timeout    grasped and still gripped when the episode hit its limit
    slipped in transit released while the gripper was still commanded closed
    missed bin         released (gripper opened) with the cube outside the bin's inner box
    knocked out        released inside the inner box, cube outside it at the end
    on rim             released inside, still inside at the end, but not resting on the floor
    in bin at timeout  released inside and resting there, but the settle window was not complete
    other              released inside, on the floor, yet not counted as success (moving or touched)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from panda_il.env.scene import BIN_CENTER, BIN_INNER_HALF, CUBE_HALF

LIFT_HEIGHT = 0.02
RELEASE_STEPS = 3
LABELS = (
    "success",
    "never grasped",
    "knocked off table",
    "held at timeout",
    "slipped in transit",
    "missed bin",
    "knocked out",
    "on rim",
    "in bin at timeout",
    "other",
)


@dataclass
class EpisodeLog:
    """What the failure tree needs, appended from the env's ``info`` after reset and every step."""

    cube_pos: list[np.ndarray] = field(default_factory=list)
    left_finger: list[bool] = field(default_factory=list)
    right_finger: list[bool] = field(default_factory=list)
    floor: list[bool] = field(default_factory=list)
    gripper_closed: list[bool] = field(default_factory=list)
    in_bin: list[bool] = field(default_factory=list)
    success: list[bool] = field(default_factory=list)
    resync_count: list[int] = field(default_factory=list)

    def append(self, info: dict[str, Any]) -> None:
        contacts = info["contacts"]
        self.cube_pos.append(np.array(info["cube_pos"], dtype=float))
        self.left_finger.append(bool(contacts["left_finger"]))
        self.right_finger.append(bool(contacts["right_finger"]))
        self.floor.append(bool(contacts["floor"]))
        self.gripper_closed.append(bool(info["gripper_closed"]))
        self.in_bin.append(bool(info["cube_in_bin_now"]))
        self.success.append(bool(info["success"]))
        self.resync_count.append(int(info["resync_count"]))

    def __len__(self) -> int:
        return len(self.cube_pos)


def in_inner_box(xy: np.ndarray) -> bool:
    return bool(np.all(np.abs(np.asarray(xy)[:2] - BIN_CENTER) <= BIN_INNER_HALF))


def label_episode(log: EpisodeLog) -> dict[str, Any]:
    """Label one episode. Returns the label with the grasp / release steps it found."""
    cube = np.stack(log.cube_pos)
    both = np.asarray(log.left_finger) & np.asarray(log.right_finger)
    lifted = cube[:, 2] > CUBE_HALF + LIFT_HEIGHT
    held = both & lifted
    out: dict[str, Any] = {
        "label": "other",
        "grasp_step": None,
        "release_step": None,
        "resync": bool(log.resync_count[-1] > 0),
        "cube_final": cube[-1].round(4).tolist(),
    }
    if log.success[-1]:
        out["label"] = "success"
    grasp_steps = np.flatnonzero(held)
    if len(grasp_steps) == 0:
        if out["label"] != "success":
            out["label"] = "knocked off table" if cube[-1, 2] < 0.0 else "never grasped"
        return out
    out["grasp_step"] = int(grasp_steps[0])
    last_held = int(grasp_steps[-1])
    release = None
    lost = ~both
    for t in range(last_held + 1, len(log)):
        if lost[t] and (bool(np.all(lost[t : t + RELEASE_STEPS])) or t + RELEASE_STEPS > len(log)):
            release = t
            break
    out["release_step"] = release
    if out["label"] == "success":
        return out
    if release is None:
        out["label"] = "held at timeout"
    elif log.gripper_closed[release]:
        out["label"] = "slipped in transit"
    elif not in_inner_box(cube[release]):
        out["label"] = "missed bin"
    elif log.in_bin[-1]:
        out["label"] = "in bin at timeout"
    elif not in_inner_box(cube[-1]):
        out["label"] = "knocked out"
    elif not log.floor[-1]:
        out["label"] = "on rim"
    else:
        out["label"] = "other"
    return out
