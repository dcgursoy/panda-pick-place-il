"""The task: initial conditions, the success predicate, contacts and the shaped reward."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from panda_il.config import RewardConfig
from panda_il.env.scene import BIN_CENTER, BIN_FLOOR_TOP, BIN_INNER_HALF, CUBE_HALF, SceneIds

CUBE_REST_Z = CUBE_HALF  # on the table
SUCCESS_Z_MIN = BIN_FLOOR_TOP
SUCCESS_Z_MAX = BIN_FLOOR_TOP + CUBE_HALF + 0.01
SUCCESS_MAX_SPEED = 0.01  # m/s
LIFT_HEIGHT = 0.05  # cube z above which the "lift" reward term is earned


@dataclass(frozen=True)
class InitialCondition:
    """Cube start pose on the table: position in metres, yaw in radians."""

    x: float
    y: float
    yaw: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> InitialCondition:
        return cls(float(d["x"]), float(d["y"]), float(d["yaw"]))

    def cube_qpos(self) -> np.ndarray:
        """The free joint's 7 qpos values (position, w-x-y-z quaternion)."""
        half = self.yaw / 2.0
        return np.array([self.x, self.y, CUBE_REST_Z, np.cos(half), 0.0, 0.0, np.sin(half)])


def sample_in_square(rng: np.random.Generator, center, half: float) -> InitialCondition:
    x = center[0] + rng.uniform(-half, half)
    y = center[1] + rng.uniform(-half, half)
    return InitialCondition(float(x), float(y), float(rng.uniform(-np.pi, np.pi)))


def sample_in_band(
    rng: np.random.Generator, center, inner_half: float, outer_half: float
) -> InitialCondition:
    """Uniform by area in the square ring between two concentric squares."""
    while True:
        ic = sample_in_square(rng, center, outer_half)
        if max(abs(ic.x - center[0]), abs(ic.y - center[1])) > inner_half:
            return ic


def ic_distance(a: InitialCondition, b: InitialCondition) -> tuple[float, float]:
    """Position distance (m) and yaw distance (rad, modulo the cube's 90-degree symmetry)."""
    dpos = float(np.hypot(a.x - b.x, a.y - b.y))
    dyaw = abs((a.yaw - b.yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4)
    return dpos, float(dyaw)


def contact_flags(model: mujoco.MjModel, data: mujoco.MjData, ids: SceneIds) -> dict[str, bool]:
    """Which things the cube touches right now, from MuJoCo's contact list."""
    cube = ids.cube_geom
    finger_geoms = set(ids.finger_geoms.tolist())
    wall_geoms = set(ids.bin_wall_geoms.tolist())
    flags = {
        "floor": False,
        "wall": False,
        "table": False,
        "left_finger": False,
        "right_finger": False,
    }
    left_geoms = set()
    for geom in finger_geoms:
        if model.body(model.geom_bodyid[geom]).name.endswith("left_finger"):
            left_geoms.add(geom)
    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = contact.geom1, contact.geom2
        if cube not in (g1, g2):
            continue
        other = g2 if g1 == cube else g1
        if other == ids.bin_floor_geom:
            flags["floor"] = True
        elif other in wall_geoms:
            flags["wall"] = True
        elif other == ids.table_geom:
            flags["table"] = True
        elif other in finger_geoms:
            if other in left_geoms:
                flags["left_finger"] = True
            else:
                flags["right_finger"] = True
    return flags


def cube_state(data: mujoco.MjData, ids: SceneIds) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cube position (3), quaternion (4, w-x-y-z) and linear velocity (3, world frame)."""
    qpos = data.qpos[ids.cube_qpos : ids.cube_qpos + 7]
    linvel = data.qvel[ids.cube_dof : ids.cube_dof + 3]
    return qpos[:3].copy(), qpos[3:].copy(), linvel.copy()


def cube_resting_in_bin(
    model: mujoco.MjModel, data: mujoco.MjData, ids: SceneIds
) -> tuple[bool, dict]:
    """The success predicate: centre inside the inner box, on the bin floor, not held, nearly still."""
    pos, _, linvel = cube_state(data, ids)
    flags = contact_flags(model, data, ids)
    inside_xy = bool(np.all(np.abs(pos[:2] - BIN_CENTER) <= BIN_INNER_HALF))
    z_ok = SUCCESS_Z_MIN <= pos[2] <= SUCCESS_Z_MAX
    held = flags["left_finger"] or flags["right_finger"]
    still = float(np.linalg.norm(linvel)) < SUCCESS_MAX_SPEED
    ok = inside_xy and z_ok and flags["floor"] and not held and still
    detail = {
        "inside_xy": inside_xy,
        "z_ok": z_ok,
        "on_floor": flags["floor"],
        "held": held,
        "still": still,
    }
    return ok, detail


def reward_terms(
    model: mujoco.MjModel, data: mujoco.MjData, ids: SceneIds, gripper_closed: bool, success: bool
) -> dict[str, float]:
    pos, _, _ = cube_state(data, ids)
    tcp = data.site_xpos[ids.tcp_site]
    flags = contact_flags(model, data, ids)
    grasped = gripper_closed and flags["left_finger"] and flags["right_finger"]
    return {
        "reach": -float(np.linalg.norm(tcp - pos)),
        "grasp": 1.0 if grasped else 0.0,
        "lift": 1.0 if pos[2] > LIFT_HEIGHT else 0.0,
        "place": -float(np.linalg.norm(pos[:2] - BIN_CENTER)),
        "success": 1.0 if success else 0.0,
    }


def shaped_reward(terms: dict[str, float], weights: RewardConfig) -> float:
    return (
        weights.reach * terms["reach"]
        + weights.grasp * terms["grasp"]
        + weights.lift * terms["lift"]
        + weights.place * terms["place"]
        + weights.success * terms["success"]
    )
