"""Differential inverse kinematics: commanded TCP pose -> joint position targets.

Two interchangeable backends:

- ``MinkIK``: a quadratic program (mink, daqp solver) with a frame task on the
  ``tcp`` site, a weak posture task that keeps the elbow near the home pose,
  joint-limit and velocity-limit constraints.
- ``DampedLeastSquaresIK``: the classic Jacobian pseudo-inverse with damping and
  a nullspace posture term, needing only MuJoCo; kept as a fallback and as a
  cross-check with the same tests.

Both keep their own kinematic configuration (an "open-loop reference"), as
mink's examples do, and expose ``drift`` so the environment can resync them to
the measured joints when contacts pull the arm away from the reference.
"""

from __future__ import annotations

import mink
import mujoco
import numpy as np

from panda_il.config import IKConfig
from panda_il.env.scene import TCP_SITE

ARM_DOFS = 7
ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))  # names in the arm-only model


def se3_from_site(data: mujoco.MjData, site_id: int) -> mink.SE3:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, data.site_xmat[site_id])
    return mink.SE3.from_rotation_and_translation(
        mink.SO3(wxyz=quat), data.site_xpos[site_id].copy()
    )


def pose_to_vector(pose: mink.SE3) -> np.ndarray:
    """(x, y, z, qw, qx, qy, qz), the same layout as the measured TCP pose in observations."""
    return np.concatenate([pose.translation(), pose.rotation().wxyz])


def pose_from_vector(vector: np.ndarray) -> mink.SE3:
    vector = np.asarray(vector, dtype=float)
    return mink.SE3.from_rotation_and_translation(mink.SO3(wxyz=vector[3:7]), vector[:3])


def pose_error(
    reference: mink.SE3, position: np.ndarray, quat_wxyz: np.ndarray
) -> tuple[float, float]:
    """Distance (m) and rotation angle (rad) between a reference pose and a measured pose."""
    dpos = float(np.linalg.norm(reference.translation() - np.asarray(position)))
    relative = reference.rotation().inverse() @ mink.SO3(wxyz=np.asarray(quat_wxyz, dtype=float))
    return dpos, float(np.linalg.norm(relative.log()))


def apply_delta(pose: mink.SE3, delta_pos, delta_rot, workspace_min, workspace_max) -> mink.SE3:
    """Translate in the world frame, rotate about the TCP's own axes, clamp the position to the workspace box."""
    moved = mink.SE3.from_translation(np.asarray(delta_pos, dtype=float)) @ pose
    rotated = moved @ mink.SE3.from_rotation(mink.SO3.exp(np.asarray(delta_rot, dtype=float)))
    position = np.clip(rotated.translation(), workspace_min, workspace_max)
    return mink.SE3.from_rotation_and_translation(rotated.rotation(), position)


class MinkIK:
    def __init__(self, arm_model: mujoco.MjModel, cfg: IKConfig):
        self.cfg = cfg
        self.model = arm_model
        self.configuration = mink.Configuration(arm_model)
        self.task = mink.FrameTask(
            TCP_SITE, "site", position_cost=1.0, orientation_cost=1.0, lm_damping=cfg.lm_damping
        )
        self.posture = mink.PostureTask(arm_model, cost=cfg.posture_cost)
        self.limits = [
            mink.ConfigurationLimit(
                arm_model,
                gain=cfg.configuration_limit_gain,
                min_distance_from_limits=cfg.min_distance_from_limits,
            ),
            mink.VelocityLimit(
                arm_model, dict(zip(ARM_JOINT_NAMES, cfg.velocity_limits, strict=True))
            ),
        ]
        self.iterations_last_solve = 0

    def reset(self, q: np.ndarray) -> None:
        """Sync to the measured joints (9: arm + fingers) and anchor the posture target there."""
        self.sync(q)
        self.posture.set_target_from_configuration(self.configuration)

    def sync(self, q: np.ndarray) -> None:
        self.configuration.update(np.asarray(q, dtype=float))

    @property
    def q(self) -> np.ndarray:
        return self.configuration.q[:ARM_DOFS].copy()

    def drift(self, q_measured: np.ndarray) -> float:
        return float(
            np.max(np.abs(self.configuration.q[:ARM_DOFS] - np.asarray(q_measured)[:ARM_DOFS]))
        )

    def tcp_pose(self) -> mink.SE3:
        return self.configuration.get_transform_frame_to_world(TCP_SITE, "site")

    def solve(self, target: mink.SE3, dt: float) -> np.ndarray:
        """Move the reference configuration toward ``target`` and return the 7 arm joint targets."""
        self.task.set_target(target)
        cfg = self.cfg
        for iteration in range(1, cfg.max_iterations + 1):
            velocity = mink.solve_ik(
                self.configuration,
                [self.task, self.posture],
                dt,
                cfg.solver,
                damping=cfg.damping,
                limits=self.limits,
            )
            self.configuration.integrate_inplace(velocity, dt)
            error = self.task.compute_error(self.configuration)
            self.iterations_last_solve = iteration
            if (
                np.linalg.norm(error[:3]) < cfg.position_tolerance
                and np.linalg.norm(error[3:]) < cfg.orientation_tolerance
            ):
                break
        return self.q

    def error(self, target: mink.SE3) -> tuple[float, float]:
        """Position (m) and orientation (rad) error between the reference TCP and ``target``."""
        self.task.set_target(target)
        error = self.task.compute_error(self.configuration)
        return float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:]))


class DampedLeastSquaresIK:
    """Fallback with the same interface, after Zakka's mjctrl diffik_nullspace.py."""

    def __init__(self, arm_model: mujoco.MjModel, cfg: IKConfig):
        self.cfg = cfg
        self.model = arm_model
        self.data = mujoco.MjData(arm_model)
        self.site_id = arm_model.site(TCP_SITE).id
        self.dof_ids = np.array([arm_model.joint(name).dofadr[0] for name in ARM_JOINT_NAMES])
        self.qpos_ids = np.array([arm_model.joint(name).qposadr[0] for name in ARM_JOINT_NAMES])
        self.joint_range = arm_model.jnt_range[
            [arm_model.joint(name).id for name in ARM_JOINT_NAMES]
        ]
        self.velocity_limits = np.asarray(cfg.velocity_limits, dtype=float)
        self.q_posture = np.zeros(ARM_DOFS)
        self.iterations_last_solve = 0

    def reset(self, q: np.ndarray) -> None:
        self.sync(q)
        self.q_posture = self.q

    def sync(self, q: np.ndarray) -> None:
        self.data.qpos[:] = np.asarray(q, dtype=float)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    @property
    def q(self) -> np.ndarray:
        return self.data.qpos[self.qpos_ids].copy()

    def drift(self, q_measured: np.ndarray) -> float:
        return float(np.max(np.abs(self.q - np.asarray(q_measured)[:ARM_DOFS])))

    def tcp_pose(self) -> mink.SE3:
        return se3_from_site(self.data, self.site_id)

    def _twist_to(self, target: mink.SE3) -> np.ndarray:
        """World-frame twist (translation, rotation vector) from the current TCP to the target."""
        twist = np.zeros(6)
        twist[:3] = target.translation() - self.data.site_xpos[self.site_id]
        site_quat = np.zeros(4)
        mujoco.mju_mat2Quat(site_quat, self.data.site_xmat[self.site_id])
        site_quat_conj = np.zeros(4)
        mujoco.mju_negQuat(site_quat_conj, site_quat)
        error_quat = np.zeros(4)
        mujoco.mju_mulQuat(error_quat, target.rotation().wxyz, site_quat_conj)
        mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
        return twist

    def solve(self, target: mink.SE3, dt: float) -> np.ndarray:
        cfg = self.cfg
        jac = np.zeros((6, self.model.nv))
        eye = np.eye(ARM_DOFS)
        for iteration in range(1, cfg.max_iterations + 1):
            twist = self._twist_to(target)
            self.iterations_last_solve = iteration
            if (
                np.linalg.norm(twist[:3]) < cfg.position_tolerance
                and np.linalg.norm(twist[3:]) < cfg.orientation_tolerance
            ):
                break
            mujoco.mj_jacSite(self.model, self.data, jac[:3], jac[3:], self.site_id)
            j = jac[:, self.dof_ids]
            dq = j.T @ np.linalg.solve(j @ j.T + cfg.dls_damping * np.eye(6), twist)
            dq += (eye - np.linalg.pinv(j) @ j) @ (
                cfg.dls_nullspace_gain * (self.q_posture - self.q) * dt
            )
            step_limit = self.velocity_limits * dt
            scale = np.max(np.abs(dq) / step_limit)
            if scale > 1.0:
                dq /= scale
            q = np.clip(self.q + dq, self.joint_range[:, 0], self.joint_range[:, 1])
            self.data.qpos[self.qpos_ids] = q
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
        return self.q

    def error(self, target: mink.SE3) -> tuple[float, float]:
        twist = self._twist_to(target)
        return float(np.linalg.norm(twist[:3])), float(np.linalg.norm(twist[3:]))


def make_ik(arm_model: mujoco.MjModel, cfg: IKConfig) -> MinkIK | DampedLeastSquaresIK:
    if cfg.backend == "mink":
        return MinkIK(arm_model, cfg)
    if cfg.backend == "dls":
        return DampedLeastSquaresIK(arm_model, cfg)
    raise ValueError(f"unknown IK backend {cfg.backend!r}")
