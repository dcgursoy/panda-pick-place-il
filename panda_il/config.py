"""Configuration dataclasses for every stage. tyro turns them into command lines.

Only environment-side settings live here for now; training and evaluation
configs are added with their milestones.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CameraConfig:
    """The observation cameras and their timing model (all times in physics substeps)."""

    names: tuple[str, ...] = ("front", "wrist")
    width: int = 96
    height: int = 96
    # Each camera captures on its own clock: one frame per control period, taken
    # `capture_offset_substeps` substeps relative to the control tick (negative =
    # before the tick) and delivered `latency_substeps` later. The defaults capture
    # 10 substeps (20 ms) before each tick and arrive 2 substeps before it, so a
    # policy always sees a 20 ms-old image, in teleop and in evaluation alike.
    period_substeps: int = 25
    capture_offset_substeps: int = -10
    latency_substeps: int = 8

    def __post_init__(self) -> None:
        if self.capture_offset_substeps + self.latency_substeps >= 0:
            raise ValueError(
                "capture_offset + latency must be negative: the frame has to arrive strictly "
                "before the control tick (an arrival exactly at the tick would be a timing tie)"
            )
        if -self.capture_offset_substeps >= self.period_substeps:
            raise ValueError("capture offset must be smaller than one control period")


@dataclass
class IKConfig:
    """Differential inverse kinematics from the commanded TCP pose to joint targets."""

    backend: str = "mink"  # "mink" (QP with limits) or "dls" (damped least squares fallback)
    solver: str = "daqp"
    damping: float = 1e-3
    lm_damping: float = 1.0
    posture_cost: float = 1e-3
    max_iterations: int = 3
    position_tolerance: float = 2e-4  # m
    orientation_tolerance: float = 1e-3  # rad
    configuration_limit_gain: float = 0.95
    min_distance_from_limits: float = 1e-3
    # Franka's published joint velocity limits (rad/s), joints 1-4 and 5-7.
    velocity_limits: tuple[float, ...] = (2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61)
    # The IK configuration is an open-loop reference. The position actuators lag it by
    # up to ~1.5 cm at full speed; when the measured TCP falls further behind than this
    # (an arm blocked by contact), the reference is resynced to the measured joints and the
    # commanded pose re-anchored.
    resync_position_m: float = 0.04
    resync_orientation_rad: float = 0.3
    # Damped least squares fallback only.
    dls_damping: float = 1e-4
    dls_nullspace_gain: float = 5.0


@dataclass
class RewardConfig:
    """Weights of the shaped reward used by the SAC baseline only."""

    reach: float = 1.0
    grasp: float = 1.0
    lift: float = 1.0
    place: float = 1.0
    success: float = 10.0


@dataclass
class EnvConfig:
    obs_mode: str = "state"  # "state", "image" or "both"
    physics_timestep: float = 0.002
    substeps: int = 25  # 500 Hz physics / 20 Hz control
    max_steps: int = 300  # provisional; re-fixed after demo collection (DESIGN.md decision 6)
    max_delta_pos: float = 0.01  # m per control step at |action| = 1
    max_delta_rot: float = 0.05  # rad per control step at |action| = 1
    workspace_min: tuple[float, float, float] = (0.25, -0.40, 0.01)
    workspace_max: tuple[float, float, float] = (0.75, 0.40, 0.60)
    cube_region_center: tuple[float, float] = (0.50, -0.15)
    cube_region_half: float = 0.08  # 16 x 16 cm square
    success_steps: int = 10  # consecutive control steps the success predicate must hold
    render_width: int = 480
    render_height: int = 360
    camera: CameraConfig = field(default_factory=CameraConfig)
    ik: IKConfig = field(default_factory=IKConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self) -> None:
        if self.obs_mode not in ("state", "image", "both"):
            raise ValueError(f"obs_mode must be state, image or both, got {self.obs_mode!r}")
        if self.camera.period_substeps != self.substeps:
            raise ValueError("camera period must equal the number of substeps per control step")

    @property
    def control_hz(self) -> float:
        return 1.0 / (self.physics_timestep * self.substeps)

    @property
    def control_dt(self) -> float:
        return self.physics_timestep * self.substeps
