"""Observation cameras with an explicit timing model.

Each camera captures one frame per control period at a fixed substep offset and
delivers it a fixed number of substeps later; the environment only hands out
frames that have arrived. All times are integer physics substeps, so there are
no floating-point ties and the delivered frame age is identical in every episode.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from panda_il.config import CameraConfig


@dataclass
class CameraClock:
    period: int
    capture_offset: int  # relative to the control tick, negative = before
    latency: int

    def capture_substep(self, tick: int) -> int:
        """Substep at which the frame meant for control tick ``tick`` (>= 1) is captured."""
        return tick * self.period + self.capture_offset

    def arrival_substep(self, tick: int) -> int:
        return self.capture_substep(tick) + self.latency

    def delivered_frame_age(self) -> int:
        """Age in substeps of the frame a policy sees at any control tick >= 1."""
        return -self.capture_offset


@dataclass
class Frame:
    image: np.ndarray
    capture_substep: int
    arrival_substep: int


def camera_intrinsics(
    model: mujoco.MjModel, camera: str, width: int, height: int
) -> dict[str, float]:
    """Pinhole intrinsics of a MuJoCo camera: fovy is the vertical field of view."""
    fovy = float(model.cam_fovy[model.camera(camera).id])
    focal = (height / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)
    return {
        "fovy_deg": fovy,
        "fx": focal,
        "fy": focal,
        "cx": width / 2.0,
        "cy": height / 2.0,
        "width": width,
        "height": height,
    }


def project_point(
    model: mujoco.MjModel, data: mujoco.MjData, camera: str, point, width: int, height: int
):
    """Pixel coordinates (u right, v down) and depth of a world point; depth <= 0 means behind the camera."""
    cam_id = model.camera(camera).id
    rotation = data.cam_xmat[cam_id].reshape(3, 3)
    local = rotation.T @ (np.asarray(point, dtype=float) - data.cam_xpos[cam_id])
    depth = -local[2]
    k = camera_intrinsics(model, camera, width, height)
    if depth <= 0:
        return np.nan, np.nan, depth
    u = k["cx"] + k["fx"] * local[0] / depth
    v = k["cy"] - k["fy"] * local[1] / depth
    return float(u), float(v), float(depth)


class CameraSet:
    """Renders the observation cameras on their clocks and delivers arrived frames."""

    def __init__(self, model: mujoco.MjModel, cfg: CameraConfig):
        self.model = model
        self.cfg = cfg
        self.names = tuple(cfg.names)
        self.clock = CameraClock(
            cfg.period_substeps, cfg.capture_offset_substeps, cfg.latency_substeps
        )
        self.renderer = mujoco.Renderer(model, cfg.height, cfg.width)
        self.latest: dict[str, Frame] = {}
        self.pending: list[tuple[str, Frame]] = []
        self.next_tick = 1

    def _render(self, data: mujoco.MjData, name: str) -> np.ndarray:
        self.renderer.update_scene(data, camera=name)
        return self.renderer.render().copy()

    def reset(self, data: mujoco.MjData) -> None:
        """The reset observation is a zero-latency render at substep 0."""
        mujoco.mj_forward(self.model, data)
        self.pending = []
        self.latest = {name: Frame(self._render(data, name), 0, 0) for name in self.names}
        self.next_tick = 1

    def maybe_capture(self, data: mujoco.MjData, substep: int) -> bool:
        """Call after every physics substep; renders when a capture is due. Returns True if it rendered."""
        if substep != self.clock.capture_substep(self.next_tick):
            return False
        mujoco.mj_forward(
            self.model, data
        )  # idempotent since MuJoCo 3.3.6: positions for rendering, state untouched
        arrival = self.clock.arrival_substep(self.next_tick)
        for name in self.names:
            self.pending.append((name, Frame(self._render(data, name), substep, arrival)))
        self.next_tick += 1
        return True

    def deliver(self, substep: int) -> dict[str, Frame]:
        """Promote frames that have arrived by ``substep`` and return the latest frame per camera."""
        still_pending = []
        for name, frame in self.pending:
            if frame.arrival_substep <= substep:
                if (
                    name not in self.latest
                    or frame.capture_substep >= self.latest[name].capture_substep
                ):
                    self.latest[name] = frame
            else:
                still_pending.append((name, frame))
        self.pending = still_pending
        return self.latest

    def images(self) -> dict[str, np.ndarray]:
        return {name: frame.image for name, frame in self.latest.items()}

    def ages(self, substep: int) -> dict[str, int]:
        return {name: substep - frame.capture_substep for name, frame in self.latest.items()}

    def intrinsics(self) -> dict[str, dict[str, float]]:
        return {
            name: camera_intrinsics(self.model, name, self.cfg.width, self.cfg.height)
            for name in self.names
        }

    def close(self) -> None:
        self.renderer.close()


class ViewRenderer:
    """A third-person render for videos and ``env.render()``."""

    def __init__(self, model: mujoco.MjModel, width: int, height: int, camera: str = "viewer"):
        self.model = model
        self.camera = camera
        self.renderer = mujoco.Renderer(model, height, width)

    def render(self, data: mujoco.MjData, camera: str | None = None) -> np.ndarray:
        mujoco.mj_forward(self.model, data)
        self.renderer.update_scene(data, camera=camera or self.camera)
        return self.renderer.render().copy()

    def close(self) -> None:
        self.renderer.close()
