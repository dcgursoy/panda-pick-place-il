"""Align the raw streams onto the 20 Hz control timeline.

The timeline is the set of control ticks, substep 0, 25, 50, ... (one per
control step plus the final observation). For each tick:

- proprio (100 Hz) is linearly interpolated (quaternions by normalised lerp);
- the cube pose comes from the privileged 20 Hz stream (exact at ticks, most
  recent sample if one is missing);
- each camera contributes the most recent frame that had *arrived* by the tick;
- the action is the most recent command logged at or before the tick (zero-order
  hold), which is what the environment executed.

Every alignment reports its error per tick, and the maximum per stream is what
the episode records as its timing quality.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SyncResult:
    arrays: dict[str, np.ndarray]
    alignment: dict[str, dict[str, float]]  # per stream: max/mean error in substeps and seconds


def latest_index_at_or_before(sample_substeps: np.ndarray, tick_substeps: np.ndarray) -> np.ndarray:
    """Index of the last sample with substep <= tick, or -1 when none exists."""
    return np.searchsorted(sample_substeps, tick_substeps, side="right") - 1


def interpolate_linear(
    sample_substeps: np.ndarray, values: np.ndarray, tick_substeps: np.ndarray
) -> np.ndarray:
    """Per-column linear interpolation; outside the sampled range the edge value is held."""
    values = np.asarray(values, dtype=np.float64)
    out = np.empty((len(tick_substeps), values.shape[1]), dtype=np.float64)
    for column in range(values.shape[1]):
        out[:, column] = np.interp(tick_substeps, sample_substeps, values[:, column])
    return out


def interpolate_quaternion(
    sample_substeps: np.ndarray, quats: np.ndarray, tick_substeps: np.ndarray
) -> np.ndarray:
    """Normalised lerp between neighbouring samples.

    Consecutive samples are made to agree in sign before interpolating (q and
    -q are the same rotation), then each result takes the sign of the raw
    sample at or before its tick: MuJoCo's quaternion is not sign-continuous,
    and the policy sees the raw sign at evaluation time, so the synced value
    at a tick must equal the logged observation exactly (the validator checks
    this)."""
    raw = np.asarray(quats, dtype=np.float64)
    quats = raw.copy()
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] *= -1
    out = interpolate_linear(sample_substeps, quats, tick_substeps)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    out = out / np.where(norms > 0, norms, 1.0)
    ref = raw[np.clip(latest_index_at_or_before(sample_substeps, tick_substeps), 0, len(raw) - 1)]
    flip = np.einsum("ij,ij->i", out, ref) < 0
    out[flip] *= -1
    return out


def nearest_sample_distance(sample_substeps: np.ndarray, tick_substeps: np.ndarray) -> np.ndarray:
    """Distance (substeps) from each tick to the closest sample; the interpolation error proxy."""
    idx = np.searchsorted(sample_substeps, tick_substeps)
    left = np.clip(idx - 1, 0, len(sample_substeps) - 1)
    right = np.clip(idx, 0, len(sample_substeps) - 1)
    return np.minimum(
        np.abs(tick_substeps - sample_substeps[left]),
        np.abs(sample_substeps[right] - tick_substeps),
    )


def _summary(errors_substeps: np.ndarray, physics_dt: float) -> dict[str, float]:
    errors_substeps = np.asarray(errors_substeps, dtype=np.float64)
    if errors_substeps.size == 0:
        return {"max_substeps": 0.0, "mean_substeps": 0.0, "max_s": 0.0, "mean_s": 0.0}
    return {
        "max_substeps": float(errors_substeps.max()),
        "mean_substeps": float(errors_substeps.mean()),
        "max_s": float(errors_substeps.max() * physics_dt),
        "mean_s": float(errors_substeps.mean() * physics_dt),
    }


def sync_episode(
    raw: dict[str, dict[str, np.ndarray]],
    n_steps: int,
    substeps_per_step: int,
    physics_dt: float,
    camera_names: tuple[str, ...] = (),
) -> SyncResult:
    """Build the synchronised arrays for an episode of ``n_steps`` control steps.

    Observation-like arrays have ``n_steps + 1`` rows (ticks 0..n_steps), action
    arrays have ``n_steps`` rows (the action taken at tick k).
    """
    ticks = np.arange(n_steps + 1, dtype=np.int64) * substeps_per_step
    action_ticks = ticks[:-1]
    arrays: dict[str, np.ndarray] = {"substep": ticks, "t": ticks.astype(np.float64) * physics_dt}
    alignment: dict[str, dict[str, float]] = {}

    # --- proprio: interpolated
    proprio = raw["proprio"]
    p_sub = np.asarray(proprio["substep"], dtype=np.int64)
    parts = [
        interpolate_linear(p_sub, proprio["qpos"], ticks),
        interpolate_linear(p_sub, proprio["qvel"], ticks),
        interpolate_linear(p_sub, np.asarray(proprio["gripper_width"]).reshape(-1, 1), ticks),
        interpolate_linear(p_sub, proprio["tcp_pos"], ticks),
        interpolate_quaternion(p_sub, proprio["tcp_quat"], ticks),
    ]
    arrays["proprio"] = np.concatenate(parts, axis=1)
    alignment["proprio"] = _summary(nearest_sample_distance(p_sub, ticks), physics_dt)

    # --- privileged 20 Hz stream: cube pose at ticks (most recent sample), executed actions
    priv = raw["priv"]
    v_sub = np.asarray(priv["substep"], dtype=np.int64)
    idx = latest_index_at_or_before(v_sub, ticks)
    if np.any(idx < 0):
        raise ValueError("no privileged sample at or before the first tick")
    cube = np.concatenate(
        [np.asarray(priv["cube_pos"])[idx], np.asarray(priv["cube_quat"])[idx]], axis=1
    )
    arrays["state"] = np.concatenate([arrays["proprio"], cube], axis=1)
    arrays["priv_index"] = idx
    alignment["priv"] = _summary(ticks - v_sub[idx], physics_dt)
    # the action executed during step k is reported by the priv sample at tick k+1
    executed_idx = latest_index_at_or_before(v_sub, ticks[1:])
    arrays["action_executed"] = np.asarray(priv["action_executed"])[executed_idx]

    # --- cameras: most recent arrived frame
    for name in camera_names:
        cam = raw[f"cam_{name}"]
        arrival = np.asarray(cam["arrival_substep"], dtype=np.int64)
        capture = np.asarray(cam["capture_substep"], dtype=np.int64)
        order = np.argsort(arrival, kind="stable")
        frame_idx = order[latest_index_at_or_before(arrival[order], ticks)]
        if np.any(latest_index_at_or_before(arrival[order], ticks) < 0):
            raise ValueError(f"camera {name}: no frame had arrived at the first tick")
        arrays[f"image_index_{name}"] = frame_idx.astype(np.int64)
        age = ticks - capture[frame_idx]
        arrays[f"frame_age_{name}"] = age
        alignment[f"cam_{name}"] = _summary(age, physics_dt)

    # --- action: zero-order hold of the latest command at or before each step's tick
    action = raw["action"]
    a_sub = np.asarray(action["substep"], dtype=np.int64)
    a_idx = latest_index_at_or_before(a_sub, action_ticks)
    if np.any(a_idx < 0):
        raise ValueError("a control step has no command logged at or before it")
    arrays["action"] = np.asarray(action["action"], dtype=np.float64)[a_idx]
    arrays["action_index"] = a_idx
    alignment["action"] = _summary(action_ticks - a_sub[a_idx], physics_dt)

    return SyncResult(arrays=arrays, alignment=alignment)
