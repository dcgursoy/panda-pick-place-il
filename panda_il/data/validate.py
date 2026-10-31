"""Validate a dataset: schema, timestamps, dropped frames, NaNs, bounds, consistency.

    uv run python -m panda_il.data.validate --dataset scripted_v1

Checks per episode:
  schema      version, required attributes, required streams and fields, dtypes and shapes
  timing      substeps and simulation time strictly increasing per stream, wall time monotonic,
              gaps larger than 1.5 nominal periods reported as dropped frames
  values      no NaN or Inf, actions inside [-1, 1], images uint8 of the configured size
  source      the episode's source and operator match the dataset's
  held-out    warns if the initial condition is within 5 mm and 5 degrees of a committed held-out pose
  sync        re-runs the sync module from the raw streams and requires the stored synced arrays,
              the logged observations and the executed actions to agree
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np

from panda_il.data.schema import SCHEMA_VERSION, Dataset, DatasetMetadata, EpisodeRecord
from panda_il.data.sync import sync_episode
from panda_il.env import sets
from panda_il.env.task import InitialCondition
from panda_il.paths import CONFIGS_DIR, Paths

REQUIRED_ATTRS = (
    "source",
    "operator",
    "input_device",
    "seed",
    "ic",
    "length",
    "success",
    "t_wall_start",
    "env_version",
    "git_commit",
    "alignment",
    "time_limit_steps",
)
HELD_OUT_POS_M = 0.005
HELD_OUT_YAW_RAD = np.deg2rad(5.0)
GAP_FACTOR = 1.5


@dataclass
class Issue:
    level: str  # "error" or "warning"
    episode: int | None
    where: str
    message: str

    def __str__(self) -> str:
        ep = f"ep {self.episode}" if self.episode is not None else "dataset"
        return f"[{self.level}] {ep} / {self.where}: {self.message}"


def _finite(array: np.ndarray) -> bool:
    if array.dtype.kind in "fc":
        return bool(np.isfinite(array).all())
    return True


def validate_episode(
    record: EpisodeRecord, metadata: DatasetMetadata, held_out: list[sets.ICSet] | None = None
) -> list[Issue]:
    issues: list[Issue] = []
    attrs = record.attrs
    ep = attrs.get("episode_id")

    def error(where: str, message: str) -> None:
        issues.append(Issue("error", ep, where, message))

    def warning(where: str, message: str) -> None:
        issues.append(Issue("warning", ep, where, message))

    # ---- schema
    if attrs.get("schema_version") != SCHEMA_VERSION:
        error("attrs", f"schema_version {attrs.get('schema_version')} != {SCHEMA_VERSION}")
    for name in REQUIRED_ATTRS:
        if name not in attrs:
            error("attrs", f"missing attribute {name!r}")
    for stream_name, spec in metadata.streams.items():
        if stream_name not in record.raw:
            error("raw", f"missing stream {stream_name!r}")
            continue
        fields = record.raw[stream_name]
        for field_name in ("t_sim", "t_wall", "substep", *spec["fields"]):
            if field_name not in fields:
                error(f"raw/{stream_name}", f"missing field {field_name!r}")
    for name in ("sim_state", "proprio", "state"):
        if name not in record.init:
            error("init", f"missing {name!r}")
    if "sim_state" in record.init and record.init["sim_state"].shape != (
        metadata.mujoco_state_size,
    ):
        error(
            "init",
            f"sim_state has shape {record.init['sim_state'].shape}, expected ({metadata.mujoco_state_size},)",
        )
    if issues:
        return issues  # the rest assumes the structure is there

    n_steps = int(attrs["length"])
    substeps = metadata.substeps

    # ---- timing per stream
    for stream_name, fields in record.raw.items():
        sub = np.asarray(fields["substep"], dtype=np.int64)
        t_sim = np.asarray(fields["t_sim"], dtype=np.float64)
        t_wall = np.asarray(fields["t_wall"], dtype=np.float64)
        where = f"raw/{stream_name}"
        if len(sub) == 0:
            error(where, "empty stream")
            continue
        period = int(
            record.raw_attrs.get(stream_name, {}).get(
                "nominal_period_substeps", metadata.streams[stream_name]["nominal_period_substeps"]
            )
        )
        if period > 0:
            if np.any(np.diff(sub) <= 0):
                error(where, "substeps are not strictly increasing")
            if np.any(np.diff(t_sim) <= 0):
                error(where, "simulation time is not strictly increasing")
            gaps = np.diff(sub)
            dropped = int(np.sum(gaps > GAP_FACTOR * period))
            if dropped:
                error(
                    where,
                    f"{dropped} gap(s) larger than {GAP_FACTOR} x {period} substeps (dropped frames)",
                )
            if sub[0] != 0:
                error(where, f"first sample at substep {sub[0]}, expected 0")
            expected = n_steps * substeps // period + 1
            if len(sub) != expected:
                warning(where, f"{len(sub)} samples, expected {expected} for {n_steps} steps")
        else:  # command stream: ordered by wall time, substeps non-decreasing
            if np.any(np.diff(sub) < 0):
                error(where, "command substeps decrease")
            if len(sub) < n_steps:
                error(where, f"{len(sub)} commands for {n_steps} steps")
        if np.any(np.diff(t_wall) < 0):
            error(where, "wall time decreases")
        if not np.all(np.isfinite(t_sim)) or not np.all(np.isfinite(t_wall)):
            error(where, "non-finite timestamps")

    # ---- values
    for stream_name, fields in record.raw.items():
        for field_name, array in fields.items():
            array = np.asarray(array)
            if not _finite(array):
                error(f"raw/{stream_name}/{field_name}", "NaN or Inf")
    for name, array in record.synced.items():
        if not _finite(np.asarray(array)):
            error(f"synced/{name}", "NaN or Inf")
    actions = np.asarray(record.raw["action"]["action"])
    if actions.size and (actions.min() < -1.0 or actions.max() > 1.0):
        error("raw/action", "action outside [-1, 1]")
    for cam, spec in metadata.cameras.items():
        images = np.asarray(record.raw[f"cam_{cam}"]["image"])
        expected_shape = (spec["height"], spec["width"], 3)
        if images.dtype != np.uint8 or images.shape[1:] != expected_shape:
            error(
                f"raw/cam_{cam}",
                f"images are {images.dtype} {images.shape[1:]}, expected uint8 {expected_shape}",
            )
        elif images.size and images.std() == 0:
            warning(f"raw/cam_{cam}", "every frame is a constant image")

    # ---- source and operator
    if attrs["source"] != metadata.source:
        error(
            "attrs",
            f"episode source {attrs['source']!r} differs from dataset source {metadata.source!r}",
        )
    if not attrs.get("operator"):
        error("attrs", "operator is empty")

    # ---- held-out proximity
    if held_out:
        ic = InitialCondition.from_dict(attrs["ic"])
        dpos, dyaw, which = sets.nearest_in_sets(ic, held_out)
        if dpos < HELD_OUT_POS_M and dyaw < HELD_OUT_YAW_RAD:
            warning(
                "attrs/ic",
                f"initial condition within {dpos * 1000:.1f} mm / {np.rad2deg(dyaw):.1f} deg of a pose in {which}",
            )

    # ---- sync consistency
    try:
        result = sync_episode(
            record.raw, n_steps, substeps, 1.0 / metadata.physics_hz, tuple(metadata.cameras)
        )
    except ValueError as exc:
        error("sync", f"sync failed: {exc}")
        return issues
    for name, array in result.arrays.items():
        stored = record.synced.get(name)
        if stored is None:
            error("synced", f"missing array {name!r}")
        elif not np.array_equal(np.asarray(stored), array):
            error(
                f"synced/{name}", "stored synced array differs from a fresh sync of the raw streams"
            )
    obs = record.raw["obs"]
    if len(obs["substep"]) == n_steps + 1:
        if not np.allclose(
            result.arrays["proprio"], np.asarray(obs["proprio"], dtype=np.float64), atol=1e-5
        ):
            error("sync/proprio", "synced proprio differs from the logged observations")
        if not np.allclose(
            result.arrays["state"], np.asarray(obs["state"], dtype=np.float64), atol=1e-5
        ):
            error("sync/state", "synced state differs from the logged observations")
        for cam in metadata.cameras:
            logged = np.asarray(obs[f"frame_substep_{cam}"])
            capture = np.asarray(record.raw[f"cam_{cam}"]["capture_substep"])[
                result.arrays[f"image_index_{cam}"]
            ]
            if not np.array_equal(logged, capture):
                error(
                    f"sync/cam_{cam}",
                    "synced frame choice differs from the frame the environment delivered",
                )
    if not np.array_equal(result.arrays["action"], result.arrays["action_executed"]):
        error(
            "sync/action",
            "zero-order-hold actions differ from the actions the environment executed",
        )
    if len(result.arrays["action"]) != n_steps:
        error("synced/action", f"{len(result.arrays['action'])} actions for length {n_steps}")
    priv = record.raw["priv"]
    final_success = bool(priv["success_streak"][-1] >= metadata.env_config["success_steps"])
    if bool(attrs["success"]) != final_success:
        error("attrs/success", "success flag disagrees with the final success streak")
    return issues


@dataclass
class Report:
    dataset: str
    episodes: int
    errors: int
    warnings: int
    issues: list[Issue]
    successes: int
    lengths: list[int]
    max_alignment_s: dict[str, float]

    @property
    def ok(self) -> bool:
        return self.errors == 0

    def summary(self) -> str:
        lines = [
            f"dataset {self.dataset}: {self.episodes} episodes, {self.successes} successful, "
            f"lengths {min(self.lengths) if self.lengths else 0}-{max(self.lengths) if self.lengths else 0} steps "
            f"(mean {np.mean(self.lengths):.1f})"
            if self.lengths
            else f"dataset {self.dataset}: no episodes",
            "max alignment error per stream: "
            + ", ".join(f"{k} {v * 1000:.1f} ms" for k, v in self.max_alignment_s.items()),
            f"{self.errors} error(s), {self.warnings} warning(s) -> {'VALID' if self.ok else 'INVALID'}",
        ]
        return "\n".join(lines)


def validate_dataset(
    dataset: Dataset, check_held_out: bool = True, episodes: list[int] | None = None
) -> Report:
    held_out = None
    if check_held_out:
        held_out = []
        for name in sets.SET_SPECS:
            path = sets.set_path(name, CONFIGS_DIR)
            if path.exists():
                held_out.append(sets.load_set(name, CONFIGS_DIR))
    issues: list[Issue] = []
    if dataset.metadata.schema_version != SCHEMA_VERSION:
        issues.append(
            Issue("error", None, "metadata", f"schema_version {dataset.metadata.schema_version}")
        )
    ids = episodes if episodes is not None else dataset.episode_ids()
    lengths, successes = [], 0
    max_align: dict[str, float] = {}
    for episode_id in ids:
        record = dataset.read_episode(episode_id)
        issues.extend(validate_episode(record, dataset.metadata, held_out))
        lengths.append(int(record.attrs.get("length", 0)))
        successes += int(bool(record.attrs.get("success")))
        for stream, value in record.attrs.get("max_alignment_error_s", {}).items():
            max_align[stream] = max(max_align.get(stream, 0.0), float(value))
    counted = dataset.metadata.counts.get("episodes")
    if counted is not None and episodes is None and counted != len(ids):
        issues.append(
            Issue(
                "warning",
                None,
                "metadata",
                f"counts.episodes = {counted} but {len(ids)} episode groups exist",
            )
        )
    errors = sum(1 for i in issues if i.level == "error")
    warnings = sum(1 for i in issues if i.level == "warning")
    return Report(dataset.name, len(ids), errors, warnings, issues, successes, lengths, max_align)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--no-held-out", action="store_true")
    parser.add_argument("--max-issues", type=int, default=50)
    args = parser.parse_args()
    dataset = Dataset.open(Paths.resolve(args.data_root), args.dataset)
    report = validate_dataset(dataset, check_held_out=not args.no_held_out)
    for issue in report.issues[: args.max_issues]:
        print(issue)
    if len(report.issues) > args.max_issues:
        print(f"... {len(report.issues) - args.max_issues} more")
    print(report.summary())
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
