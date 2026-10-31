"""Dataset layout and the Zarr schema, version 1.

A dataset is a directory under ``DATA_ROOT/datasets/<name>/``:

    metadata.json      one file per dataset (env version, rates, cameras, source, operator, git commit, ...)
    episodes.zarr/     one Zarr group per episode, ``ep_000000``, ``ep_000001``, ...
    splits.json        train / validation episode ids (milestone 4)
    stats.json         normalisation statistics (milestone 4)

Each episode group holds the raw streams exactly as logged (each with its own
timestamps and rate), the initial state, the synchronised 20 Hz arrays, and the
episode attributes (success, source, operator, seed, initial condition, timing
errors, git commit). Images are chunked 32 frames at a time and compressed with
Blosc/zstd. All times are seconds of simulation time unless named ``t_wall``.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from panda_il.paths import REPO_ROOT, Paths

SCHEMA_VERSION = 1
EPISODE_PREFIX = "ep_"
IMAGE_CHUNK_FRAMES = 32

TIME_FIELDS = ("t_sim", "t_wall", "substep")


@dataclass
class EpisodeRecord:
    """One episode in memory: what the recorder produces and what the store returns."""

    attrs: dict[str, Any]
    init: dict[str, np.ndarray]
    raw: dict[str, dict[str, np.ndarray]]  # stream name -> field -> array
    raw_attrs: dict[str, dict[str, Any]]  # stream name -> {"nominal_period_substeps": int, ...}
    synced: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return int(self.attrs["length"])


@dataclass
class DatasetMetadata:
    """The per-dataset metadata file."""

    name: str
    source: str  # "human" or "scripted"
    operator: str
    input_device: str
    env_version: str
    env_config: dict[str, Any]
    control_hz: float
    physics_hz: float
    substeps: int
    cameras: dict[str, dict[str, Any]]
    action: dict[str, Any]
    observation: dict[str, Any]
    streams: dict[str, dict[str, Any]]
    mujoco_state_spec: str
    mujoco_state_size: int
    versions: dict[str, str]
    git_commit: str
    git_dirty: bool
    created_at: str
    schema_version: int = SCHEMA_VERSION
    counts: dict[str, int] = field(
        default_factory=lambda: {"episodes": 0, "successes": 0, "discarded": 0}
    )
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1, default=_json_default)

    @classmethod
    def from_json(cls, text: str) -> DatasetMetadata:
        return cls(**json.loads(text))


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value)}")


def git_state(repo: Path = REPO_ROOT) -> tuple[str, bool]:
    """Current commit hash and whether the working tree has uncommitted changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        ).stdout.strip()
        return commit, bool(status)
    except (OSError, subprocess.SubprocessError):
        return "unknown", True


def library_versions() -> dict[str, str]:
    import importlib.metadata as md

    out = {}
    for name in (
        "mujoco",
        "mink",
        "mujoco-menagerie",
        "torch",
        "numpy",
        "zarr",
        "numcodecs",
        "gymnasium",
    ):
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = "missing"
    from panda_il import __version__

    out["panda_il"] = __version__
    return out


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Dataset:
    """A dataset directory with its metadata file and Zarr episode store."""

    def __init__(self, directory: Path, mode: str = "r"):
        self.directory = Path(directory)
        self.mode = mode
        if mode == "r" and not self.metadata_path.exists():
            raise FileNotFoundError(f"no dataset at {self.directory}")
        self.metadata = DatasetMetadata.from_json(self.metadata_path.read_text(encoding="utf-8"))
        self.root = zarr.open_group(str(self.store_path), mode="r" if mode == "r" else "a")

    # ------------------------------------------------------------ construction
    @classmethod
    def create(cls, directory: Path, metadata: DatasetMetadata) -> Dataset:
        directory = Path(directory)
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"{directory} already exists and is not empty")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "metadata.json").write_text(metadata.to_json(), encoding="utf-8")
        root = zarr.open_group(str(directory / "episodes.zarr"), mode="w")
        root.attrs["schema_version"] = SCHEMA_VERSION
        root.attrs["name"] = metadata.name
        return cls(directory, mode="a")

    @classmethod
    def open(cls, paths: Paths, name: str, mode: str = "r") -> Dataset:
        return cls(paths.datasets / name, mode=mode)

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def metadata_path(self) -> Path:
        return self.directory / "metadata.json"

    @property
    def store_path(self) -> Path:
        return self.directory / "episodes.zarr"

    def save_metadata(self) -> None:
        self.metadata_path.write_text(self.metadata.to_json(), encoding="utf-8")

    # ---------------------------------------------------------------- episodes
    def episode_ids(self) -> list[int]:
        ids = []
        for key in self.root.group_keys():
            if key.startswith(EPISODE_PREFIX):
                ids.append(int(key[len(EPISODE_PREFIX) :]))
        return sorted(ids)

    @staticmethod
    def episode_key(episode_id: int) -> str:
        return f"{EPISODE_PREFIX}{episode_id:06d}"

    def write_episode(self, record: EpisodeRecord, episode_id: int | None = None) -> int:
        if self.mode == "r":
            raise PermissionError("dataset opened read-only")
        if episode_id is None:
            ids = self.episode_ids()
            episode_id = (ids[-1] + 1) if ids else 0
        group = self.root.create_group(self.episode_key(episode_id))
        record.attrs["episode_id"] = episode_id
        record.attrs["schema_version"] = SCHEMA_VERSION
        attrs = dict(record.attrs)
        group.attrs.update(_jsonable(attrs))

        init = group.create_group("init")
        for name, array in record.init.items():
            _write_array(init, name, np.asarray(array))

        raw = group.create_group("raw")
        for stream_name, fields in record.raw.items():
            stream = raw.create_group(stream_name)
            stream.attrs.update(_jsonable(record.raw_attrs.get(stream_name, {})))
            for field_name, array in fields.items():
                _write_array(stream, field_name, np.asarray(array))

        synced = group.create_group("synced")
        for name, array in record.synced.items():
            _write_array(synced, name, np.asarray(array))

        self.metadata.counts["episodes"] = len(self.episode_ids())
        self.metadata.counts["successes"] = self.metadata.counts.get("successes", 0) + int(
            bool(attrs.get("success"))
        )
        self.save_metadata()
        return episode_id

    def read_episode(self, episode_id: int, load_images: bool = True) -> EpisodeRecord:
        group = self.root[self.episode_key(episode_id)]
        attrs = dict(group.attrs)
        init = {name: group["init"][name][...] for name in group["init"].array_keys()}
        raw: dict[str, dict[str, np.ndarray]] = {}
        raw_attrs: dict[str, dict[str, Any]] = {}
        for stream_name in group["raw"].group_keys():
            stream = group["raw"][stream_name]
            raw_attrs[stream_name] = dict(stream.attrs)
            fields = {}
            for field_name in stream.array_keys():
                if field_name == "image" and not load_images:
                    continue
                fields[field_name] = stream[field_name][...]
            raw[stream_name] = fields
        synced = {name: group["synced"][name][...] for name in group["synced"].array_keys()}
        return EpisodeRecord(attrs=attrs, init=init, raw=raw, raw_attrs=raw_attrs, synced=synced)

    def read_attrs(self, episode_id: int) -> dict[str, Any]:
        return dict(self.root[self.episode_key(episode_id)].attrs)

    def read_images(
        self, episode_id: int, camera: str, indices: np.ndarray | None = None
    ) -> np.ndarray:
        array = self.root[self.episode_key(episode_id)]["raw"][f"cam_{camera}"]["image"]
        return array[...] if indices is None else array[np.asarray(indices)]

    def read_synced(self, episode_id: int, name: str) -> np.ndarray:
        return self.root[self.episode_key(episode_id)]["synced"][name][...]


def _jsonable(d: dict) -> dict:
    return json.loads(json.dumps(d, default=_json_default))


def _write_array(group: zarr.Group, name: str, array: np.ndarray) -> None:
    if array.ndim == 4 and array.dtype == np.uint8:  # (N, H, W, 3) images
        chunks = (min(IMAGE_CHUNK_FRAMES, max(array.shape[0], 1)), *array.shape[1:])
        compressors = [zarr.codecs.BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")]
    else:
        chunks = tuple(max(s, 1) for s in array.shape) if array.ndim else ()
        compressors = [zarr.codecs.BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")]
    if array.dtype == object:
        raise TypeError(f"array {name!r} has object dtype; store strings in attrs")
    out = group.create_array(
        name=name,
        shape=array.shape,
        chunks=chunks or (1,),
        dtype=array.dtype,
        compressors=compressors,
    )
    out[...] = array
