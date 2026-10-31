"""Committed sets of initial conditions, drawn from named RNG streams.

The evaluation sets are JSON files under ``configs/`` so that every method is
evaluated on exactly the same cube poses and nobody trains on them. The
generator is a pure function of (base seed, stream id), so the files can be
regenerated and compared.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from panda_il.env.task import InitialCondition, ic_distance, sample_in_band, sample_in_square
from panda_il.paths import CONFIGS_DIR

BASE_SEED = 20261031
CUBE_REGION_CENTER = (0.50, -0.15)
CUBE_REGION_HALF = 0.08

# name -> (kind, inner half, outer half, count, stream id)
SET_SPECS: dict[str, tuple[str, float, float, int, int]] = {
    "eval_set_v1": ("square", 0.0, CUBE_REGION_HALF, 100, 1),
    "dev_set_v1": ("square", 0.0, CUBE_REGION_HALF, 20, 2),
    "robust_set_v1": ("band", CUBE_REGION_HALF, CUBE_REGION_HALF + 0.04, 100, 3),
    "robust_set_v1b": ("band", CUBE_REGION_HALF + 0.04, CUBE_REGION_HALF + 0.08, 100, 4),
}


@dataclass
class ICSet:
    name: str
    kind: str
    center: tuple[float, float]
    inner_half: float
    outer_half: float
    base_seed: int
    stream: int
    ics: list[InitialCondition]

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "schema_version": 1,
            "kind": self.kind,
            "center": list(self.center),
            "inner_half": self.inner_half,
            "outer_half": self.outer_half,
            "rng": {
                "base_seed": self.base_seed,
                "stream": self.stream,
                "generator": "numpy PCG64 default_rng([base_seed, stream])",
            },
            "count": len(self.ics),
            "ics": [ic.to_dict() for ic in self.ics],
        }

    @classmethod
    def from_json(cls, d: dict) -> ICSet:
        return cls(
            name=d["name"],
            kind=d["kind"],
            center=tuple(d["center"]),
            inner_half=float(d["inner_half"]),
            outer_half=float(d["outer_half"]),
            base_seed=int(d["rng"]["base_seed"]),
            stream=int(d["rng"]["stream"]),
            ics=[InitialCondition.from_dict(x) for x in d["ics"]],
        )


def stream_rng(base_seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng([base_seed, stream])


def generate_set(name: str, base_seed: int = BASE_SEED, center=CUBE_REGION_CENTER) -> ICSet:
    kind, inner, outer, count, stream = SET_SPECS[name]
    rng = stream_rng(base_seed, stream)
    if kind == "square":
        ics = [sample_in_square(rng, center, outer) for _ in range(count)]
    else:
        ics = [sample_in_band(rng, center, inner, outer) for _ in range(count)]
    return ICSet(name, kind, tuple(center), inner, outer, base_seed, stream, ics)


def set_path(name: str, directory: Path = CONFIGS_DIR) -> Path:
    return directory / f"{name}.json"


def save_set(ic_set: ICSet, directory: Path = CONFIGS_DIR) -> Path:
    path = set_path(ic_set.name, directory)
    path.write_text(json.dumps(ic_set.to_json(), indent=1), encoding="utf-8")
    return path


def load_set(name: str, directory: Path = CONFIGS_DIR) -> ICSet:
    return ICSet.from_json(json.loads(set_path(name, directory).read_text(encoding="utf-8")))


def nearest_in_sets(ic: InitialCondition, sets: list[ICSet]) -> tuple[float, float, str]:
    """Smallest (position, yaw) distance from ``ic`` to any member of the given sets."""
    best = (np.inf, np.inf, "")
    for ic_set in sets:
        for other in ic_set.ics:
            dpos, dyaw = ic_distance(ic, other)
            if dpos < best[0]:
                best = (dpos, dyaw, ic_set.name)
    return best
