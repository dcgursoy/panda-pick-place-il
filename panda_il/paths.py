"""Where things live on disk.

The repository is inside a OneDrive folder, so datasets, checkpoints, TensorBoard
runs and rendered videos never go there. They live under DATA_ROOT, resolved as:

1. an explicit path passed on the command line (``--data-root``),
2. the ``PANDA_IL_DATA_ROOT`` environment variable,
3. ``C:\\robodata\\panda-pick-place-il`` on Windows, ``~/robodata/panda-pick-place-il`` elsewhere.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets"
CONFIGS_DIR = REPO_ROOT / "configs"
DOCS_DIR = REPO_ROOT / "docs"

ENV_VAR = "PANDA_IL_DATA_ROOT"


def default_data_root() -> Path:
    if sys.platform == "win32":
        return Path(r"C:\robodata\panda-pick-place-il")
    return Path.home() / "robodata" / "panda-pick-place-il"


def resolve_data_root(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Apply the three-level precedence described in the module docstring."""
    if explicit is not None:
        return Path(explicit).expanduser()
    from_env = os.environ.get(ENV_VAR)
    if from_env:
        return Path(from_env).expanduser()
    return default_data_root()


@dataclass(frozen=True)
class Paths:
    """The standard sub-folders under DATA_ROOT."""

    root: Path

    @classmethod
    def resolve(cls, explicit: str | os.PathLike[str] | None = None) -> Paths:
        return cls(resolve_data_root(explicit))

    @property
    def datasets(self) -> Path:
        return self.root / "datasets"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def videos(self) -> Path:
        return self.root / "videos"

    @property
    def lerobot(self) -> Path:
        return self.root / "lerobot"

    @property
    def bench(self) -> Path:
        return self.root / "bench"

    @property
    def xpu_check_file(self) -> Path:
        """Record written by scripts/check_xpu.py; read by panda_il.device."""
        return self.root / "xpu_check.json"

    def make_dirs(self) -> None:
        for p in (self.root, self.datasets, self.runs, self.videos, self.lerobot, self.bench):
            p.mkdir(parents=True, exist_ok=True)
