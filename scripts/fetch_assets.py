"""Vendor the MuJoCo Menagerie Franka Panda model into assets/franka_emika_panda.

    uv run python scripts/fetch_assets.py

Uses the official ``mujoco-menagerie`` package (version pinned in pyproject.toml),
copies every file that the robot's scene depends on plus its LICENSE, README and
CHANGELOG, and writes PROVENANCE.md so the exact source is recorded. The repo is
self-contained afterwards; nothing is downloaded at runtime.
"""

from __future__ import annotations

import importlib.metadata
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import mujoco_menagerie as mm

from panda_il.paths import ASSETS_DIR

ROBOT = "franka_emika_panda"
EXTRA_FILES = ("LICENSE", "README.md", "CHANGELOG.md", "panda_nohand.xml", "hand.xml")


def main() -> int:
    robot = mm.get(ROBOT)
    source_dir = Path(robot.path())
    destination = ASSETS_DIR / ROBOT
    destination.mkdir(parents=True, exist_ok=True)

    wanted: set[Path] = set()
    for entry in robot.files("scene"):
        path = Path(entry)
        wanted.add(path.relative_to(source_dir) if path.is_absolute() else path)
    for name in EXTRA_FILES:
        if (source_dir / name).exists():
            wanted.add(Path(name))

    copied: list[str] = []
    for rel in sorted(wanted):
        src = source_dir / rel
        dst = destination / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(rel.as_posix())

    version = importlib.metadata.version("mujoco-menagerie")
    oid = getattr(robot, "oid", None)
    license_name = getattr(robot, "license", "see LICENSE")
    provenance = destination / "PROVENANCE.md"
    provenance.write_text(
        "\n".join(
            [
                f"# Provenance of assets/{ROBOT}",
                "",
                "Copied verbatim from the MuJoCo Menagerie Franka Emika Panda model, obtained through the",
                f"official `mujoco-menagerie` PyPI package, version `{version}` (object id `{oid}`), on",
                f"{datetime.now(UTC).date().isoformat()} by `scripts/fetch_assets.py`.",
                f"License: {license_name} (see LICENSE in this folder).",
                "",
                "Our scene (`panda_il/env/scene.xml`) attaches `panda.xml` through MjSpec and adds its own",
                "table, cube, bin, cameras and a `tcp` site; the files here are not modified.",
                "",
                "Files:",
                *[f"- {name}" for name in copied],
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"copied {len(copied)} files from {source_dir} to {destination}")
    print(f"mujoco-menagerie {version}, oid {oid}; wrote {provenance}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
