"""Figures for the milestone logs and RESULTS.md (matplotlib, PNG into docs/figures).

uv run python -m panda_il.eval.plots sac --runs sac/shaped/seed0 sac/shaped/seed1 sac/shaped/seed2 --out docs/figures/m6_sac_curves.png
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from panda_il.paths import Paths  # noqa: E402


def read_progress(run_dir: Path) -> dict[str, list[float]]:
    """Columns of SB3's progress.csv as float lists (missing cells skipped per column)."""
    columns: dict[str, list[tuple[float, float]]] = {}
    with (run_dir / "progress.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            step = row.get("time/total_timesteps")
            if not step:
                continue
            for key, value in row.items():
                if value not in ("", None) and key != "time/total_timesteps":
                    columns.setdefault(key, []).append((float(step), float(value)))
    out: dict[str, list[float]] = {}
    for key, pairs in columns.items():
        out[key + ":steps"] = [p[0] for p in pairs]
        out[key] = [p[1] for p in pairs]
    return out


def plot_sac_curves(
    run_dirs: list[Path], out: Path, title: str = "SAC on the shaped reward"
) -> Path:
    """Training reward, training success rate, final reach distance and dev-set success
    against env steps, one line per seed."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    panels = [
        ("rollout/ep_rew_mean", "episode return (train)"),
        ("rollout/success_rate", "success rate, last 100 train episodes"),
        ("rollout/final_reach_m", "final TCP-to-cube distance (m), train"),
    ]
    for run_dir in run_dirs:
        label = run_dir.name
        data = read_progress(run_dir)
        for ax, (key, _) in zip(axes[:3], panels, strict=True):
            if key in data:
                ax.plot(data[key + ":steps"], data[key], label=label, linewidth=1.2)
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            curve = json.loads(summary_path.read_text(encoding="utf-8")).get("dev_curve", [])
            if curve:
                axes[3].plot(
                    [c["env_steps"] for c in curve],
                    [c["success_rate"] for c in curve],
                    marker="o",
                    label=label,
                    linewidth=1.2,
                )
    for ax, (_, ylabel) in zip(axes[:3], panels, strict=True):
        ax.set_ylabel(ylabel)
    axes[3].set_ylabel("dev-set success (20 episodes)")
    axes[3].set_ylim(-0.02, 1.02)
    for ax in axes:
        ax.set_xlabel("env steps")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="figure", required=True)
    sac = sub.add_parser("sac", help="SAC learning curves")
    sac.add_argument("--runs", nargs="+", required=True, help="run dirs relative to DATA_ROOT/runs")
    sac.add_argument("--out", required=True)
    sac.add_argument("--title", default="SAC on the shaped reward")
    sac.add_argument("--data-root", default=None)
    args = parser.parse_args()
    paths = Paths.resolve(args.data_root)
    if args.figure == "sac":
        out = plot_sac_curves([paths.runs / r for r in args.runs], Path(args.out), args.title)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
