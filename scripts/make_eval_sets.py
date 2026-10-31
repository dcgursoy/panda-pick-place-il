"""Generate the committed held-out initial-condition sets under configs/.

    uv run python scripts/make_eval_sets.py

Every set comes from its own RNG stream (see panda_il.env.sets). Each cube pose
is checked to be reachable (inside the TCP workspace box) and visible in the
front camera with a margin, and the sets are checked against each other.
"""

from __future__ import annotations

import sys

import mujoco
import numpy as np

from panda_il.config import EnvConfig
from panda_il.env import sets
from panda_il.env.cameras import project_point
from panda_il.env.scene import BIN_CENTER, BIN_INNER_HALF, CUBE_HALF, compile_scene_model, scene_ids
from panda_il.env.task import ic_distance

PIXEL_MARGIN = 4


def main() -> int:
    cfg = EnvConfig()
    model = compile_scene_model()
    data = mujoco.MjData(model)
    ids = scene_ids(model)
    mujoco.mj_resetDataKeyframe(model, data, ids.home_key)
    mujoco.mj_forward(model, data)
    width, height = cfg.camera.width, cfg.camera.height

    generated = {name: sets.generate_set(name) for name in sets.SET_SPECS}
    all_ok = True
    print(
        "| set | kind | n | x range | y range | min pixel margin (front) | min distance to other sets |"
    )
    print("|---|---|---:|---|---|---:|---:|")
    for name, ic_set in generated.items():
        xs = [ic.x for ic in ic_set.ics]
        ys = [ic.y for ic in ic_set.ics]
        margin = np.inf
        for ic in ic_set.ics:
            if not (
                cfg.workspace_min[0] <= ic.x <= cfg.workspace_max[0]
                and cfg.workspace_min[1] <= ic.y <= cfg.workspace_max[1]
            ):
                print(f"  {name}: {ic} outside the TCP workspace box")
                all_ok = False
            u, v, depth = project_point(
                model, data, "front", [ic.x, ic.y, CUBE_HALF], width, height
            )
            if not np.isfinite(u) or depth <= 0:
                margin = -np.inf
                continue
            margin = min(margin, u, v, width - u, height - v)
        if margin < PIXEL_MARGIN:
            print(
                f"  {name}: a cube pose is less than {PIXEL_MARGIN} px from the front image border"
            )
            all_ok = False
        others = [s for other, s in generated.items() if other != name]
        nearest = (
            min(sets.nearest_in_sets(ic, others)[0] for ic in ic_set.ics) if others else np.inf
        )
        print(
            f"| {name} | {ic_set.kind} | {len(ic_set.ics)} | {min(xs):.3f}..{max(xs):.3f} | {min(ys):.3f}..{max(ys):.3f} "
            f"| {margin:.1f} px | {nearest * 1000:.1f} mm |"
        )

    # The bin must be fully visible too.
    corners = [
        [BIN_CENTER[0] + sx * BIN_INNER_HALF, BIN_CENTER[1] + sy * BIN_INNER_HALF, 0.06]
        for sx in (-1, 1)
        for sy in (-1, 1)
    ]
    bin_margin = min(
        min(u, v, width - u, height - v)
        for u, v, _ in (project_point(model, data, "front", c, width, height) for c in corners)
    )
    print(f"\nbin rim corners: min front-image margin {bin_margin:.1f} px")
    if bin_margin < PIXEL_MARGIN:
        all_ok = False

    # Smallest distance between any two members of the same set (how distinct the ICs are).
    for name, ic_set in generated.items():
        best = min(
            ic_distance(a, b)[0] for i, a in enumerate(ic_set.ics) for b in ic_set.ics[i + 1 :]
        )
        print(f"{name}: closest pair {best * 1000:.1f} mm apart")

    if not all_ok:
        print("\nNOT writing sets: fix the camera or the regions first")
        return 1
    for ic_set in generated.values():
        path = sets.save_set(ic_set)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
