# panda-pick-place-il

[![ci](https://github.com/dcgursoy/panda-pick-place-il/actions/workflows/ci.yml/badge.svg)](https://github.com/dcgursoy/panda-pick-place-il/actions/workflows/ci.yml)

An imitation learning study on a simulated Franka Panda: a cube is picked up and
placed in a bin, demonstrated by a human through keyboard or gamepad teleop in
MuJoCo, and learned by behavior cloning (MLP and ResNet18) and by Diffusion
Policy (Chi et al. 2023), both written as plain PyTorch, with a Stable Baselines3
SAC agent as the reinforcement learning baseline. Every method is trained with
several seeds and evaluated on the same 100 held-out initial conditions, with
bootstrap confidence intervals, a success-versus-demos curve, a human-versus-
scripted demo ablation and a robustness test. The dataset tooling (multi-rate
logger, Zarr format, timestamp sync, validator, replay, LeRobot export) is our
own, and LeRobot's Diffusion Policy trained on the exported data serves as a
cross-check.

![A scripted pick-and-place in the environment: third-person view, front camera, wrist camera](docs/gifs/m1_env_demo.gif)

> Status: **milestone 1 of 8 (environment).** [DESIGN.md](DESIGN.md) holds the
> design decisions, the alternatives considered and the milestone plan;
> [RESULTS.md](RESULTS.md) will hold the raw data behind every number reported
> here, with the command that produced it; [docs/milestone-logs/](docs/milestone-logs/)
> is the verification log of each milestone.

## Hardware this was built on

Windows 11, Intel Core Ultra 7 258V with its integrated Intel Arc 140V GPU (no
CUDA). Training runs on the iGPU through PyTorch's XPU backend, which is 15 to
40 times faster than the CPU for the ResNet18 / UNet models here (measured in
[docs/milestone-logs/m0.md](docs/milestone-logs/m0.md)). CPU-only runs work but
are impractical for the image-based policies.

## Setup

1. Intel Arc & Iris Xe graphics driver **32.0.101.9150 or newer**. Intel
   requires at least 32.0.101.9033 for PyTorch 2.15; on an older driver the XPU
   backend returned silently wrong results on this machine, so the code refuses
   to use the XPU until `scripts/check_xpu.py` has passed.
2. [uv](https://docs.astral.sh/uv/). Everything else is installed by it.

```bash
uv sync                                  # main environment (Python 3.12, torch 2.15 XPU); CI uses --no-default-groups --group dev --group cpu
uv run python scripts/check_xpu.py       # CPU-vs-XPU equality checks; writes DATA_ROOT/xpu_check.json
uv run python scripts/fetch_assets.py    # vendors the Menagerie Panda into assets/ (already committed)
uv run python scripts/env_demo.py --gif  # scripted pick-and-place video of the environment
uv run pytest                            # unit tests
```

Heavy artifacts (datasets, checkpoints, TensorBoard runs, videos) live outside
the repository under `DATA_ROOT`, which defaults to `C:\robodata\panda-pick-place-il`
on Windows (`~/robodata/panda-pick-place-il` elsewhere) and can be set with the
`PANDA_IL_DATA_ROOT` environment variable or `--data-root`.

The LeRobot cross-check uses a second, isolated environment because LeRobot
0.6.1 pins an older torch:

```bash
cd tools/lerobot_env && uv sync
```

## Layout

```
panda_il/          the package: env, teleop, data, policies, train, eval
scripts/           one-off commands: asset vendoring, XPU self-check, device benchmark
tests/             pytest suite
tools/lerobot_env/ the isolated LeRobot environment and its cross-check scripts
assets/            vendored MuJoCo Menagerie Panda (with LICENSE and PROVENANCE.md)
configs/           committed held-out initial-condition sets
docs/              milestone logs, figures and GIFs shown in this README
```

## Results

None yet; they are produced in milestone 7 and every number will be traced in
[RESULTS.md](RESULTS.md).
