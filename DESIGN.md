# Design notes

Every design decision of panda-pick-place-il, the alternatives considered, and
the milestone plan with its exit criteria. Sections are revised as milestones
land; each milestone's verification log lives in `docs/milestone-logs/`.

## 0. Goals and non-goals

Goal: a portfolio-quality, reproducible study comparing behavior cloning,
Diffusion Policy and SAC on one simulated manipulation task, with our own
dataset tooling, on a laptop without CUDA. The deliverables are the code, the
recorded demonstrations, a results table with confidence intervals, a
success-versus-demos curve, a human-versus-scripted demo ablation, a
robustness test, rollout GIFs and a failure analysis, each number traceable to
the command and run that produced it.

Non-goals: real-robot transfer, more than one task or robot, state-of-the-art
policies, and cloud compute (everything runs on the laptop described below).

## 1. The task and the machine

A Franka Panda (MuJoCo Menagerie model) on a table with a 4 cm cube whose
position and yaw are randomised in a 16 x 16 cm square and a fixed open-top bin.
Physics at 500 Hz, control at 20 Hz. Success: the cube rests on the bin floor
inside the inner box, not held, at episode end.

The machine: Windows 11 Home, Intel Core Ultra 7 258V (4 P + 4 LP-E cores, AVX2
only), 32 GB LPDDR5X, integrated Intel Arc 140V GPU, no CUDA, no MPS. Measured
in milestone 0: the iGPU trains the ResNet18 / 1D-UNet models of this project
15-40 times faster than the CPU, so all CNN/UNet work runs on it through
PyTorch's XPU backend, while SAC and the MLP baseline stay on the CPU.

## 2. System architecture

```
teleop (keyboard / gamepad / scripted expert)
   | 7-D delta-pose + gripper actions at the input rate
   v
env.PandaEnv (gymnasium) -- mink differential IK -> joint targets -> MuJoCo 500 Hz x 25
   | proprio 100 Hz, cameras on their own clocks, actions, privileged state, env observation
   v
data.logger -> Zarr schema v1 (raw streams + synced 20 Hz arrays, per-episode attrs)
   | validate / stats / replay / render / splits / export_lerobot
   v
policies (BC-MLP, BC-ResNet, Diffusion Policy) and train.train_sac (SB3)
   v
eval.evaluate on committed held-out initial-condition sets -> aggregate (bootstrap CIs, paired tests) -> RESULTS.md
```

## 3. Repository layout

```
panda_il/
  paths.py, device.py, device_checks.py     DATA_ROOT resolution; XPU trust logic and equality checks
  env/      scene.xml, scene.py, panda_env.py, ik.py, cameras.py, task.py, sets.py        (M1)
  teleop/   teleop.py, keyboard.py, gamepad.py, expert.py                                 (M3)
  data/     logger.py, schema.py, sync.py, validate.py, stats.py, replay.py, render.py,
            splits.py, export_lerobot.py, dataset.py                                      (M2, M4)
  policies/ encoders.py, normalize.py, bc_mlp.py, bc_resnet.py, diffusion/{unet1d,schedulers,ema,policy}.py (M0, M5)
  train/    loop.py, train_bc.py, train_diffusion.py, train_sac.py                        (M5, M6)
  eval/     evaluate.py, aggregate.py, plots.py, gifs.py, failures.py                     (M5, M7)
scripts/    fetch_assets.py, check_xpu.py, bench_device.py, make_eval_sets.py, collect_expert.py, run_matrix.py
tools/lerobot_env/   isolated LeRobot environment: crosscheck_dataset.py, train_dp.py, rollout_dp.py
tests/      pytest suite (markers: render, viewer, xpu, slow)
assets/franka_emika_panda/   vendored Menagerie model + LICENSE + PROVENANCE.md
configs/    eval_set_v1.json, robust_set_v1.json, robust_set_v1b.json, dev_set_v1.json
docs/       milestone-logs/, figures/, gifs/
```

Heavy artifacts live under `DATA_ROOT` (`C:\robodata\panda-pick-place-il`),
never in the repository, which sits in a OneDrive folder.

## 4. Toolchain and pinned versions

| Component | Version | Why |
|---|---|---|
| Python | 3.12 (uv-managed) | every dependency has cp312 Windows wheels; LeRobot requires >= 3.12; Intel's XPU CI uses 3.12 |
| torch / torchvision (main env) | 2.15.0+xpu / 0.29.2+xpu | the XPU build for the Arc 140V; 2.15 carries XPU correctness fixes absent from 2.11-2.14 |
| Intel graphics driver | >= 32.0.101.9033 (9150 used) | Intel's floor for torch 2.15; on 8626 torch 2.15 returned wrong `nonzero()` counts and crashed optimizer steps on this machine |
| mujoco | 3.17.0 | current; exact replay is only guaranteed within one version |
| mujoco-menagerie | 2026.10.4 | official pip distribution of the Panda model; vendored into `assets/` |
| mink | 1.3.0 (daqp) | differential IK with joint limits and velocity limits; Windows wheels |
| gymnasium | 1.4.0 | environment API |
| stable-baselines3 | 2.9.2 | SAC baseline |
| zarr / numcodecs | 3.4.2 / 0.17.1 | dataset storage (v3 API) |
| imageio-ffmpeg | 0.7.0 | bundled ffmpeg for MP4 and GIF output, no system install |
| pygame-ce / pynput | 2.5.8 / 1.8.2 | gamepad (SDL2) and global keyboard input for teleop |
| tyro / tensorboard | 1.0.16 / 2.21.0 | dataclass CLIs; experiment logging |
| LeRobot (own env) | 0.6.1 with torch 2.11.0+xpu | pins torch < 2.12, hence the second environment |

Both environments are locked (`uv.lock`); the CPU variant used by CI is
selected with `--extra cpu`.

## 5. Decisions

1. **Zarr v3, not HDF5**: chunked compressed image arrays, JSON attributes per episode, append-friendly groups.
2. **Two environments** (torch 2.15 main, torch 2.11 LeRobot), both Python 3.12; the package declares `torch>=2.11,<2.16` so it installs into both.
3. **Observations**: `proprio` (22-D: qpos 7, qvel 7, gripper width, tcp position 3, tcp quaternion 4) and `state` (29-D: proprio + cube pose). Image policies and the LeRobot image cross-check see proprio only; state policies and SAC see `state`.
4. **Action**: 7-D delta TCP pose (max 1 cm, 0.05 rad per step) plus a gripper command in [-1, 1]; the commanded absolute pose and the joint targets are logged too.
5. **IK**: a commanded TCP pose integrated from the deltas (world-frame translation, local-frame rotation, workspace box), mink FrameTask + posture task + joint and velocity limits, one solve per control step, open-loop reference resynced when it drifts more than 0.1 rad from the measured joints; joint targets go to the Menagerie position actuators for 25 substeps.
6. **Success and termination**: cube centre inside the bin's inner box, z within the floor band, in contact with the bin floor and with neither finger, speed under 1 cm/s, for 10 consecutive control steps; otherwise the episode runs to the time limit (300 steps provisional, re-fixed after demo collection as 1.5x the 95th percentile of successful human demo length). SAC gets `terminated` on success and `truncated` at the limit.
7. **Held-out sets** are committed JSON files drawn from named RNG streams; demos are checked for proximity to them.
8. **Only successful episodes are used for training**; the validation split is 20 fixed episodes; demo-count subsets are nested prefixes of a per-seed shuffle of the training split.
9. **Checkpoint rule**: the final checkpoint at a fixed per-modality budget (EMA weights where used), for every method, no selection; the dev set is for curves only; a ledger in RESULTS.md records every evaluation on a held-out set.
10. **Statistics**: per-seed rates, mean and SD, two-level bootstrap CI (seeds, then episodes), the single-level seed bootstrap and a pooled Wilson interval as secondary, paired per-IC tests for method-vs-method claims, pre-stated minimum detectable differences.
11. **One training protocol per modality shared by BC and Diffusion Policy** (AdamW 1e-4, cosine with warmup, EMA, same batch, steps and crops), so only the head and loss differ.
12. **Diffusion Policy** follows Chi et al. 2023 exactly in architecture; the matrix uses a quarter-size UNet (down_dims 128/256/512), the paper size is a config option checked with one seed; DDIM-10 at evaluation everywhere, DDPM-100 reported once.
13. **Camera timing** in integer substeps: capture 10 substeps before each control tick, arrival 8 substeps later, so policies see 20 ms-old images in teleop and evaluation alike.
14. **Data in RAM** for training (`num_workers=0`): Windows spawn workers are slow and cannot return XPU tensors; the full dataset is under 4 GB.
15. **CI** runs lint and the CPU tests on Linux; rendering, viewer and XPU tests run locally and are logged.
16. **Git**: linear history, commit and push per milestone, never rewrite pushed history.

## 6. Milestone plan and exit criteria

| # | Milestone | Exit criteria |
|---|---|---|
| M0 | Bootstrap: repo, environments, package skeleton, CI, XPU self-check, device benchmark | tests green; CI green; benchmark table; self-check verdict recorded |
| M1 | Environment: scene, env, IK, cameras, seeding, held-out sets | determinism test; IK tracking < 2 mm; `check_env`; MP4 of both cameras; timing table |
| M2 | Dataset core: logger, schema, sync, validator | synthetic timing-offset tests; one recorded episode validated |
| M3 | Expert, teleop, replay/render; gamepad mapping | expert >= 90 % on 50 ICs; teleop video; replay reproduces an episode; demos recorded by the author |
| M4 | Stats, splits, LeRobot export and cross-check | LeRobot loads both exports; batch comparison; validator report on the demos |
| M5 | Policies, training, evaluation | loss curves; dev-set success per method on scripted demos; timing per method |
| M6 | SAC baseline | reach-only smoke test; learning curves; env steps and wall-clock |
| M7 | Experiments, LeRobot DP, analysis | results table with CIs, plots, GIFs, failure analysis |
| M8 | Final documentation | every README number traced to a run |

## 7. Compute budget

Re-fitted from the milestone 0 benchmark (`docs/milestone-logs/m0.md`; iGPU,
bf16 autocast for image models, fp32 for the state UNet):

| Run type | Per run | Runs | Total |
|---|---|---|---|
| BC-MLP chunk, single-step, no-EMA (state), 20k steps at 2.7 ms | ~1 min | 55 | ~1 h |
| DP-state, small UNet, 20k steps at 98 ms | ~33 min | 25 | ~14 h |
| DP-state, paper UNet, one seed at 488 ms | ~2.7 h | 1 | ~3 h |
| BC-ResNet chunk, 20k steps at ~200 ms (bf16; encoders without the UNet) | ~1.1 h | 11 | ~12 h |
| BC-ResNet single-step | ~1.1 h | 3 | ~3.5 h |
| DP-image, 20k steps at 261 ms (bf16) | ~1.5 h | 11 | ~16 h |
| Human vs scripted at 100 demos (5 methods) | as above | 15 state + 6 image | ~8 h |
| SAC, 1-2 M env steps (CPU; confirmed in M6) | 2-4 h | 3 + smoke | ~10 h |
| LeRobot DP state + image | ~0.5 h + ~2 h | 2 | ~3 h |
| Dev-set evaluations during training | | | ~5 h |
| Held-out evaluations (~180 checkpoints; policy calls batched across lockstep episodes) | | | ~20 h |

About 95 laptop-hours in total, run through a resumable queue (state-based
matrix first, image runs overnight). The matrix: BC-MLP (chunk, single-step,
no-EMA) and DP-state at 10/25/50/100/all demos x 5 seeds; BC-ResNet and
DP-image at all demos x 5 seeds and at 50/100 demos x 3 seeds; BC-ResNet
single-step x 3 seeds; human versus scripted at 100 demos for all methods;
SAC x 3 seeds; LeRobot DP (state and image) x 1. CPU-only training of the
image policies would take 22 h per run and is not an option.

## 8. Risks and fallbacks

- XPU faults are process-fatal: one process per run, checkpoints every 1k iterations, a resumable queue.
- mink misbehaving on Windows: a damped-least-squares IK behind the same interface.
- Teleop over the 50 ms budget with the viewer open: a single two-viewport render.
- The scripted expert below 90 %: tune grasp pose and timing before recording.
- Any tier that proves infeasible is reported with measured timings and the exact reduction.

## 9. Milestone 0: bootstrap

What was built: the repository and package skeleton, the two uv environments
and their lock files, DATA_ROOT resolution (`paths.py`), the XPU trust logic
(`device.py`, `device_checks.py`, `scripts/check_xpu.py`), the device benchmark
(`scripts/bench_device.py`), the vendored Menagerie Panda
(`scripts/fetch_assets.py`), CI, and the two model definitions that the
benchmark needs and milestone 5 will reuse: the 1D conditional UNet
(`policies/diffusion/unet1d.py`, checked against the official parameter count)
and the ResNet18-GroupNorm + SpatialSoftmax image encoder (`policies/encoders.py`).

Measurements and the self-check verdict: see `docs/milestone-logs/m0.md`.
