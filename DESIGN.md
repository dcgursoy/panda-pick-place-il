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
  policies/ encoders.py, normalize.py, obs_encoder.py, bc.py, checkpoint.py, diffusion/{unet1d,schedulers,ema,policy}.py (M0, M5)
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

Re-fit after milestone 5 (measured on the scripted runs, 20k steps each
with four 20-episode dev evaluations included, `docs/milestone-logs/m5.md`):

| Run type | Measured per run | Runs | Total |
|---|---|---|---|
| BC-MLP chunk, single-step, no-EMA (state), 7 ms/step | 2.5 min | 55 | ~2.5 h |
| DP-state, small UNet, 90 ms/step | 32 min | 25 | ~13.5 h |
| DP-state, paper UNet (not measured; benchmark 488 ms) | ~2.7 h | 1 | ~3 h |
| BC-ResNet chunk, 298 ms/step | 106 min | 11 | ~19.5 h |
| BC-ResNet single-step | ~106 min | 3 | ~5.5 h |
| DP-image, 330 ms/step | 119 min | 11 | ~22 h |
| Human vs scripted at 100 demos (5 methods) | as above | 15 state + 6 image | ~14 h |
| SAC, 1-2 M env steps (CPU; confirmed in M6) | 2-4 h | 3 + smoke | ~10 h |
| LeRobot DP state + image | ~0.5 h + ~2 h | 2 | ~3 h |
| Held-out evaluations (100 episodes: state ~1.5 min, image ~6.5 min with ten lockstep envs) | | ~180 | ~10 h |

About 100 laptop-hours. The image runs are 1.5 to 1.7 times slower than the
milestone 0 benchmark predicted (the benchmark timed the networks alone; a
training step also crops, normalises and moves 256 images, updates the EMA
and runs the optimiser), the state runs match it. One caveat for the matrix:
DP-image was still improving at 20k steps on scripted data (dev success
0 / 35 / 80 / 90 % at 5k / 10k / 15k / 20k, validation loss still falling),
so the image budget is a decision to revisit with the human data before
milestone 7 (section 14, `m5.md`).

## 8. Risks and fallbacks

- XPU faults are process-fatal: one process per run, checkpoints every 1k iterations, a resumable queue.
- mink misbehaving on Windows: a damped-least-squares IK behind the same interface.
- Teleop over the 50 ms budget with the viewer open: a single two-viewport render.
- The scripted expert below 90 %: tune grasp pose and timing before recording.
- Any tier that proves infeasible is reported with measured timings and the exact reduction.

## 9. The environment (milestone 1)

**Scene.** The table top is the z = 0 plane; the Panda base stands on it at the
origin. The cube (4 cm, 50 g) starts in a 16 x 16 cm square centred at (0.50,
-0.15) with a uniform yaw; the bin (inner 12 x 12 x 5 cm) is fixed at (0.50,
0.25). The Panda is attached through MjSpec with the prefix `panda/`, so the
cube's free joint comes first in `qpos` and everything is addressed by name.
Shadows are off (10 ms versus 44 ms per render on this iGPU). The `front`
camera at (0.98, -0.04, 0.84) looks at the workspace centre with a 48 deg
field of view, chosen so that every held-out cube pose and the whole bin rim
stay at least 10 px inside the 96 x 96 image; the `wrist` camera sits on the
+x face of the hand, tilted 20 deg toward the fingertips.

**Gravity compensation.** Menagerie's Panda uses position actuators (kp 4500 /
3500 / 2000) without gravity compensation, which sags the TCP 2-7 mm. Every
Panda body has `gravcomp = 1`, set on the spec before compiling (the compiler
counts compensated bodies). The real Franka controller compensates gravity
too, and mjctrl's reference IK example does the same.

**Control stack.** The 7-D action (3 translation, 3 rotation, gripper) is
scaled to at most 1 cm and 0.05 rad per 20 Hz step and integrated into a
commanded TCP pose: translation in the world frame, rotation about the TCP's
own axes, position clamped to a workspace box. mink solves one QP per step
(frame task on the `tcp` site, posture task with cost 1e-3 anchored at the
home pose, joint-limit and Franka velocity-limit constraints, daqp) on an
arm-only kinematic model, moving an open-loop reference configuration; its
seven arm joints become the actuator targets for the 25 physics substeps. The
reference is resynced to the measured joints (and the commanded pose
re-anchored) when the measured TCP falls more than 4 cm or 0.3 rad behind it,
which only happens when the arm is blocked; the actuator lag at full speed is
about 1.6 cm, and the settled error is 1.2-1.3 mm. A damped-least-squares
backend with the same interface passes the same tests.

**Success.** Cube centre inside the inner box, cube z within [bin floor, floor
+ half edge + 1 cm], in contact with the bin floor and with neither finger,
speed below 1 cm/s, for 10 consecutive control steps; `terminated` on success,
`truncated` at the step limit.

**Cameras.** Each camera captures 10 substeps before every control tick and the
frame arrives 8 substeps later, so a policy always sees a 20 ms-old image; the
reset observation is a zero-latency render. The config rejects any
offset/latency pair whose arrival would coincide with a tick.

**Observations and info.** `proprio` (22-D) and `state` (29-D) as in decision 3;
`info` carries the full `mjSTATE_INTEGRATION` vector, contacts, reward terms,
the commanded pose, the joint targets, the tracking lag and the success flag.
`snapshot()`/`restore()` capture the simulator and the controller together so
an episode can be continued bit-for-bit (replay in milestone 2 builds on this).

Measurements: `docs/milestone-logs/m1.md`.

## 10. The dataset tooling (milestone 2)

**Why our own format.** The brief asks for the logging and the format to be
written here rather than taken from LeRobot, so that every timing decision is
explicit: each stream keeps its own clock, and the alignment onto the control
timeline is a separate, tested step whose errors are measured and stored.

**Streams.** `proprio` at 100 Hz (every fifth physics substep, from a
kinematics pass so the TCP pose is exact), `cam_<name>` on each camera's own
clock with capture and arrival substeps, `action` at the input rate (every
command the teleop or the expert issued), `obs` and `priv` at every control
tick (what the policy saw, and the privileged simulator state, contacts,
reward terms and the executed action), `init` at reset. Every sample carries
simulation time, wall time and the physics substep; the substep is the clock
the sync works in, because it is an integer.

**Sync.** Onto ticks 0, 25, 50, ...: proprio by linear interpolation
(quaternions by normalised lerp), the cube pose from the privileged stream,
cameras by the most recent frame that had *arrived* by the tick (as an index
into the raw frames), actions zero-order held. The maximum alignment error
per stream is stored in the episode attributes; with the default camera
timing it is 20 ms for the cameras and 0 for everything else.

**Store.** One Zarr v3 group per episode under `episodes.zarr/` with `raw/`,
`init/`, `synced/` and the episode attributes; images chunked 32 frames at a
time with Blosc/zstd; a `metadata.json` per dataset with the environment
version and config, rates, camera intrinsics and clock settings, action and
observation specs, library versions, git commit, source, operator, input
device and counts. The source (`human` or `scripted`) is in both the dataset
metadata and every episode, and the validator refuses a mismatch.

**Validator, replay, render.** The validator checks structure, timing, values
and, above all, consistency: a fresh sync must reproduce the stored arrays,
the synced observations must equal the logged ones, and the zero-order-held
commands must equal what the environment executed. Replay re-runs the synced
actions from the logged seed and initial condition and compares the simulator
state at every tick bit for bit; because MuJoCo and the controller are
deterministic, anything but exact equality means the recording or the
environment changed. Render writes the frames the policy saw next to a
third-person view re-rendered from the logged state.

Measurements and the synthetic test cases: `docs/milestone-logs/m2.md`.

## 11. The scripted expert and the teleop tool (milestone 3)

**Expert.** A privileged state machine (align above the cube with the finger
axis matched to the cube's yaw modulo 90 degrees, descend, close for 12
steps, lift, carry over the bin, lower, open, retreat) that moves the
commanded TCP proportionally toward each phase target through the same 7-D
action interface a human uses. It exists to test the pipeline and for the
human-versus-scripted ablation; it can quantise its actions to the keyboard's
{-1, 0, +1} levels so that ablation can be run with matched action
distributions, and everything it records is tagged `source="scripted"`.
Measured: 100 % success on 50 random poses and on every committed held-out
set, 120-153 steps per episode.

**Teleop.** The 20 Hz loop is paced by the wall clock and polls the input at
50 Hz; every polled command is logged at its own time and the command in force
at a tick is what the environment executes. Keys are read with a global
`pynput` listener (press and release), gated on the MuJoCo window being in
the foreground, because the passive viewer only reports key presses and
consumes Space, Backspace, Tab and the F-keys itself; the single-key commands
(N start, K keep, X discard, G gripper, R reset) are letters the viewer does
not use. Gamepads are read through pygame-ce (SDL2) with a mapping file and a
`--calibrate` mode. An `--autopilot` mode drives the same session with the
expert so the whole recording path can be tested without a human.

**Viewer contention.** With the passive viewer open, the two 96 x 96 offscreen
renders per step take ~110 ms instead of ~22 ms: the viewer's render thread
competes for the iGPU. Holding the viewer's lock during the step confines its
drawing to the idle gap and brings the step to ~60 ms, i.e. about 16 Hz
wall-clock at the default window size. The recorded data is simulation-timed
and unaffected; a smaller viewer window helps, and rendering both cameras in
one offscreen pass (one readback) is the remaining optimisation if the feel
matters.

Measurements: `docs/milestone-logs/m3.md`.

## 12. Milestone 0: bootstrap

What was built: the repository and package skeleton, the two uv environments
and their lock files, DATA_ROOT resolution (`paths.py`), the XPU trust logic
(`device.py`, `device_checks.py`, `scripts/check_xpu.py`), the device benchmark
(`scripts/bench_device.py`), the vendored Menagerie Panda
(`scripts/fetch_assets.py`), CI, and the two model definitions that the
benchmark needs and milestone 5 will reuse: the 1D conditional UNet
(`policies/diffusion/unet1d.py`, checked against the official parameter count)
and the ResNet18-GroupNorm + SpatialSoftmax image encoder (`policies/encoders.py`).

Measurements and the self-check verdict: see `docs/milestone-logs/m0.md`.

## 13. Splits, statistics, windows and the LeRobot export (milestone 4)

**Splits** (`data/splits.py`). `splits.json` next to a dataset lists its
training, validation and failed episode ids. Only successful episodes are
used; the validation split is 20 episodes drawn once (seed 0) and never
changes afterwards; failures are listed so their count stays visible. The
demo-count subsets are `subset_ids(train_ids, n, seed)`: the first `n` ids of
a shuffle seeded by `[seed, 7]`, so within a training seed the subsets are
nested (the 25-demo run sees every episode the 10-demo run saw) and across
seeds they differ, which makes the seed spread include "which demonstrations
were drawn". The normaliser is fitted on the subset actually used
(decision 9).

**Statistics** (`data/stats.py`). Per-dimension min / max / mean / std of
proprio, state and action over a split, action histograms and saturation
fractions (|a| > 0.95), the gripper-closed fraction, episode-length summary
and per-channel image means over a sample of episodes. The scripted table is
in `docs/milestone-logs/m3.md`; the normaliser (decision 15) uses the min and
max of the training subset, computed again at training time.

**Windows** (`data/dataset.py`). `WindowDataset` holds every episode in RAM
(the synced 20 Hz arrays plus the delivered camera frames) and indexes
windows the way Diffusion Policy does: a window at step `i` carries the
`obs_horizon` observations ending at `i` and the `pred_horizon` actions
starting at `i`; out-of-range frames repeat the first or last one
(`pad_before = To - 1`, `pad_after = Ta - 1`). Because the action chunk
starts at the current observation, the executed slice at inference is its
first `Ta` actions; these are the same actions Chi et al. execute as
`[To-1 : To-1+Ta]` under their alignment, where the chunk starts `To - 1`
steps earlier. Image policies read `proprio` (22-D), state policies `state`
(29-D); the dataset never hands an image policy the cube pose.

**LeRobot export** (`data/export_lerobot.py`, run in the LeRobot
environment). LeRobot's own `LeRobotDataset.create` / `add_frame` /
`save_episode` write the v3.0 format, so the result is whatever LeRobot
0.6.1 considers valid, not an imitation of it. Two variants of the same
dataset: `<name>_image` (observation.state = proprio, two 96 x 96 cameras,
cube pose under `privileged.cube_pose`, a key LeRobot does not treat as an
observation, so no LeRobot policy can condition on it) and `<name>_state`
(observation.state = proprio, observation.environment_state = cube pose).
Images go in as PNG inside the parquet files (`dtype "image"`) rather than
AV1 video, so the codec is not a confound in the cross-check; only
successful train and validation episodes are exported, and
`export_map.json` maps LeRobot episode indices back to our ids.

**Cross-check** (`tools/lerobot_env/crosscheck_dataset.py`). Loads the export
with `LeRobotDataset`, compares 200 random frames against our Zarr (state,
action, cube pose and decoded images must match exactly), builds a window
with LeRobot's `delta_timestamps` for `To = 2`, `Tp = 16` and checks its
shapes, and asks LeRobot's `dataset_to_policy_features` which keys a policy
would be given: the image variant must list the two cameras and the 22-D
state only. Result for `scripted_v1`: both variants exact, inputs as
required (`docs/milestone-logs/m4.md`). What the cross-check does not cover
is training; that is milestone 7's LeRobot Diffusion Policy run.

## 14. Policies, training and evaluation (milestone 5)

**Normalisation** (`policies/normalize.py`). Min-max per dimension to
[-1, 1] from the training subset; a dimension whose range is below 1e-4 maps
to 0 (the expert never rolls or pitches, so those two action dimensions are
constant in scripted data). Images map to [-1, 1]. The normalisers are
`nn.Module` buffers, so they travel inside the checkpoint.

**Observation encoder** (`policies/obs_encoder.py`), shared by behaviour
cloning and Diffusion Policy so the two differ only in head and loss
(decision 11). State mode: the normalised 29-D state of the last `To = 2`
steps, concatenated (58-D). Image mode: the normalised 22-D proprio plus, per
camera and per step, the ResNet18-GroupNorm + SpatialSoftmax(32) + Linear(64)
encoder from milestone 0 (`policies/encoders.py`, no pretrained weights),
concatenated to 2 x (22 + 2 x 64) = 300-D. Random 84 x 84 crops of the
96 x 96 frames in training, the centre crop at evaluation. An image policy
never sees the cube pose: its lowdim input is `proprio` by construction.

**Behaviour cloning** (`policies/bc.py`). Encoder, then a 3 x 512 MLP that
regresses the normalised action chunk (`Tp = 16` steps x 7); MSE loss. The
single-step ablation is the same model with `pred_horizon = action_horizon = 1`.
With the image encoder this is "BC-ResNet", with the state input "BC-MLP".

**Diffusion Policy** (`policies/diffusion/`). The ConditionalUnet1D of Chi
et al. (2023) from milestone 0 with FiLM conditioning on the encoder output
and the sinusoidal step embedding; `down_dims = (128, 256, 512)` for the
matrix (40.9 M parameters with the image encoders, 9 M for state),
`(256, 512, 1024)` as the paper-size option. Training: DDPM with 100
squared-cosine steps, epsilon prediction, MSE. Inference: DDIM with 10 steps
and eta = 0 by default, DDPM-100 as an option. Both schedulers are written
here (`schedulers.py`) and unit-tested against `diffusers` to 1e-5 on
`add_noise`, every DDPM step and whole 10 / 16-step DDIM trajectories
(`tests/test_policies.py`). Noise is drawn on the CPU from one generator per
batch element, so a sample depends on its own seed only, never on what else
is in the batch. EMA weights (`ema.py`: power 0.75, max decay 0.9999, the
schedule of the reference implementation) are what gets evaluated.

**Training loop** (`train/loop.py`, CLIs `train/train_bc.py` and
`train/train_diffusion.py`). One protocol per modality (decision 11): AdamW
(lr 1e-4, weight decay 1e-6, betas 0.95 / 0.999, fused kernel on the XPU),
cosine schedule with 500 warm-up steps, 20k gradient steps, batch 256 for
state and 64 for images, bf16 autocast on the XPU with fp32 optimiser and
EMA maths, EMA for both methods. The dataset is held in RAM and batched in
the main process (decision 14). A run directory under
`DATA_ROOT/runs/<method>/<dataset>_<obs>_<subset>/seed<k>` holds
`config.json` (resolved configs, episode ids, device, git commit),
TensorBoard events (train and validation loss, learning rate, ms per step,
dev-set success), checkpoints every 5k steps plus `last.pt`, and
`summary.json` with the timings. Checkpoints hold raw and EMA weights, the
normalisers, both configs and the provenance; `policies/checkpoint.py`
rebuilds a policy from one. Seeding covers Python, NumPy, torch and the XPU;
the DataLoader has its own seeded generator and `subset_ids` makes the
demonstration subset a function of the seed.

**Evaluation** (`eval/evaluate.py`). `n_envs` environments run in lockstep
so one batched network call serves all of them, which matters on this iGPU
where a single-sample 10-step denoise is launch-bound (~180 ms) and a batch of
ten costs about the same. Every episode is seeded from (evaluation seed, IC
index) with `SeedSequence`, including its diffusion noise, so results are
reproducible and independent of batch composition. The policy sees the last
`To` observations (edge padded after reset, as training windows are), predicts
a chunk and executes its first `Ta = 8` actions (section 13), clipped to
[-1, 1]. Each episode gets a failure label from the privileged log
(`eval/failures.py`, section 7.6's decision tree: grasp event, release event,
where the cube ended) and the report records per-episode outcomes, label
counts, inference ms per call, env ms per step, device and git state. GIFs
show the third-person view next to what the policy saw. Evaluating on a
held-out set appends to `docs/eval_ledger.jsonl` (decision 9); the dev set is
used for curves during training (`--train.dev-eval-every`) and for this
milestone's smoke numbers.

Measurements: `docs/milestone-logs/m5.md`.
