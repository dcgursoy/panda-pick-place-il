# Results: raw data behind the numbers

Every number in the README is traced here to the command and run directory
that produced it. Nothing in this file is filled in by hand from memory; each
entry is pasted from a run's output.

## 0. Where each README number comes from

| README number | Section below | Command | Run directory |
|---|---|---|---|
| (none yet) | | | |

## 1. Evaluation ledger

Every evaluation that touches a held-out initial-condition set
(`eval_set_v1`, `robust_set_v1`, `robust_set_v1b`) is appended here by
`panda_il.eval.evaluate`, so the number of looks at the test sets is auditable.

| Date (UTC) | Run / checkpoint | IC set | Episodes | Purpose | Success | Command |
|---|---|---|---|---|---|---|
| (none yet) | | | | | | |

## 2. Milestone 0: device benchmark and XPU self-check

Command: `uv run python scripts/bench_device.py --force-xpu --iters 10 --warmup 2`
(2026-10-31, torch 2.15.0+xpu, driver 32.0.101.8626, 8 CPU threads). Raw JSON under
`DATA_ROOT/bench/`: `bench_cpu_20261031_020339.json`, `bench_xpu_20261031_020510.json`.
The XPU column was measured on a driver below Intel's floor (timings only; the
correctness self-check failed on it, see below) and is re-measured after the update.

| benchmark (ms per iteration) | CPU | XPU fp32 | XPU bf16 autocast |
|---|---:|---:|---:|
| MLP-BC 3x512, batch 256, train step | 15.0 | 2.7 | |
| ResNet18 84x84, batch 64, train step | 933.5 | 100.8 | 50.8 |
| UNet small (16.7 M), batch 256, train step | 1311.8 | 98.2 | 91.5 |
| UNet paper (66.1 M), batch 256, train step | 5105.6 | 488.4 | |
| Image DP full step, batch 64 | 4046.0 | 590.7 | 261.3 |
| UNet small, 10 denoising steps, batch 1 | 187.7 | 177.7 | |
| UNet paper, 10 denoising steps, batch 1 | 225.0 | 221.2 | |
| ResNet18, 4 images, inference | 28.5 | 13.8 | |

XPU self-check (`uv run python scripts/check_xpu.py`) on driver 32.0.101.8626:
`nonzero` / boolean indexing returned 506 of 1023, 1998 of 2507, 22470 of 50059
and 31551 of 500608 true entries; conv and optimizer checks passed; verdict NOT
TRUSTED. Full output in `docs/milestone-logs/m0.md`.
