"""Time the training steps and inference calls this project will run, on CPU and XPU.

    uv run python scripts/bench_device.py            # CPU in-process, then XPU in a subprocess
    uv run python scripts/bench_device.py --device cpu
    uv run python scripts/bench_device.py --device xpu --force-xpu   # measure even if the self-check failed

The numbers feed the compute budget in DESIGN.md. Results are printed as a
markdown table and saved as JSON under DATA_ROOT/bench/.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime

import torch
from torch import nn

from panda_il import device as device_mod
from panda_il.paths import Paths
from panda_il.policies.diffusion.unet1d import ConditionalUnet1D, count_parameters
from panda_il.policies.encoders import ImageEncoder

ACTION_DIM = 7
HORIZON = 16
PROPRIO_DIM = 22
STATE_DIM = 29
OBS_STEPS = 2
CROP = 84

UNET_SMALL = dict(diffusion_step_embed_dim=128, down_dims=(128, 256, 512))
UNET_PAPER = dict(diffusion_step_embed_dim=256, down_dims=(256, 512, 1024))


def make_optimizer(params, device: torch.device) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        params, lr=1e-4, weight_decay=1e-6, betas=(0.95, 0.999), fused=device.type == "xpu"
    )


def synchronize(device: torch.device) -> None:
    if device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def time_fn(fn: Callable[[], None], device: torch.device, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    synchronize(device)
    return (time.perf_counter() - start) / iters * 1000.0


def autocast(device: torch.device, dtype: torch.dtype | None):
    if dtype is None:
        return torch.autocast(device.type, enabled=False)
    return torch.autocast(device.type, dtype=dtype)


def bench_mlp(device: torch.device, batch: int = 256) -> Callable[[], None]:
    model = nn.Sequential(
        nn.Linear(STATE_DIM * OBS_STEPS, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, HORIZON * ACTION_DIM),
    ).to(device)
    opt = make_optimizer(model.parameters(), device)
    x = torch.randn(batch, STATE_DIM * OBS_STEPS, device=device)
    y = torch.randn(batch, HORIZON * ACTION_DIM, device=device)

    def step() -> None:
        loss = nn.functional.mse_loss(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    return step


def bench_encoder_train(
    device: torch.device, dtype: torch.dtype | None, batch: int = 64
) -> Callable[[], None]:
    encoder = ImageEncoder().to(device)
    head = nn.Linear(64, HORIZON * ACTION_DIM).to(device)
    opt = make_optimizer(list(encoder.parameters()) + list(head.parameters()), device)
    x = torch.rand(batch, 3, CROP, CROP, device=device) * 2 - 1
    y = torch.randn(batch, HORIZON * ACTION_DIM, device=device)

    def step() -> None:
        with autocast(device, dtype):
            loss = nn.functional.mse_loss(head(encoder(x)).float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    return step


def bench_unet_train(
    device: torch.device, dtype: torch.dtype | None, cfg: dict, batch: int
) -> Callable[[], None]:
    unet = ConditionalUnet1D(ACTION_DIM, STATE_DIM * OBS_STEPS, **cfg).to(device)
    opt = make_optimizer(unet.parameters(), device)
    actions = torch.randn(batch, HORIZON, ACTION_DIM, device=device)
    cond = torch.randn(batch, STATE_DIM * OBS_STEPS, device=device)
    timesteps = torch.randint(0, 100, (batch,), device=device)

    def step() -> None:
        with autocast(device, dtype):
            noise = torch.randn_like(actions)
            pred = unet(actions + noise, timesteps, cond).float()
            loss = nn.functional.mse_loss(pred, noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    return step


def bench_image_dp_train(
    device: torch.device, dtype: torch.dtype | None, batch: int = 64
) -> Callable[[], None]:
    """A full image Diffusion Policy step: two cameras x two observation steps through two encoders, then the small UNet."""
    encoders = nn.ModuleList([ImageEncoder(), ImageEncoder()]).to(device)
    cond_dim = (64 * 2 + PROPRIO_DIM) * OBS_STEPS
    unet = ConditionalUnet1D(ACTION_DIM, cond_dim, **UNET_SMALL).to(device)
    opt = make_optimizer(list(encoders.parameters()) + list(unet.parameters()), device)
    images = torch.rand(batch, OBS_STEPS, 2, 3, CROP, CROP, device=device) * 2 - 1
    proprio = torch.randn(batch, OBS_STEPS, PROPRIO_DIM, device=device)
    actions = torch.randn(batch, HORIZON, ACTION_DIM, device=device)
    timesteps = torch.randint(0, 100, (batch,), device=device)

    def step() -> None:
        with autocast(device, dtype):
            feats = []
            for cam, encoder in enumerate(encoders):
                frames = images[:, :, cam].reshape(batch * OBS_STEPS, 3, CROP, CROP)
                feats.append(encoder(frames).reshape(batch, OBS_STEPS, 64))
            cond = torch.cat(feats + [proprio], dim=-1).reshape(batch, -1)
            noise = torch.randn_like(actions)
            pred = unet(actions + noise, timesteps, cond).float()
            loss = nn.functional.mse_loss(pred, noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    return step


def bench_unet_infer(device: torch.device, cfg: dict, steps: int = 10) -> Callable[[], None]:
    unet = ConditionalUnet1D(ACTION_DIM, STATE_DIM * OBS_STEPS, **cfg).to(device).eval()
    cond = torch.randn(1, STATE_DIM * OBS_STEPS, device=device)

    @torch.no_grad()
    def run() -> None:
        x = torch.randn(1, HORIZON, ACTION_DIM, device=device)
        for t in range(steps):
            x = x - 0.1 * unet(x, t, cond)

    return run


def bench_encoder_infer(device: torch.device) -> Callable[[], None]:
    encoder = ImageEncoder().to(device).eval()
    x = torch.rand(4, 3, CROP, CROP, device=device) * 2 - 1

    @torch.no_grad()
    def run() -> None:
        encoder(x)

    return run


def run_benchmarks(device: torch.device, iters: int, warmup: int, quick: bool) -> dict[str, float]:
    bf16 = torch.bfloat16 if device.type != "cpu" else None
    cases: list[tuple[str, Callable[[], Callable[[], None]]]] = [
        ("mlp_bc bs256 train", lambda: bench_mlp(device)),
        ("resnet18 84x84 bs64 train fp32", lambda: bench_encoder_train(device, None)),
        (
            "unet_small (128,256,512) bs256 train fp32",
            lambda: bench_unet_train(device, None, UNET_SMALL, 256),
        ),
        (
            "unet_paper (256,512,1024) bs256 train fp32",
            lambda: bench_unet_train(device, None, UNET_PAPER, 256),
        ),
        ("dp_image full step bs64 train fp32", lambda: bench_image_dp_train(device, None)),
        ("unet_small infer 10 steps bs1", lambda: bench_unet_infer(device, UNET_SMALL)),
        ("unet_paper infer 10 steps bs1", lambda: bench_unet_infer(device, UNET_PAPER)),
        ("resnet18 infer 4 images", lambda: bench_encoder_infer(device)),
    ]
    if bf16 is not None:
        cases[2:2] = [("resnet18 84x84 bs64 train bf16", lambda: bench_encoder_train(device, bf16))]
        cases[4:4] = [
            ("unet_small bs256 train bf16", lambda: bench_unet_train(device, bf16, UNET_SMALL, 256))
        ]
        cases[7:7] = [
            ("dp_image full step bs64 train bf16", lambda: bench_image_dp_train(device, bf16))
        ]
    if quick:
        cases = [c for c in cases if "paper" not in c[0]]
    results: dict[str, float] = {}
    for name, make in cases:
        fn = make()
        n_iters = (
            max(2, iters // 4)
            if device.type == "cpu" and ("unet" in name or "dp_image" in name or "resnet" in name)
            else iters
        )
        ms = time_fn(fn, device, n_iters, warmup)
        results[name] = ms
        print(f"| {name:44s} | {ms:10.1f} |", flush=True)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", choices=["all", "cpu", "xpu"], default="all")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--quick", action="store_true", help="skip the paper-size UNet")
    parser.add_argument(
        "--force-xpu",
        action="store_true",
        help="benchmark the XPU even if the self-check did not pass",
    )
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    paths = Paths.resolve(args.data_root)
    paths.make_dirs()

    if args.device == "all":
        rc = subprocess.call([sys.executable, __file__, "--device", "cpu", *_passthrough(args)])
        if device_mod.has_xpu():
            print(
                "\nXPU benchmarks run in a separate process so a fatal Level Zero error cannot hide the CPU numbers."
            )
            rc |= subprocess.call(
                [sys.executable, __file__, "--device", "xpu", *_passthrough(args)]
            )
        return rc

    if args.device == "xpu":
        if not device_mod.has_xpu():
            print("no XPU available")
            return 1
        trusted, why = device_mod.xpu_check_passed(paths)
        if not trusted and not args.force_xpu:
            print(f"XPU not trusted ({why}); pass --force-xpu to measure anyway")
            return 1
        device = torch.device("xpu")
        label = "xpu" + ("" if trusted else " (untrusted: self-check not passed)")
    else:
        device = torch.device("cpu")
        label = f"cpu ({torch.get_num_threads()} threads)"

    print(f"\n### {label}: {device_mod.device_summary(device)}")
    unet_small = count_parameters(
        ConditionalUnet1D(ACTION_DIM, STATE_DIM * OBS_STEPS, **UNET_SMALL)
    )
    unet_paper = count_parameters(
        ConditionalUnet1D(ACTION_DIM, STATE_DIM * OBS_STEPS, **UNET_PAPER)
    )
    print(
        f"UNet parameters: small {unet_small / 1e6:.1f} M, paper {unet_paper / 1e6:.1f} M; ResNet18 encoder {count_parameters(ImageEncoder()) / 1e6:.1f} M\n"
    )
    print(f"| {'benchmark':44s} | {'ms / iter':>10s} |")
    print(f"|{'-' * 46}|{'-' * 12}|")
    results = run_benchmarks(device, args.iters, args.warmup, args.quick)
    record = {
        "device": label,
        "torch": torch.__version__,
        "driver": device_mod.windows_gpu_driver_version(),
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "iters": args.iters,
        "ms_per_iter": results,
    }
    out = paths.bench / f"bench_{device.type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\nsaved {out}")
    return 0


def _passthrough(args: argparse.Namespace) -> list[str]:
    extra = ["--iters", str(args.iters), "--warmup", str(args.warmup)]
    if args.quick:
        extra.append("--quick")
    if args.force_xpu:
        extra.append("--force-xpu")
    if args.data_root:
        extra += ["--data-root", args.data_root]
    return extra


if __name__ == "__main__":
    sys.exit(main())
