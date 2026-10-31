"""CPU-versus-accelerator equality checks used by scripts/check_xpu.py and the test suite.

Each check returns a dict with an ``ok`` flag and the raw numbers, so a failure
is explainable. They target the operations this project depends on and the
ones Intel's XPU backend has got wrong on old drivers: ``nonzero`` / boolean
indexing, conv1d forward and backward (including stride-2 downsampling and
length-1 inputs), GroupNorm gradients, and fused / foreach AdamW steps.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def _close(a: torch.Tensor, b: torch.Tensor, rtol: float, atol: float) -> tuple[bool, float]:
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    err = float((a - b).abs().max()) if a.numel() else 0.0
    return bool(torch.allclose(a, b, rtol=rtol, atol=atol)), err


def check_indexing(device: torch.device | str) -> dict:
    """nonzero() and boolean indexing must count exactly like the CPU."""
    torch.manual_seed(0)
    results: dict[str, dict] = {}
    ok = True
    for n in (2061, 5000, 100_000, 1_000_000):
        mask = torch.rand(n) > 0.5
        cpu_count = int(mask.sum())
        mask_dev = mask.to(device)
        nonzero_count = int(mask_dev.nonzero().numel())
        index_count = int(torch.arange(n, device=device)[mask_dev].numel())
        results[f"mask_{n}"] = {"cpu": cpu_count, "nonzero": nonzero_count, "index": index_count}
        ok = ok and cpu_count == nonzero_count == index_count
    mask2 = torch.rand(512, 4096) > 0.5
    cpu_count = int(mask2.sum())
    dev_count = int(mask2.to(device).nonzero().shape[0])
    results["mask_2d"] = {"cpu": cpu_count, "nonzero": dev_count}
    ok = ok and cpu_count == dev_count
    return {"ok": ok, "results": results}


def check_conv(device: torch.device | str) -> dict:
    """conv1d / conv2d / GroupNorm forward and backward must match the CPU to fp32 tolerance."""
    torch.manual_seed(0)
    results: dict[str, dict] = {}
    ok = True
    cases = {
        "conv1d_k5": (nn.Conv1d(7, 64, 5, padding=2), (8, 7, 16)),
        "conv1d_stride2": (nn.Conv1d(64, 64, 3, stride=2, padding=1), (8, 64, 16)),
        "conv1d_len1": (nn.Conv1d(64, 64, 3, stride=2, padding=1), (8, 64, 1)),
        "conv1d_depthwise_len1": (nn.Conv1d(64, 64, 3, stride=2, padding=1, groups=64), (8, 64, 1)),
        "conv_transpose1d": (nn.ConvTranspose1d(64, 64, 4, stride=2, padding=1), (8, 64, 8)),
        "groupnorm1d": (nn.GroupNorm(8, 64), (8, 64, 16)),
        "conv2d_k3": (nn.Conv2d(3, 32, 3, stride=2, padding=1), (4, 3, 84, 84)),
        "groupnorm2d": (nn.GroupNorm(4, 64), (4, 64, 11, 11)),
    }
    for name, (module, shape) in cases.items():
        x = torch.randn(*shape, requires_grad=True)
        ref_module = module
        out_ref = ref_module(x)
        grad_out = torch.randn_like(out_ref)
        out_ref.backward(grad_out)
        ref_grads = [p.grad.clone() for p in ref_module.parameters()]
        ref_xgrad = x.grad.clone()

        dev_module = type(module)  # placeholder to keep the structure obvious below
        del dev_module
        module_dev = _clone_module(ref_module).to(device)
        x_dev = x.detach().to(device).requires_grad_(True)
        out_dev = module_dev(x_dev)
        out_dev.backward(grad_out.to(device))

        fwd_ok, fwd_err = _close(out_ref, out_dev, rtol=1e-4, atol=1e-4)
        xgrad_ok, xgrad_err = _close(ref_xgrad, x_dev.grad, rtol=1e-3, atol=1e-4)
        pgrad_ok = True
        pgrad_err = 0.0
        for g_ref, p_dev in zip(ref_grads, module_dev.parameters(), strict=True):
            good, err = _close(g_ref, p_dev.grad, rtol=1e-3, atol=1e-4)
            pgrad_ok = pgrad_ok and good
            pgrad_err = max(pgrad_err, err)
        case_ok = fwd_ok and xgrad_ok and pgrad_ok
        results[name] = {
            "ok": case_ok,
            "forward_max_err": fwd_err,
            "input_grad_max_err": xgrad_err,
            "param_grad_max_err": pgrad_err,
        }
        ok = ok and case_ok
    return {"ok": ok, "results": results}


def _clone_module(module: nn.Module) -> nn.Module:
    import copy

    clone = copy.deepcopy(module)
    for p in clone.parameters():
        p.grad = None
    return clone


def check_optimizer(device: torch.device | str, steps: int = 200) -> dict:
    """AdamW must run ``steps`` updates on a small MLP without errors or NaNs, for each path.

    On torch 2.15 with an old driver the default (foreach) path raised fatal Level Zero
    errors within the first few steps; run this in a subprocess from the CLI.
    """
    torch.manual_seed(0)
    results: dict[str, dict] = {}
    ok = True
    for path_name, kwargs in (
        ("foreach", {"foreach": True}),
        ("single_tensor", {"foreach": False}),
        ("fused", {"fused": True}),
    ):
        model = nn.Sequential(
            nn.Linear(44, 512), nn.Mish(), nn.Linear(512, 512), nn.Mish(), nn.Linear(512, 112)
        ).to(device)
        try:
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6, **kwargs)
        except (RuntimeError, ValueError, TypeError) as exc:
            results[path_name] = {"ok": False, "error": f"unsupported: {exc}"}
            continue
        x = torch.randn(256, 44, device=device)
        target = torch.randn(256, 112, device=device)
        first = last = math.nan
        try:
            for step in range(steps):
                loss = nn.functional.mse_loss(model(x), target)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                value = float(loss)
                if step == 0:
                    first = value
                last = value
            path_ok = math.isfinite(first) and math.isfinite(last) and last < first
            results[path_name] = {"ok": path_ok, "first_loss": first, "last_loss": last}
        except RuntimeError as exc:
            results[path_name] = {"ok": False, "error": str(exc)[:300]}
            path_ok = False
        ok = ok and path_ok
    return {"ok": ok, "results": results}


ALL_CHECKS = {"indexing": check_indexing, "conv": check_conv, "optimizer": check_optimizer}
