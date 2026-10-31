"""Pick the torch device, and refuse the Intel XPU unless its self-check passed.

On this laptop the only accelerator is an Intel Arc 140V iGPU, used through
PyTorch's XPU backend. With a graphics driver older than Intel's floor for the
installed torch, XPU kernels can return silently wrong results (observed with
torch 2.15 on driver 32.0.101.8626: wrong ``nonzero`` counts, fatal optimizer
steps). So ``scripts/check_xpu.py`` runs a set of CPU-vs-XPU equality tests and
writes its verdict to ``DATA_ROOT/xpu_check.json``; this module only hands out
the XPU device when that record exists, passed, and matches the installed torch
and driver.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

from panda_il.paths import Paths

log = logging.getLogger(__name__)

# Minimum Windows Intel graphics driver (last component of 32.0.101.xxxx) per torch minor
# version, from Intel's "PyTorch Prerequisites for Intel GPUs" pages.
DRIVER_FLOOR = {"2.11": 6127, "2.12": 6127, "2.13": 8801, "2.14": 8801, "2.15": 9033}


def has_xpu() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def windows_gpu_driver_version() -> str | None:
    """The Windows driver package version (e.g. '32.0.101.9150'), or None off Windows."""
    if sys.platform != "win32":
        return None
    cmd = (
        "(Get-CimInstance Win32_VideoController | Where-Object { $_.Name -like '*Arc*' "
        "-or $_.Name -like '*Intel*' } | Select-Object -First 1).DriverVersion"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", cmd],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    version = out.stdout.strip()
    return version or None


def driver_meets_floor(
    driver_version: str | None, torch_version: str = torch.__version__
) -> bool | None:
    """True/False when both sides are known, None when the comparison cannot be made."""
    if not driver_version:
        return None
    minor = ".".join(torch_version.split("+")[0].split(".")[:2])
    floor = DRIVER_FLOOR.get(minor)
    try:
        build = int(driver_version.split(".")[-1])
    except ValueError:
        return None
    if floor is None:
        return None
    return build >= floor


def xpu_check_passed(paths: Paths | None = None) -> tuple[bool, str]:
    """Whether the recorded self-check allows using the XPU with the installed torch and driver."""
    paths = paths or Paths.resolve()
    record_file = paths.xpu_check_file
    if not record_file.exists():
        return (
            False,
            f"no self-check record at {record_file}; run `uv run python scripts/check_xpu.py`",
        )
    record = json.loads(record_file.read_text(encoding="utf-8"))
    if not record.get("passed"):
        return False, f"the last self-check failed ({record.get('summary', 'no summary')})"
    if record.get("torch") != torch.__version__:
        return (
            False,
            f"self-check was run with torch {record.get('torch')}, installed is {torch.__version__}",
        )
    current_driver = windows_gpu_driver_version()
    if current_driver and record.get("driver") and current_driver != record["driver"]:
        return (
            False,
            f"driver changed since the self-check ({record['driver']} -> {current_driver})",
        )
    return True, "ok"


def write_check_record(paths: Paths, passed: bool, summary: str, details: dict) -> Path:
    paths.root.mkdir(parents=True, exist_ok=True)
    record = {
        "passed": passed,
        "summary": summary,
        "torch": torch.__version__,
        "driver": windows_gpu_driver_version(),
        "device": torch.xpu.get_device_name(0) if has_xpu() else None,
        "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "details": details,
    }
    paths.xpu_check_file.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return paths.xpu_check_file


def resolve_device(name: str = "auto", paths: Paths | None = None) -> torch.device:
    """'cpu', 'xpu', 'cuda[:n]' or 'auto' (XPU if present and trusted, else CPU)."""
    name = name.lower()
    if name == "cpu":
        return torch.device("cpu")
    if name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"{name} requested but CUDA is not available")
        return torch.device(name)
    if name == "xpu":
        if not has_xpu():
            raise RuntimeError("xpu requested but torch.xpu.is_available() is False")
        ok, why = xpu_check_passed(paths)
        if not ok:
            raise RuntimeError(f"xpu requested but it is not trusted: {why}")
        return torch.device("xpu")
    if name == "auto":
        if has_xpu():
            ok, why = xpu_check_passed(paths)
            if ok:
                return torch.device("xpu")
            log.warning("XPU is present but not trusted (%s); falling back to CPU", why)
        return torch.device("cpu")
    raise ValueError(f"unknown device name {name!r}")


def device_summary(device: torch.device) -> str:
    if device.type == "xpu":
        props = torch.xpu.get_device_properties(0)
        return f"xpu: {props.name}, {props.total_memory / 2**30:.1f} GiB, torch {torch.__version__}"
    if device.type == "cuda":
        return f"cuda: {torch.cuda.get_device_name(device)}, torch {torch.__version__}"
    return f"cpu: {torch.get_num_threads()} threads, torch {torch.__version__}"
