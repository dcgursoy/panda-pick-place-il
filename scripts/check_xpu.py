"""Self-check of the Intel XPU backend. Run it after installing or after any driver change.

    uv run python scripts/check_xpu.py

It compares the installed Intel graphics driver with Intel's floor for this torch
version, then runs the CPU-vs-XPU equality checks from panda_il.device_checks,
each in its own subprocess so that a fatal Level Zero error cannot take down the
parent. The verdict is written to DATA_ROOT/xpu_check.json, which
panda_il.device.resolve_device consults before handing out the XPU. Exit code 0
means the XPU is trusted.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import torch

from panda_il import device as device_mod
from panda_il.device_checks import ALL_CHECKS
from panda_il.paths import Paths

WORKER_TIMEOUT_S = 300


def run_worker(check_name: str) -> dict:
    """Run one check in a fresh interpreter and parse its JSON verdict."""
    cmd = [sys.executable, __file__, "--worker", check_name]
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=WORKER_TIMEOUT_S, check=False
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timed out after {WORKER_TIMEOUT_S} s (hung accelerator?)"}
    elapsed = time.perf_counter() - started
    verdict_line = next(
        (line for line in reversed(proc.stdout.splitlines()) if line.startswith("{")), None
    )
    if proc.returncode != 0 or verdict_line is None:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-5:]
        return {
            "ok": False,
            "error": f"worker exit {proc.returncode}: " + " | ".join(tail),
            "seconds": elapsed,
        }
    verdict = json.loads(verdict_line)
    verdict["seconds"] = elapsed
    return verdict


def worker_main(check_name: str) -> None:
    result = ALL_CHECKS[check_name](torch.device("xpu"))
    print(json.dumps(result))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--worker", choices=sorted(ALL_CHECKS), help=argparse.SUPPRESS)
    parser.add_argument(
        "--data-root", default=None, help="where to write xpu_check.json (default: DATA_ROOT)"
    )
    args = parser.parse_args()
    if args.worker:
        worker_main(args.worker)
        return 0

    paths = Paths.resolve(args.data_root)
    print(f"torch {torch.__version__}")
    if not device_mod.has_xpu():
        record = device_mod.write_check_record(
            paths, False, "torch.xpu.is_available() is False", {}
        )
        print(f"no XPU device visible to torch; wrote {record}")
        return 1

    props = torch.xpu.get_device_properties(0)
    driver = device_mod.windows_gpu_driver_version()
    floor_ok = device_mod.driver_meets_floor(driver)
    minor = ".".join(torch.__version__.split("+")[0].split(".")[:2])
    print(
        f"device: {props.name}, {props.total_memory / 2**30:.1f} GiB, Level Zero driver {props.driver_version}"
    )
    print(
        f"Windows driver package: {driver or 'unknown'}; Intel's floor for torch {minor}: 32.0.101.{device_mod.DRIVER_FLOOR.get(minor, '?')} -> {'ok' if floor_ok else 'BELOW FLOOR' if floor_ok is False else 'unknown'}"
    )

    details: dict = {"driver_floor_ok": floor_ok, "level_zero_driver": props.driver_version}
    all_ok = True
    for name in ALL_CHECKS:
        verdict = run_worker(name)
        details[name] = verdict
        status = "pass" if verdict.get("ok") else "FAIL"
        print(
            f"  {name:10s} {status}  ({verdict.get('seconds', 0):.1f} s)"
            + (f"  {verdict['error']}" if "error" in verdict else "")
        )
        if not verdict.get("ok"):
            all_ok = False
            for case, numbers in verdict.get("results", {}).items():
                if (
                    isinstance(numbers, dict)
                    and numbers.get("ok") is False
                    or (
                        isinstance(numbers, dict)
                        and "cpu" in numbers
                        and len({v for k, v in numbers.items() if k != "ok"}) > 1
                    )
                ):
                    print(f"             {case}: {numbers}")

    passed = all_ok and floor_ok is not False
    if passed:
        summary = "all checks passed"
    elif floor_ok is False and all_ok:
        summary = f"checks passed but driver {driver} is below Intel's floor for torch {minor}; update the driver"
    else:
        failed = [name for name in ALL_CHECKS if not details[name].get("ok")]
        summary = (
            "failed checks: "
            + ", ".join(failed)
            + ("" if floor_ok is not False else "; driver below floor")
        )
    record = device_mod.write_check_record(paths, passed, summary, details)
    print(f"\nverdict: {'TRUSTED' if passed else 'NOT TRUSTED'} - {summary}\nrecord: {record}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
