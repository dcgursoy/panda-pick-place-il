"""CPU-vs-XPU equality, run only where a trusted XPU exists (never in CI)."""

import pytest
import torch

from panda_il import device as device_mod
from panda_il.device_checks import check_conv, check_indexing

pytestmark = pytest.mark.xpu


def _xpu_or_skip() -> torch.device:
    if not device_mod.has_xpu():
        pytest.skip("no XPU device")
    ok, why = device_mod.xpu_check_passed()
    if not ok:
        pytest.skip(f"XPU not trusted: {why}")
    return torch.device("xpu")


def test_nonzero_and_boolean_indexing_match_cpu():
    verdict = check_indexing(_xpu_or_skip())
    assert verdict["ok"], verdict["results"]


def test_conv_and_groupnorm_gradients_match_cpu():
    verdict = check_conv(_xpu_or_skip())
    assert verdict["ok"], {k: v for k, v in verdict["results"].items() if not v["ok"]}


def test_checks_pass_on_cpu_against_itself():
    assert check_indexing(torch.device("cpu"))["ok"]
    assert check_conv(torch.device("cpu"))["ok"]
