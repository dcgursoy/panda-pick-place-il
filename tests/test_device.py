import json

import pytest
import torch

from panda_il import device as device_mod
from panda_il.paths import Paths


def test_cpu_is_always_available():
    assert device_mod.resolve_device("cpu").type == "cpu"


def test_auto_without_record_falls_back_to_cpu(tmp_path):
    assert device_mod.resolve_device("auto", Paths(tmp_path)).type == "cpu"


def test_explicit_xpu_without_record_is_refused(tmp_path):
    with pytest.raises(RuntimeError):
        device_mod.resolve_device("xpu", Paths(tmp_path))


def test_record_must_match_installed_torch(tmp_path):
    paths = Paths(tmp_path)
    paths.xpu_check_file.write_text(json.dumps({"passed": True, "torch": "0.0.0", "driver": None}))
    ok, why = device_mod.xpu_check_passed(paths)
    assert not ok and "0.0.0" in why


def test_passing_record_is_trusted(tmp_path, monkeypatch):
    monkeypatch.setattr(device_mod, "windows_gpu_driver_version", lambda: "32.0.101.9150")
    paths = Paths(tmp_path)
    paths.xpu_check_file.write_text(
        json.dumps({"passed": True, "torch": torch.__version__, "driver": "32.0.101.9150"})
    )
    assert device_mod.xpu_check_passed(paths) == (True, "ok")


def test_driver_change_invalidates_record(tmp_path, monkeypatch):
    monkeypatch.setattr(device_mod, "windows_gpu_driver_version", lambda: "32.0.101.9150")
    paths = Paths(tmp_path)
    paths.xpu_check_file.write_text(
        json.dumps({"passed": True, "torch": torch.__version__, "driver": "32.0.101.8626"})
    )
    ok, why = device_mod.xpu_check_passed(paths)
    assert not ok and "driver changed" in why


@pytest.mark.parametrize(
    ("driver", "torch_version", "expected"),
    [
        ("32.0.101.9150", "2.15.0+xpu", True),
        ("32.0.101.8626", "2.15.0+xpu", False),
        ("32.0.101.8626", "2.11.0+xpu", True),
        (None, "2.15.0+xpu", None),
        ("32.0.101.9150", "9.9.9", None),
    ],
)
def test_driver_floor(driver, torch_version, expected):
    assert device_mod.driver_meets_floor(driver, torch_version) is expected


def test_unknown_device_name():
    with pytest.raises(ValueError):
        device_mod.resolve_device("tpu")
