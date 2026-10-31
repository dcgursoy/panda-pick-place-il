from pathlib import Path

from panda_il import paths


def test_explicit_argument_wins(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / "from_env"))
    assert paths.resolve_data_root(tmp_path / "explicit") == tmp_path / "explicit"


def test_environment_variable_beats_default(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / "from_env"))
    assert paths.resolve_data_root() == tmp_path / "from_env"


def test_default_depends_on_platform(monkeypatch):
    monkeypatch.delenv(paths.ENV_VAR, raising=False)
    monkeypatch.setattr(paths.sys, "platform", "win32")
    assert paths.default_data_root() == Path(r"C:\robodata\panda-pick-place-il")
    monkeypatch.setattr(paths.sys, "platform", "linux")
    assert paths.default_data_root() == Path.home() / "robodata" / "panda-pick-place-il"


def test_paths_layout(tmp_path):
    p = paths.Paths(tmp_path)
    p.make_dirs()
    assert p.datasets.is_dir() and p.runs.is_dir() and p.videos.is_dir() and p.lerobot.is_dir()
    assert p.xpu_check_file.parent == tmp_path
