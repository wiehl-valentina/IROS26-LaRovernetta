import sys
import types
from pathlib import Path

import pytest

from rover_traversability import weights
from rover_traversability.weights import (
    CheckpointNotFoundError,
    SamNotInstalledError,
    resolve_checkpoint,
    resolve_config,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for var in (weights.ENV_CHECKPOINT, weights.ENV_HF_REPO,
                weights.ENV_HF_FILENAME, weights.ENV_CONFIG):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    yield


def test_explicit_path_returned(tmp_path):
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"x")
    assert resolve_checkpoint(ckpt) == ckpt


def test_explicit_missing_is_error_not_fallthrough(tmp_path):
    with pytest.raises(CheckpointNotFoundError, match="explicitly"):
        resolve_checkpoint(tmp_path / "nope.pt")


def test_env_var_honored(tmp_path, monkeypatch):
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"x")
    monkeypatch.setenv(weights.ENV_CHECKPOINT, str(ckpt))
    assert resolve_checkpoint() == ckpt


def test_env_var_missing_is_error(tmp_path, monkeypatch):
    monkeypatch.setenv(weights.ENV_CHECKPOINT, str(tmp_path / "nope.pt"))
    with pytest.raises(CheckpointNotFoundError, match=weights.ENV_CHECKPOINT):
        resolve_checkpoint()


def test_cache_hit(tmp_path):
    cache = weights.default_cache_dir()
    cache.mkdir(parents=True)
    ckpt = cache / weights.DEFAULT_CHECKPOINT_FILENAME
    ckpt.write_bytes(b"x")
    assert resolve_checkpoint(auto_download=False) == ckpt


def test_nothing_found_message_is_actionable():
    with pytest.raises(CheckpointNotFoundError) as exc:
        resolve_checkpoint(auto_download=False)
    msg = str(exc.value)
    assert weights.ENV_CHECKPOINT in msg
    assert "rover_traversability" in msg  # cache path mentioned
    assert "hf" in msg.lower()


def test_resolve_config_env_override(tmp_path, monkeypatch):
    cfg = tmp_path / "custom.yaml"
    cfg.write_text("model: {}")
    monkeypatch.setenv(weights.ENV_CONFIG, str(cfg))
    assert resolve_config() == cfg


def test_resolve_config_from_fake_sam2(tmp_path, monkeypatch):
    pkg_dir = tmp_path / "sam2"
    cfg = pkg_dir / weights.CONFIG_RELPATH
    cfg.parent.mkdir(parents=True)
    cfg.write_text("model: {}")
    fake = types.ModuleType("sam2")
    fake.__path__ = [str(pkg_dir)]
    monkeypatch.setitem(sys.modules, "sam2", fake)
    assert resolve_config() == cfg


def test_resolve_config_without_sam2_is_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "sam2", None)  # forces ImportError on import

    with pytest.raises((SamNotInstalledError, ImportError)) as exc:
        resolve_config()
    assert "genie" in str(exc.value)


def test_resolve_config_missing_package_data(tmp_path, monkeypatch):
    fake = types.ModuleType("sam2")
    fake.__path__ = [str(tmp_path / "sam2")]
    monkeypatch.setitem(sys.modules, "sam2", fake)
    with pytest.raises(FileNotFoundError, match="package data"):
        resolve_config()
