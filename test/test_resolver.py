"""Resolver helpers (no network required)."""
from __future__ import annotations

import os

import pytest

from audiosronnx.resolver import (
    get_cache_dir,
    is_onnx_path,
    is_url,
    resolve,
    xdg_data_home,
)


def test_is_url():
    assert is_url("http://example.com/m.onnx")
    assert is_url("https://example.com/m.onnx")
    assert not is_url("/local/m.onnx")


def test_is_onnx_path():
    assert is_onnx_path("model.onnx")
    assert is_onnx_path("MODEL.ONNX")
    assert not is_onnx_path("model.bin")


def test_cache_dir_created(tmp_path):
    d = get_cache_dir(str(tmp_path / "cache"))
    assert os.path.isdir(d)


def test_xdg_data_home_env(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert xdg_data_home() == str(tmp_path)


def test_resolve_local_onnx(tmp_path):
    p = os.path.join(tmp_path, "m.onnx")
    with open(p, "wb") as f:
        f.write(b"\x00")
    assert resolve(p) == p


def test_resolve_missing_raises():
    with pytest.raises(FileNotFoundError):
        resolve("nonexistent-model.onnx")
