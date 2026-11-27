"""Family contract: array input needs a rate, and every Hub download is pinned."""
import importlib
import pkgutil
import re

import numpy as np
import pytest

import audiosronnx.engines
from audiosronnx.audio import load_audio

SHA = re.compile(r"[0-9a-f]{40}")


def test_array_without_rate_raises():
    with pytest.raises(ValueError, match="sample_rate is required"):
        load_audio(np.zeros(160, dtype=np.float32))


def test_array_with_rate_keeps_the_rate():
    x = np.array([0.25, -0.5], dtype=np.float32)
    out, sr = load_audio(x, sample_rate=8000)
    assert sr == 8000
    np.testing.assert_array_equal(out, x)


def _engine_modules():
    for info in pkgutil.iter_modules(audiosronnx.engines.__path__):
        yield importlib.import_module(f"audiosronnx.engines.{info.name}")


def test_every_engine_pins_a_commit_sha():
    hub = [m for m in _engine_modules() if hasattr(m, "_HF_REPO")]
    assert len(hub) >= 19
    for mod in hub:
        assert SHA.fullmatch(mod._HF_REVISION or ""), f"{mod.__name__}: {mod._HF_REVISION!r}"
