"""Engine registry behaviour (no weights required)."""
from __future__ import annotations

import numpy as np
import pytest

from audiosronnx import (
    ENGINE_REGISTRY,
    EngineEntry,
    SRModel,
    available_models,
    get_engine,
    register_engine,
)


def test_builtin_engines_registered():
    models = available_models()
    for name in ("lavasr", "novasr", "hifiganbwe", "apbwe", "sidon"):
        assert name in models


def test_all_engines_output_48k():
    for name in available_models():
        assert get_engine(name).output_sample_rate == 48000


def test_get_engine_returns_entry():
    entry = get_engine("lavasr")
    assert isinstance(entry, EngineEntry)
    assert entry.alias == "lavasr"
    assert entry.output_sample_rate == 48000
    assert entry.license == "Apache-2.0"


def test_get_unknown_engine_raises():
    with pytest.raises(KeyError):
        get_engine("does-not-exist")


def test_default_engine_is_lavasr():
    from audiosronnx.api import DEFAULT_ENGINE

    assert DEFAULT_ENGINE == "lavasr"


def test_register_custom_engine():
    class _Dummy(SRModel):
        def _upscale_array(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
            return audio

    entry = EngineEntry(alias="_dummy_test", adapter_class=_Dummy)
    try:
        register_engine(entry)
        assert "_dummy_test" in available_models()
        assert get_engine("_dummy_test").adapter_class is _Dummy
    finally:
        ENGINE_REGISTRY.pop("_dummy_test", None)
