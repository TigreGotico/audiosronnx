"""Shared pytest fixtures and helpers."""
from __future__ import annotations

import os

import numpy as np
import pytest

RESOURCES = os.path.join(os.path.dirname(__file__), "resources")
SPEECH_16K = os.path.join(RESOURCES, "speech16k.wav")
SPEECH_8K = os.path.join(RESOURCES, "speech8k.wav")


@pytest.fixture
def speech16k_path() -> str:
    return SPEECH_16K


@pytest.fixture
def speech8k_path() -> str:
    return SPEECH_8K


@pytest.fixture
def sine_16k() -> np.ndarray:
    """1 second of a 300 Hz sine at 16 kHz, mono float32."""
    t = np.arange(16000, dtype=np.float32) / 16000.0
    return (0.3 * np.sin(2 * np.pi * 300.0 * t)).astype(np.float32)


@pytest.fixture
def load_engine():
    """Return the :func:`load_engine_or_skip` helper for end-to-end tests."""
    return load_engine_or_skip


def load_engine_or_skip(engine: str, **kwargs):
    """Load a real engine, downloading weights; skip the test if that is not possible.

    This gates the end-to-end tests: on a machine without network access (or before
    the ONNX weights are published) the download raises and the test is skipped rather
    than failing. When weights are present the test runs real ONNX inference.
    """
    from audiosronnx import load_sr

    try:
        model = load_sr(engine, **kwargs)
        # force the (lazy) model download / session creation now
        model._ensure_models()
        return model
    except Exception as exc:  # network / missing-weights only
        import pytest

        pytest.skip(f"weights for engine {engine!r} unavailable: {exc}")
