"""SRModel base dispatch logic, exercised with a synthetic adapter (no weights)."""
from __future__ import annotations

import os

import numpy as np
import pytest

from audiosronnx.audio import read_wav, resample
from audiosronnx.base import SRModel


class DummySR(SRModel):
    """Trivial adapter: linear-resample the input to 48 kHz. No ONNX involved."""

    output_sample_rate = 48000
    input_sample_rate = 0

    def _upscale_array(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        return resample(audio, sample_rate, 48000)


def test_upscale_array_returns_48k():
    m = DummySR()
    x = np.random.randn(16000).astype(np.float32)
    out, sr = m.upscale(x, 16000)
    assert sr == 48000
    assert out.dtype == np.float32
    assert abs(len(out) - 48000) <= 8


def test_upscale_requires_sample_rate_for_array():
    m = DummySR()
    with pytest.raises(ValueError):
        m.upscale(np.zeros(10, dtype=np.float32))


def test_upscale_empty_audio():
    m = DummySR()
    out, sr = m.upscale(np.zeros(0, dtype=np.float32), 16000)
    assert sr == 48000
    assert out.size == 0


def test_upscale_from_path(speech16k_path):
    m = DummySR()
    out, sr = m.upscale(speech16k_path)
    assert sr == 48000
    assert out.size > 0


def test_upscale_file_writes_48k(tmp_path, speech16k_path):
    m = DummySR()
    out_path = os.path.join(tmp_path, "out.wav")
    m.upscale_file(speech16k_path, out_path)
    audio, sr = read_wav(out_path)
    assert sr == 48000
    assert audio.size > 0


def test_upscale_dir(tmp_path, speech16k_path, speech8k_path):
    in_dir = os.path.join(tmp_path, "in")
    out_dir = os.path.join(tmp_path, "out")
    os.makedirs(in_dir)
    for src in (speech16k_path, speech8k_path):
        with open(src, "rb") as f, open(os.path.join(in_dir, os.path.basename(src)), "wb") as g:
            g.write(f.read())
    m = DummySR()
    written = m.upscale_dir(in_dir, out_dir)
    assert len(written) == 2
    for p in written:
        _, sr = read_wav(p)
        assert sr == 48000


def test_sample_rate_property():
    assert DummySR().sample_rate == 48000
