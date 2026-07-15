"""Audio helpers (no weights required)."""
from __future__ import annotations

import os

import numpy as np

from audiosronnx.audio import read_wav, resample, to_float32_mono, write_wav


def test_to_float32_mono_from_int16():
    arr = np.array([0, 16384, -16384, 32767], dtype=np.int16)
    out = to_float32_mono(arr)
    assert out.dtype == np.float32
    assert -1.0 <= out.min() and out.max() <= 1.0


def test_to_float32_mono_downmix_stereo():
    stereo = np.zeros((100, 2), dtype=np.float32)
    stereo[:, 0] = 1.0
    out = to_float32_mono(stereo)
    assert out.shape == (100,)
    assert np.allclose(out, 0.5)


def test_to_float32_mono_from_bytes():
    pcm = np.array([0, 1000, -1000], dtype="<i2").tobytes()
    out = to_float32_mono(pcm)
    assert out.shape == (3,)
    assert out.dtype == np.float32


def test_resample_changes_length():
    x = np.random.randn(16000).astype(np.float32)
    up = resample(x, 16000, 48000)
    assert abs(len(up) - 48000) <= 8
    assert up.dtype == np.float32


def test_resample_noop():
    x = np.ones(10, dtype=np.float32)
    assert resample(x, 16000, 16000) is x or np.array_equal(resample(x, 16000, 16000), x)


def test_wav_roundtrip(tmp_path):
    x = (0.2 * np.sin(np.linspace(0, 50, 8000))).astype(np.float32)
    p = os.path.join(tmp_path, "t.wav")
    write_wav(p, x, 16000)
    back, sr = read_wav(p)
    assert sr == 16000
    assert back.shape[0] == x.shape[0]
    assert np.max(np.abs(back - x)) < 1e-3


def test_read_fixture(speech16k_path):
    audio, sr = read_wav(speech16k_path)
    assert sr == 16000
    assert audio.ndim == 1
    assert audio.size > 0
