"""Pure-numpy DSP helpers: kaiser resampling and torch-faithful STFT/ISTFT."""
from __future__ import annotations

import numpy as np

from audiosronnx._kaiser import kaiser_resample
from audiosronnx._stft import istft, stft


def test_kaiser_resample_length():
    x = np.random.randn(8000).astype(np.float32)
    y = kaiser_resample(x, 8000, 48000)
    assert abs(len(y) - 48000) <= 8
    assert y.dtype == np.float32


def test_kaiser_resample_noop():
    x = np.ones(100, dtype=np.float32)
    assert np.array_equal(kaiser_resample(x, 16000, 16000), x)


def test_kaiser_resample_preserves_tone():
    # a 1 kHz tone at 16 kHz, upsampled to 48 kHz, should stay a ~1 kHz tone
    t = np.arange(16000) / 16000.0
    x = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    y = kaiser_resample(x, 16000, 48000)
    spec = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1 / 48000)
    assert abs(freqs[int(np.argmax(spec))] - 1000) < 30


def test_stft_shape():
    x = np.random.randn(8000).astype(np.float32)
    spec = stft(x, 1024, 80, 320)
    assert spec.shape[0] == 1024 // 2 + 1


def test_stft_istft_roundtrip():
    x = (0.2 * np.sin(np.linspace(0, 200, 8000))).astype(np.float32)
    spec = stft(x, 1024, 80, 320)
    y = istft(spec, 1024, 80, 320, length=len(x))
    # ignore edges affected by centre padding
    a, b = 600, len(x) - 600
    assert np.max(np.abs(y[a:b] - x[a:b])) < 1e-3
