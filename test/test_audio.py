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


def _sine_wav_bytes(tmp_path, sr=22050):
    import soundfile as sf
    t = np.arange(sr // 10) / sr
    x = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = str(tmp_path / "sine.wav")
    sf.write(path, x, sr, subtype="PCM_16")
    with open(path, "rb") as f:
        return path, f.read()


def test_load_audio_wav_bytes_match_soundfile(tmp_path):
    import soundfile as sf
    from audiosronnx.audio import load_audio
    path, data = _sine_wav_bytes(tmp_path)
    expected, expected_sr = sf.read(path, dtype="float32")
    out, sr = load_audio(data)
    assert sr == expected_sr == 22050
    np.testing.assert_array_equal(out, expected)


def test_load_audio_wav_bytes_ignore_stated_rate(tmp_path):
    from audiosronnx.audio import load_audio
    _, data = _sine_wav_bytes(tmp_path)
    out, sr = load_audio(data, sample_rate=16000)
    assert sr == 22050
    assert out.shape == (2205,)


def test_load_audio_raw_pcm_with_rate():
    from audiosronnx.audio import load_audio
    pcm = np.array([0, 16384, -16384, 32767], dtype="<i2").tobytes()
    out, sr = load_audio(pcm, sample_rate=16000)
    assert sr == 16000
    np.testing.assert_array_equal(out, np.array([0, 0.5, -0.5, 32767 / 32768], dtype=np.float32))


def test_load_audio_raw_pcm_without_rate_raises():
    import pytest
    from audiosronnx.audio import load_audio
    pcm = np.array([0, 1000, -1000], dtype="<i2").tobytes()
    with pytest.raises(ValueError, match="RIFF/WAVE.*sample_rate="):
        load_audio(pcm)


def test_resample_surfaces_scipy_runtime_error(monkeypatch):
    import pytest
    import scipy.signal

    def boom(*a, **k):
        raise RuntimeError("scipy failed")

    monkeypatch.setattr(scipy.signal, "resample_poly", boom)
    with pytest.raises(RuntimeError, match="scipy failed"):
        resample(np.zeros(160, dtype=np.float32), 16000, 48000)
