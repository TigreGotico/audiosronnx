"""End-to-end FLowHigh bandwidth extension + adversarial inputs.

FLowHigh is generative: it integrates a velocity field from a noisy start, so the
weight-free tests pin the mel front-end conventions and the cutoff search, and the
inference tests check that it *adds high-frequency content* rather than that it matches a
reference waveform.
"""
from __future__ import annotations

import numpy as np
import pytest

from audiosronnx.engines.flowhigh import (
    _N_FFT,
    _N_MELS,
    _cutoff_bin,
    FLowHighAdapter,
    log_mel,
)


# --------------------------------------------------------------------------- #
# Front-end and cutoff (no weights) — always runs.
# --------------------------------------------------------------------------- #
def test_log_mel_shape_and_frame_count():
    """Uncentred STFT after a (n_fft - hop)/2 reflect pad, the HiFi-GAN convention."""
    audio = np.zeros(48000, dtype=np.float32)
    mel = log_mel(audio)
    assert mel.shape[0] == 1 and mel.shape[2] == _N_MELS
    assert mel.shape[1] == 1 + (48000 + (_N_FFT - 480) - _N_FFT) // 480


def test_log_mel_silence_sits_at_the_clamp_floor():
    mel = log_mel(np.zeros(24000, dtype=np.float32))
    assert np.isfinite(mel).all()
    assert np.allclose(mel, np.log(1e-5), atol=1e-6)


def test_log_mel_is_finite_on_signal():
    t = np.arange(48000, dtype=np.float32) / 48000.0
    assert np.isfinite(log_mel(0.3 * np.sin(2 * np.pi * 220 * t))).all()


def test_cutoff_bin_finds_a_band_limited_edge():
    """A spectrum with energy only in the low bins must cut off inside them."""
    spec = np.zeros((100, 10), dtype=np.complex128)
    spec[:40] = 1.0
    cutoff = _cutoff_bin(spec)
    assert 0 < cutoff <= 40


def test_cutoff_bin_on_silence_is_zero():
    assert _cutoff_bin(np.zeros((50, 5), dtype=np.complex128)) == 0


def test_cutoff_bin_on_fullband_reaches_the_top():
    spec = np.ones((100, 10), dtype=np.complex128)
    assert _cutoff_bin(spec) >= 90


def test_invalid_step_count_rejected():
    with pytest.raises(ValueError):
        FLowHighAdapter(steps=0)


def test_seed_defaults_to_deterministic():
    assert FLowHighAdapter()._seed == 0


# --------------------------------------------------------------------------- #
# Inference (weights) — skipped if unavailable.
# --------------------------------------------------------------------------- #
def _engine(**kw):
    from audiosronnx import load_sr

    try:
        eng = load_sr("flowhigh", **kw)
        eng._ensure_models()
        return eng
    except Exception as exc:
        pytest.skip(f"flowhigh weights unavailable: {exc}")


def _band_energy(x, rate, lo, hi):
    spectrum = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(x.size, 1 / rate)
    return float((spectrum[(freqs >= lo) & (freqs < hi)] ** 2).sum())


def test_flowhigh_adds_high_band_content(speech16k_path):
    """The point of the engine: energy above the source band that was not there before."""
    import soundfile as sf

    narrow, rate = sf.read(speech16k_path, dtype="float32")
    out, out_rate = _engine().upscale(narrow, rate)
    assert out_rate == 48000

    naive = np.interp(np.linspace(0, narrow.size - 1, out.size),
                      np.arange(narrow.size), narrow).astype(np.float32)
    assert _band_energy(out, 48000, 8000, 16000) > 2 * _band_energy(naive, 48000, 8000, 16000)


def test_flowhigh_output_length_tracks_input(speech16k_path):
    import soundfile as sf

    narrow, rate = sf.read(speech16k_path, dtype="float32")
    out, _ = _engine().upscale(narrow, rate)
    assert out.size == int(round(48000 / rate * narrow.size))


def test_flowhigh_is_deterministic_by_default():
    """Upstream draws fresh noise each run; the seeded default must reproduce."""
    eng = _engine()
    t = np.arange(16000, dtype=np.float32) / 16000.0
    x = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    assert np.array_equal(eng.upscale(x, 16000)[0], eng.upscale(x, 16000)[0])


def test_flowhigh_step_count_changes_the_result():
    eng4, eng8 = _engine(steps=4), _engine(steps=8)
    t = np.arange(16000, dtype=np.float32) / 16000.0
    x = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    assert not np.allclose(eng4.upscale(x, 16000)[0], eng8.upscale(x, 16000)[0])


def test_flowhigh_merge_preserves_more_of_the_source():
    """With the merge on, the low band should stay closer to the input."""
    eng_on, eng_off = _engine(merge=True), _engine(merge=False)
    t = np.arange(16000, dtype=np.float32) / 16000.0
    x = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    on, _ = eng_on.upscale(x, 16000)
    off, _ = eng_off.upscale(x, 16000)
    assert not np.allclose(on, off)
    assert np.isfinite(on).all() and np.isfinite(off).all()


# --- adversarial --- #
def test_flowhigh_empty_input():
    out, _ = _engine().upscale(np.zeros(0, dtype=np.float32), 16000)
    assert out.size == 0


def test_flowhigh_silence_returns_finite_zeros():
    out, rate = _engine().upscale(np.zeros(16000, dtype=np.float32), 16000)
    assert rate == 48000 and out.size == 48000
    assert np.isfinite(out).all()
    assert float(np.max(np.abs(out))) == 0.0


def test_flowhigh_single_sample():
    out, _ = _engine().upscale(np.array([0.5], dtype=np.float32), 16000)
    assert np.isfinite(out).all()


def test_flowhigh_non_finite_input_does_not_propagate():
    x = np.array([0.1, np.nan, np.inf, -np.inf] * 4000, dtype=np.float32)
    out, _ = _engine().upscale(x, 16000)
    assert np.isfinite(out).all()


def test_flowhigh_extremely_loud_input():
    x = (np.random.RandomState(1).randn(16000) * 1e4).astype(np.float32)
    out, _ = _engine().upscale(x, 16000)
    assert np.isfinite(out).all()
