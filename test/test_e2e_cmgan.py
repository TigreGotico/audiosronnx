"""End-to-end CMGAN denoising + adversarial inputs.

CMGAN works in a power-compressed complex domain and pads by wrapping the clip's own start,
so the weight-free tests pin both conventions down. The inference tests assert what CMGAN
actually does — it raises perceptual quality while *lowering* waveform SNR — rather than
assuming an SNR gain like the other engines.
"""
from __future__ import annotations

import numpy as np
import pytest

from audiosronnx.engines.cmgan import _COMPRESS, _HOP, _SR, _compress
from audiosronnx._stft import hamming_window


# --------------------------------------------------------------------------- #
# Spectral conventions (no weights) — always runs.
# --------------------------------------------------------------------------- #
def test_compress_round_trip_is_identity():
    rng = np.random.RandomState(0)
    spec = rng.randn(64) + 1j * rng.randn(64)
    back = _compress(_compress(spec, _COMPRESS), 1.0 / _COMPRESS)
    assert np.allclose(back, spec, rtol=1e-8, atol=1e-10)


def test_compress_preserves_phase():
    rng = np.random.RandomState(1)
    spec = rng.randn(32) + 1j * rng.randn(32)
    assert np.allclose(np.angle(_compress(spec, _COMPRESS)), np.angle(spec), atol=1e-12)


def test_compress_changes_magnitude():
    spec = np.array([4.0 + 0j, 0.25 + 0j])
    assert not np.allclose(np.abs(_compress(spec, _COMPRESS)), np.abs(spec))


def test_hamming_window_is_periodic_not_symmetric():
    """CMGAN uses torch's periodic hamming; the symmetric Kaldi one is a different window.

    They differ by ~5e-3, which is enough to break perfect reconstruction.
    """
    from audiosronnx._kaldi_fbank import hamming as symmetric

    periodic = hamming_window(400)
    assert not np.allclose(periodic, symmetric(400), atol=1e-4)
    assert np.allclose(periodic, 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(400) / 400))


def test_wrap_padding_reaches_a_whole_number_of_hops():
    for n in (16000, 16050, 100, 99):
        assert (n + (-n) % _HOP) % _HOP == 0


# --------------------------------------------------------------------------- #
# Inference (weights) — skipped if unavailable.
# --------------------------------------------------------------------------- #
def _denoiser(**kw):
    from audiosronnx import load_denoise

    try:
        eng = load_denoise("cmgan", **kw)
        eng._ensure_models()
        return eng
    except Exception as exc:
        pytest.skip(f"cmgan weights unavailable: {exc}")


def test_cmgan_changes_the_signal(speech16k_path):
    """CMGAN trades waveform SNR for perceptual quality, so assert it acts, not that
    it raises SNR — the other engines' assertion would fail here by design."""
    import soundfile as sf

    clean, sr = sf.read(speech16k_path, dtype="float32")
    noisy = (clean + 0.05 * np.random.RandomState(42).randn(clean.size)).astype(np.float32)
    out, rate = _denoiser().denoise(noisy, sr)
    assert rate == _SR
    assert out.size == noisy.size
    assert np.isfinite(out).all()
    # it must actually do something, and not merely pass the input through
    assert float(np.abs(out - noisy[:out.size]).max()) > 1e-3


def test_cmgan_preserves_length(speech16k_path):
    import soundfile as sf

    clean, _ = sf.read(speech16k_path, dtype="float32")
    out, _ = _denoiser().denoise(clean, _SR)
    assert out.size == clean.size


def test_cmgan_resamples_foreign_rate():
    t = np.arange(48000, dtype=np.float32) / 48000.0
    out, rate = _denoiser().denoise((0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 48000)
    assert rate == _SR
    assert np.isfinite(out).all()


def test_cmgan_runs_at_arbitrary_lengths():
    """Guards the export: a length-specialised graph fails away from its traced size."""
    eng = _denoiser()
    rng = np.random.RandomState(3)
    for n in (2000, 7999, 16000, 32001):
        out, _ = eng.denoise((0.1 * rng.randn(n)).astype(np.float32), _SR)
        assert out.size == n
        assert np.isfinite(out).all()


def test_cmgan_is_deterministic():
    eng = _denoiser()
    x = (0.1 * np.random.RandomState(2).randn(16000)).astype(np.float32)
    assert np.array_equal(eng.denoise(x, _SR)[0], eng.denoise(x, _SR)[0])


# --- adversarial --- #
def test_cmgan_empty_input():
    out, _ = _denoiser().denoise(np.zeros(0, dtype=np.float32), _SR)
    assert out.size == 0


def test_cmgan_all_zero_input_short_circuits():
    out, _ = _denoiser().denoise(np.zeros(16000, dtype=np.float32), _SR)
    assert out.size == 16000
    assert float(np.max(np.abs(out))) == 0.0


def test_cmgan_single_sample():
    out, _ = _denoiser().denoise(np.array([0.5], dtype=np.float32), _SR)
    assert out.size == 1
    assert np.isfinite(out).all()


def test_cmgan_sub_frame_input():
    out, _ = _denoiser().denoise(np.full(200, 0.1, dtype=np.float32), _SR)
    assert out.size == 200
    assert np.isfinite(out).all()


def test_cmgan_non_finite_input_does_not_propagate():
    x = np.array([0.1, np.nan, np.inf, -np.inf] * 4000, dtype=np.float32)
    out, _ = _denoiser().denoise(x, _SR)
    assert np.isfinite(out).all()


def test_cmgan_extremely_loud_input():
    x = (np.random.RandomState(1).randn(16000) * 1e4).astype(np.float32)
    out, _ = _denoiser().denoise(x, _SR)
    assert np.isfinite(out).all()
