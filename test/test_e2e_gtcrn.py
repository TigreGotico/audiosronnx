"""End-to-end GTCRN denoising + adversarial inputs.

The weight-free tests cover the sqrt-Hann window, which must match
``torch.hann_window(512).pow(0.5)`` for the graph to see the features it was trained on.
Inference tests are skipped when the ONNX weights are unavailable.
"""
from __future__ import annotations

import numpy as np
import pytest

from audiosronnx.engines.gtcrn import _N_FFT, _SR, _sqrt_hann


# --------------------------------------------------------------------------- #
# Window (no weights) — always runs.
# --------------------------------------------------------------------------- #
def test_sqrt_hann_matches_torch_definition():
    """``torch.hann_window(n)`` is periodic; the model uses its square root."""
    n = _N_FFT
    k = np.arange(n)
    periodic_hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * k / n))
    assert np.allclose(_sqrt_hann(n), np.sqrt(periodic_hann), atol=1e-12)


def test_sqrt_hann_is_power_complementary_at_50pc_overlap():
    """sqrt-Hann squared sums to 1 across a half-window shift — exact reconstruction."""
    w = _sqrt_hann(_N_FFT) ** 2
    hop = _N_FFT // 2
    assert np.allclose(w[:hop] + w[hop:], 1.0, atol=1e-12)


def test_sqrt_hann_is_non_negative_and_bounded():
    w = _sqrt_hann(_N_FFT)
    assert w.min() >= 0.0 and w.max() <= 1.0
    assert np.isfinite(w).all()


# --------------------------------------------------------------------------- #
# Inference (weights) — skipped if unavailable.
# --------------------------------------------------------------------------- #
def _denoiser(**kw):
    from audiosronnx import load_denoise

    try:
        eng = load_denoise("gtcrn", **kw)
        eng._ensure_models()
        return eng
    except Exception as exc:
        pytest.skip(f"gtcrn weights unavailable: {exc}")


def test_gtcrn_denoises_real_speech(speech16k_path):
    import soundfile as sf

    clean, sr = sf.read(speech16k_path, dtype="float32")
    assert sr == _SR
    noisy = (clean + 0.05 * np.random.RandomState(0).randn(clean.size)).astype(np.float32)
    out, rate = _denoiser().denoise(noisy, _SR)
    assert rate == _SR

    def snr(x):
        n = min(x.size, clean.size)
        return 10 * np.log10((clean[:n] ** 2).sum() / (((x[:n] - clean[:n]) ** 2).sum() + 1e-20))

    assert snr(out) > snr(noisy) + 2.0   # measured ~+7 dB; assert a wide margin


def test_gtcrn_preserves_length(speech16k_path):
    import soundfile as sf

    clean, _ = sf.read(speech16k_path, dtype="float32")
    out, rate = _denoiser().denoise(clean, _SR)
    assert out.size == clean.size
    assert rate == _SR


def test_gtcrn_resamples_foreign_rate():
    """48 kHz in is resampled to the model's 16 kHz, and comes back at 16 kHz."""
    t = np.arange(48000, dtype=np.float32) / 48000.0
    out, rate = _denoiser().denoise((0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 48000)
    assert rate == _SR
    assert np.isfinite(out).all()


def test_gtcrn_caches_start_zeroed_each_call():
    """Caches are rebuilt per call, so repeated input gives identical output."""
    eng = _denoiser()
    x = (0.1 * np.random.RandomState(2).randn(16000)).astype(np.float32)
    assert np.array_equal(eng.denoise(x, _SR)[0], eng.denoise(x, _SR)[0])


# --- adversarial: must not crash, must stay finite, must keep the length contract --- #
def test_gtcrn_empty_input():
    out, _ = _denoiser().denoise(np.zeros(0, dtype=np.float32), _SR)
    assert out.size == 0


def test_gtcrn_single_sample():
    out, _ = _denoiser().denoise(np.array([0.5], dtype=np.float32), _SR)
    assert out.size == 1
    assert np.isfinite(out).all()


def test_gtcrn_sub_frame_input():
    out, _ = _denoiser().denoise(np.full(100, 0.1, dtype=np.float32), _SR)
    assert out.size == 100
    assert np.isfinite(out).all()


def test_gtcrn_silence_stays_silent():
    out, _ = _denoiser().denoise(np.zeros(16000, dtype=np.float32), _SR)
    assert np.isfinite(out).all()
    assert float(np.max(np.abs(out))) < 1e-3


def test_gtcrn_extremely_loud_input():
    x = (np.random.RandomState(1).randn(16000) * 1e4).astype(np.float32)
    out, _ = _denoiser().denoise(x, _SR)
    assert np.isfinite(out).all()
