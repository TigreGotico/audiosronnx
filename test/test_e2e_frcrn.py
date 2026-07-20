"""End-to-end FRCRN denoising + adversarial inputs.

The weight-free tests cover the two-stage RMS normalisation and the padding rule, both of
which live outside the graph and measurably change the output if wrong. Inference tests
are skipped when the ONNX weights are unavailable.
"""
from __future__ import annotations

import numpy as np
import pytest

from audiosronnx.engines.frcrn import _SR, _audio_norm, _pad_for_decode


# --------------------------------------------------------------------------- #
# Normalisation / padding (no weights) — always runs.
# --------------------------------------------------------------------------- #
def test_audio_norm_hits_the_target_level():
    """The second stage measures RMS over above-average-power samples only."""
    rng = np.random.RandomState(0)
    x = (rng.randn(16000) * 0.01).astype(np.float32)
    normed, scalar = _audio_norm(x)
    power = normed ** 2
    loud = power[power > power.mean()]
    assert np.isclose(loud.mean() ** 0.5, 10 ** (-25 / 20), rtol=1e-3)
    assert scalar > 0


def test_audio_norm_scalar_restores_original_scale():
    rng = np.random.RandomState(1)
    x = (rng.randn(16000) * 0.2).astype(np.float32)
    normed, scalar = _audio_norm(x)
    assert np.allclose(normed * scalar, x, atol=1e-4)


def test_audio_norm_on_silence_is_finite():
    """An all-zero clip must not divide by zero — EPS carries it."""
    normed, scalar = _audio_norm(np.zeros(1000, dtype=np.float32))
    assert np.isfinite(normed).all() and np.isfinite(scalar)


def test_pad_short_input_reaches_one_window():
    out = _pad_for_decode(np.zeros(1000), _SR)
    assert out.size == _SR


def test_pad_between_window_and_window_plus_stride():
    window, stride = _SR, int(_SR * 0.75)
    out = _pad_for_decode(np.zeros(window + 100), _SR)
    assert out.size == window + stride


def test_pad_long_input_matches_upstream_formula():
    """2 s at 16 kHz pads to 52000 — the value upstream's rule produces."""
    assert _pad_for_decode(np.zeros(32000), _SR).size == 52000


def test_pad_leaves_aligned_input_untouched():
    window, stride = _SR, int(_SR * 0.75)
    n = window + stride * 3
    assert _pad_for_decode(np.zeros(n), _SR).size == n


# --------------------------------------------------------------------------- #
# Inference (weights) — skipped if unavailable.
# --------------------------------------------------------------------------- #
def _denoiser(**kw):
    from audiosronnx import load_denoise

    try:
        eng = load_denoise("frcrn", **kw)
        eng._ensure_models()
        return eng
    except Exception as exc:
        pytest.skip(f"frcrn weights unavailable: {exc}")


def test_frcrn_denoises_real_speech(speech16k_path):
    import soundfile as sf

    clean, sr = sf.read(speech16k_path, dtype="float32")
    noisy = (clean + 0.05 * np.random.RandomState(0).randn(clean.size)).astype(np.float32)
    out, rate = _denoiser().denoise(noisy, sr)
    assert rate == _SR

    def snr(x):
        n = min(x.size, clean.size)
        return 10 * np.log10((clean[:n] ** 2).sum() / (((x[:n] - clean[:n]) ** 2).sum() + 1e-20))

    assert snr(out) > snr(noisy) + 3.0   # measured ~+9 dB


def test_frcrn_preserves_length(speech16k_path):
    import soundfile as sf

    clean, _ = sf.read(speech16k_path, dtype="float32")
    out, _ = _denoiser().denoise(clean, _SR)
    assert out.size == clean.size


def test_frcrn_resamples_foreign_rate():
    t = np.arange(48000, dtype=np.float32) / 48000.0
    out, rate = _denoiser().denoise((0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 48000)
    assert rate == _SR
    assert np.isfinite(out).all()


# --- adversarial --- #
def test_frcrn_empty_input():
    out, _ = _denoiser().denoise(np.zeros(0, dtype=np.float32), _SR)
    assert out.size == 0


def test_frcrn_single_sample():
    out, _ = _denoiser().denoise(np.array([0.5], dtype=np.float32), _SR)
    assert out.size == 1
    assert np.isfinite(out).all()


def test_frcrn_all_zero_input_short_circuits():
    """Silence has no RMS to normalise against; it must return silence, not NaN."""
    out, _ = _denoiser().denoise(np.zeros(16000, dtype=np.float32), _SR)
    assert out.size == 16000
    assert np.isfinite(out).all()
    assert float(np.max(np.abs(out))) == 0.0


def test_frcrn_non_finite_input_does_not_propagate():
    x = np.array([0.1, np.nan, np.inf, -np.inf] * 4000, dtype=np.float32)
    out, _ = _denoiser().denoise(x, _SR)
    assert np.isfinite(out).all()


def test_frcrn_extremely_loud_input():
    x = (np.random.RandomState(1).randn(16000) * 1e4).astype(np.float32)
    out, _ = _denoiser().denoise(x, _SR)
    assert np.isfinite(out).all()


def test_frcrn_is_deterministic():
    eng = _denoiser()
    x = (0.1 * np.random.RandomState(2).randn(16000)).astype(np.float32)
    assert np.array_equal(eng.denoise(x, _SR)[0], eng.denoise(x, _SR)[0])
