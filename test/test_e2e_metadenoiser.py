"""End-to-end Meta Denoiser + adversarial inputs.

The graphs take a fixed 10 s window because Demucs bakes its padding arithmetic into the
trace, so the weight-free tests cover the windowing arithmetic and the inference tests
exercise inputs both shorter and longer than one window.
"""
from __future__ import annotations

import numpy as np
import pytest

from audiosronnx.engines.metadenoiser import (
    _MODELS,
    _OVERLAP,
    _SR,
    _WINDOW,
    MetaDenoiserAdapter,
)


# --------------------------------------------------------------------------- #
# Configuration (no weights) — always runs.
# --------------------------------------------------------------------------- #
def test_default_model_is_dns64():
    assert MetaDenoiserAdapter()._model == "dns64"


def test_both_variants_selectable():
    assert set(_MODELS) == {"dns64", "dns48"}
    assert MetaDenoiserAdapter(model="dns48")._model == "dns48"


def test_unknown_model_rejected():
    with pytest.raises(ValueError):
        MetaDenoiserAdapter(model="dns96")


def test_window_is_ten_seconds():
    assert _WINDOW == _SR * 10


def test_overlap_is_smaller_than_the_window():
    """A stride of window - overlap must advance, or the slide would never terminate."""
    assert 0 < _OVERLAP < _WINDOW


def test_engine_is_labelled_non_commercial():
    """This is the only CC-BY-NC denoiser; the registry must say so."""
    from audiosronnx import get_engine

    assert "NC" in get_engine("metadenoiser").license.upper()


# --------------------------------------------------------------------------- #
# Inference (weights) — skipped if unavailable.
# --------------------------------------------------------------------------- #
def _denoiser(**kw):
    from audiosronnx import load_denoise

    try:
        eng = load_denoise("metadenoiser", **kw)
        eng._ensure_models()
        return eng
    except Exception as exc:
        pytest.skip(f"metadenoiser weights unavailable: {exc}")


def _snr(x, clean):
    n = min(x.size, clean.size)
    return 10 * np.log10((clean[:n] ** 2).sum() / (((x[:n] - clean[:n]) ** 2).sum() + 1e-20))


def test_metadenoiser_denoises_real_speech(speech16k_path):
    import soundfile as sf

    clean, sr = sf.read(speech16k_path, dtype="float32")
    noisy = (clean + 0.05 * np.random.RandomState(42).randn(clean.size)).astype(np.float32)
    out, rate = _denoiser().denoise(noisy, sr)
    assert rate == _SR
    assert _snr(out, clean) > _snr(noisy, clean) + 3.0    # measured ~+6.9 dB


def test_dns48_also_denoises(speech16k_path):
    import soundfile as sf

    clean, sr = sf.read(speech16k_path, dtype="float32")
    noisy = (clean + 0.05 * np.random.RandomState(42).randn(clean.size)).astype(np.float32)
    out, _ = _denoiser(model="dns48").denoise(noisy, sr)
    assert _snr(out, clean) > _snr(noisy, clean) + 3.0


def test_metadenoiser_preserves_length_below_one_window(speech16k_path):
    import soundfile as sf

    clean, _ = sf.read(speech16k_path, dtype="float32")
    out, _ = _denoiser().denoise(clean, _SR)
    assert out.size == clean.size


def test_metadenoiser_handles_input_longer_than_one_window():
    """Crosses the fixed-window boundary, exercising the crossfaded slide."""
    eng = _denoiser()
    n = _WINDOW + _SR * 3          # 13 s: two windows
    t = np.arange(n, dtype=np.float32) / _SR
    x = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    out, rate = eng.denoise(x, _SR)
    assert out.size == n and rate == _SR
    assert np.isfinite(out).all()


def test_metadenoiser_seam_is_not_discontinuous():
    """The crossfade should leave no step at the window boundary."""
    eng = _denoiser()
    n = _WINDOW + _SR * 3
    t = np.arange(n, dtype=np.float32) / _SR
    x = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    out, _ = eng.denoise(x, _SR)
    stride = _WINDOW - _OVERLAP
    jump = float(np.abs(np.diff(out[stride - 5:stride + 5])).max())
    assert jump < 0.5           # a hard seam would show a step near the signal amplitude


def test_metadenoiser_resamples_foreign_rate():
    t = np.arange(48000, dtype=np.float32) / 48000.0
    out, rate = _denoiser().denoise((0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 48000)
    assert rate == _SR
    assert np.isfinite(out).all()


def test_metadenoiser_is_deterministic():
    eng = _denoiser()
    x = (0.1 * np.random.RandomState(2).randn(16000)).astype(np.float32)
    assert np.array_equal(eng.denoise(x, _SR)[0], eng.denoise(x, _SR)[0])


# --- adversarial --- #
def test_metadenoiser_empty_input():
    out, _ = _denoiser().denoise(np.zeros(0, dtype=np.float32), _SR)
    assert out.size == 0


def test_metadenoiser_single_sample():
    out, _ = _denoiser().denoise(np.array([0.5], dtype=np.float32), _SR)
    assert out.size == 1
    assert np.isfinite(out).all()


def test_metadenoiser_silence_stays_finite():
    out, _ = _denoiser().denoise(np.zeros(16000, dtype=np.float32), _SR)
    assert out.size == 16000
    assert np.isfinite(out).all()


def test_metadenoiser_non_finite_input_does_not_propagate():
    x = np.array([0.1, np.nan, np.inf, -np.inf] * 4000, dtype=np.float32)
    out, _ = _denoiser().denoise(x, _SR)
    assert np.isfinite(out).all()


def test_metadenoiser_extremely_loud_input():
    x = (np.random.RandomState(1).randn(16000) * 1e4).astype(np.float32)
    out, _ = _denoiser().denoise(x, _SR)
    assert np.isfinite(out).all()
