"""End-to-end MossFormerGAN + adversarial inputs.

The graph takes a fixed 401-frame window because MossFormer's group attention captures the
traced length in a reshape, so the weight-free tests cover the windowing arithmetic and the
inference tests cross the window boundary.
"""
from __future__ import annotations

import numpy as np
import pytest

from audiosronnx.engines.mossformergan import (
    _COMPRESS,
    _FRAMES,
    _HOP,
    _OVERLAP_FRAMES,
    _SR,
    _compress,
)


# --------------------------------------------------------------------------- #
# Conventions (no weights) — always runs.
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


def test_window_stride_advances():
    """A stride of frames - overlap must be positive, or the slide would not terminate."""
    assert 0 < _OVERLAP_FRAMES < _FRAMES


def test_wrap_padding_reaches_a_whole_number_of_hops():
    for n in (16000, 16050, 99):
        assert (n + (-n) % _HOP) % _HOP == 0


def test_engine_is_registered_as_apache():
    from audiosronnx import get_engine

    assert get_engine("mossformergan").license == "Apache-2.0"


# --------------------------------------------------------------------------- #
# Inference (weights) — skipped if unavailable.
# --------------------------------------------------------------------------- #
def _denoiser(**kw):
    from audiosronnx import load_denoise

    try:
        eng = load_denoise("mossformergan", **kw)
        eng._ensure_models()
        return eng
    except Exception as exc:
        pytest.skip(f"mossformergan weights unavailable: {exc}")


def _snr(x, clean):
    n = min(x.size, clean.size)
    return 10 * np.log10((clean[:n] ** 2).sum() / (((x[:n] - clean[:n]) ** 2).sum() + 1e-20))


def test_mossformergan_denoises_real_speech(speech16k_path):
    import soundfile as sf

    clean, sr = sf.read(speech16k_path, dtype="float32")
    noisy = (clean + 0.05 * np.random.RandomState(42).randn(clean.size)).astype(np.float32)
    out, rate = _denoiser().denoise(noisy, sr)
    assert rate == _SR
    assert _snr(out, clean) > _snr(noisy, clean) + 2.0


def test_mossformergan_preserves_length(speech16k_path):
    import soundfile as sf

    clean, _ = sf.read(speech16k_path, dtype="float32")
    out, _ = _denoiser().denoise(clean, _SR)
    assert out.size == clean.size


def test_mossformergan_crosses_the_window_boundary():
    """Longer than one 401-frame window, exercising the crossfaded slide."""
    eng = _denoiser()
    n = _SR * 8                       # well past ~2.5 s per window
    t = np.arange(n, dtype=np.float32) / _SR
    x = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    out, rate = eng.denoise(x, _SR)
    assert out.size == n and rate == _SR
    assert np.isfinite(out).all()


def test_mossformergan_resamples_foreign_rate():
    t = np.arange(48000, dtype=np.float32) / 48000.0
    out, rate = _denoiser().denoise((0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 48000)
    assert rate == _SR
    assert np.isfinite(out).all()


def test_mossformergan_is_deterministic():
    eng = _denoiser()
    x = (0.1 * np.random.RandomState(2).randn(16000)).astype(np.float32)
    assert np.array_equal(eng.denoise(x, _SR)[0], eng.denoise(x, _SR)[0])


# --- adversarial --- #
def test_mossformergan_empty_input():
    out, _ = _denoiser().denoise(np.zeros(0, dtype=np.float32), _SR)
    assert out.size == 0


def test_mossformergan_all_zero_input_short_circuits():
    out, _ = _denoiser().denoise(np.zeros(16000, dtype=np.float32), _SR)
    assert out.size == 16000
    assert float(np.max(np.abs(out))) == 0.0


def test_mossformergan_single_sample():
    out, _ = _denoiser().denoise(np.array([0.5], dtype=np.float32), _SR)
    assert out.size == 1
    assert np.isfinite(out).all()


def test_mossformergan_sub_frame_input():
    out, _ = _denoiser().denoise(np.full(200, 0.1, dtype=np.float32), _SR)
    assert out.size == 200
    assert np.isfinite(out).all()


def test_mossformergan_non_finite_input_does_not_propagate():
    x = np.array([0.1, np.nan, np.inf, -np.inf] * 4000, dtype=np.float32)
    out, _ = _denoiser().denoise(x, _SR)
    assert np.isfinite(out).all()


def test_mossformergan_extremely_loud_input():
    x = (np.random.RandomState(1).randn(16000) * 1e4).astype(np.float32)
    out, _ = _denoiser().denoise(x, _SR)
    assert np.isfinite(out).all()
