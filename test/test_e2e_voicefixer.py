"""End-to-end VoiceFixer restoration + adversarial inputs.

VoiceFixer is generative and works over a fixed 5 s window, so the weight-free tests cover
the window arithmetic and the log inverse, and the inference tests check that it produces
sane restored audio rather than matching a reference waveform.
"""
from __future__ import annotations

import numpy as np
import pytest

from audiosronnx.engines.voicefixer import (
    _FRAMES,
    _HOP,
    _LOG_CLIP,
    _N_MELS,
    _OVERLAP_FRAMES,
    _SR,
    _WINDOW_SAMPLES,
)


# --------------------------------------------------------------------------- #
# Conventions (no weights) — always runs.
# --------------------------------------------------------------------------- #
def test_window_arithmetic_is_consistent():
    assert _WINDOW_SAMPLES == _FRAMES * _HOP
    assert 4.5 < _WINDOW_SAMPLES / _SR < 5.5      # about five seconds


def test_overlap_leaves_a_positive_stride():
    assert 0 < _OVERLAP_FRAMES < _FRAMES


def test_log_inverse_matches_upstream():
    """Upstream's from_log is 10 ** clip(x, max=5)."""
    x = np.array([-2.0, 0.0, 3.0, 7.0])
    assert np.allclose(np.power(10.0, np.clip(x, None, _LOG_CLIP)),
                       [1e-2, 1.0, 1e3, 1e5])


def test_log_inverse_clips_runaway_values():
    """Without the clip, a large prediction would overflow to inf."""
    assert np.isfinite(np.power(10.0, np.clip(np.array([400.0]), None, _LOG_CLIP))).all()


def test_registered_as_enhance_not_denoise():
    """It restores several defects at once, so it is an enhance engine."""
    from audiosronnx import get_engine

    entry = get_engine("voicefixer")
    assert entry.kind == "enhance"
    assert entry.input_sample_rate == entry.output_sample_rate == _SR


def test_reachable_through_load_denoise():
    """load_denoise accepts enhance engines as well as denoise ones."""
    from audiosronnx import available_denoisers

    assert "voicefixer" in available_denoisers()


def test_mel_dimension_is_128():
    assert _N_MELS == 128


# --------------------------------------------------------------------------- #
# Inference (weights) — skipped if unavailable.
# --------------------------------------------------------------------------- #
def _engine(**kw):
    from audiosronnx import load_denoise

    try:
        eng = load_denoise("voicefixer", **kw)
        eng._ensure_models()
        return eng
    except Exception as exc:
        pytest.skip(f"voicefixer weights unavailable: {exc}")


def test_voicefixer_restores_real_speech(speech16k_path):
    """Narrowband, noisy input should come back as finite 44.1 kHz audio with content."""
    import soundfile as sf

    clean, sr = sf.read(speech16k_path, dtype="float32")
    noisy = (clean + 0.05 * np.random.RandomState(42).randn(clean.size)).astype(np.float32)
    out, rate = _engine().denoise(noisy, sr)
    assert rate == _SR
    assert np.isfinite(out).all()
    assert float(np.max(np.abs(out))) > 1e-3


def test_voicefixer_preserves_length(speech16k_path):
    import soundfile as sf

    clean, sr = sf.read(speech16k_path, dtype="float32")
    out, _ = _engine().denoise(clean, sr)
    assert out.size == int(round(_SR / sr * clean.size))


def test_voicefixer_crosses_the_window_boundary():
    """Longer than one 5 s window, exercising the crossfaded slide."""
    eng = _engine()
    n = _WINDOW_SAMPLES + _SR * 2
    t = np.arange(n, dtype=np.float32) / _SR
    x = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    out, rate = eng.denoise(x, _SR)
    assert out.size == n and rate == _SR
    assert np.isfinite(out).all()


def test_voicefixer_is_deterministic():
    eng = _engine()
    x = (0.1 * np.random.RandomState(2).randn(_SR)).astype(np.float32)
    assert np.array_equal(eng.denoise(x, _SR)[0], eng.denoise(x, _SR)[0])


# --- adversarial --- #
def test_voicefixer_empty_input():
    out, _ = _engine().denoise(np.zeros(0, dtype=np.float32), _SR)
    assert out.size == 0


def test_voicefixer_single_sample():
    out, _ = _engine().denoise(np.array([0.5], dtype=np.float32), _SR)
    assert out.size == 1
    assert np.isfinite(out).all()


def test_voicefixer_silence_stays_finite():
    out, _ = _engine().denoise(np.zeros(_SR, dtype=np.float32), _SR)
    assert out.size == _SR
    assert np.isfinite(out).all()


def test_voicefixer_non_finite_input_does_not_propagate():
    x = np.array([0.1, np.nan, np.inf, -np.inf] * 4000, dtype=np.float32)
    out, _ = _engine().denoise(x, _SR)
    assert np.isfinite(out).all()


def test_voicefixer_extremely_loud_input():
    x = (np.random.RandomState(1).randn(_SR) * 1e4).astype(np.float32)
    out, _ = _engine().denoise(x, _SR)
    assert np.isfinite(out).all()
