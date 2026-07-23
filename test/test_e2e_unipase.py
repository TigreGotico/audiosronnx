"""End-to-end UniPASE universal enhancement + adversarial inputs.

UniPASE is generative and runs over a fixed 8 s window, so the weight-free tests cover the
window arithmetic and the numpy Vocos ISTFT, and the inference tests check that it produces
sane enhanced audio rather than matching a reference waveform. Inference tests skip when the
ONNX weights are unavailable (no network / pre-publish).
"""
from __future__ import annotations

import numpy as np
import pytest

from audiosronnx.engines.unipase import (
    _HOP,
    _HOP_LEN,
    _N_FFT,
    _OLP_HALF,
    _SEG_LEN,
    _SR,
    _vocos_istft,
)


# --------------------------------------------------------------------------- #
# Conventions (no weights) — always runs.
# --------------------------------------------------------------------------- #
def test_window_arithmetic_is_consistent():
    assert _SEG_LEN == _SR * 8
    assert _HOP_LEN == _SR * 4
    assert _OLP_HALF == _SR * 2          # half the 4 s overlap, trimmed at each seam


def test_vocos_istft_length_tracks_frames():
    # Vocos "same" ISTFT: T frames -> T * hop samples.
    for t in (10, 50, 400):
        spec = np.zeros((1, _N_FFT + 2, t), dtype=np.float32)
        out = _vocos_istft(spec)
        assert out.shape == (t * _HOP,)
        assert out.dtype == np.float32


def test_vocos_istft_reconstructs_a_tone():
    """A constant magnitude + linear phase ramp must invert to finite, non-trivial audio."""
    t = 64
    rng = np.random.RandomState(0)
    spec = np.zeros((1, _N_FFT + 2, t), dtype=np.float32)
    spec[0, : _N_FFT // 2 + 1] = -1.0                      # log-magnitude
    spec[0, _N_FFT // 2 + 1:] = rng.uniform(-np.pi, np.pi, (_N_FFT // 2 + 1, t))
    out = _vocos_istft(spec)
    assert np.isfinite(out).all()
    assert float(np.max(np.abs(out))) > 0.0


def test_vocos_istft_magnitude_is_clamped():
    """A runaway log-magnitude is clipped (exp(5)), so the output stays finite."""
    spec = np.full((1, _N_FFT + 2, 8), 500.0, dtype=np.float32)
    assert np.isfinite(_vocos_istft(spec)).all()


def test_registered_as_enhance_not_denoise():
    """It removes noise and reverberation at once, so it is an enhance engine."""
    from audiosronnx import get_engine

    entry = get_engine("unipase")
    assert entry.kind == "enhance"
    assert entry.input_sample_rate == entry.output_sample_rate == _SR


def test_precision_selects_variant_filenames():
    """fp32 (default) and fp16 map to the published HF filenames; int8 is rejected."""
    from audiosronnx.engines.unipase import UniPASEAdapter, _FILES

    assert set(_FILES) == {"fp32", "fp16"}
    assert _FILES["fp32"] == {"enc": "encoder_adapter.onnx", "voc": "vocoder.onnx"}
    assert _FILES["fp16"] == {"enc": "encoder_adapter.fp16.onnx", "voc": "vocoder.fp16.onnx"}
    assert UniPASEAdapter()._precision == "fp32"                       # fp32 by default
    assert UniPASEAdapter(precision="fp16")._precision == "fp16"


def test_unknown_precision_rejected():
    import pytest

    from audiosronnx.engines.unipase import UniPASEAdapter

    with pytest.raises(ValueError):
        UniPASEAdapter(precision="int8")


def test_reachable_through_load_denoise():
    from audiosronnx import available_denoisers

    assert "unipase" in available_denoisers()


# --------------------------------------------------------------------------- #
# Inference (weights) — skipped if unavailable.
# --------------------------------------------------------------------------- #
def _engine(**kw):
    from audiosronnx import load_denoise

    try:
        eng = load_denoise("unipase", **kw)
        eng._ensure_models()
        return eng
    except Exception as exc:
        pytest.skip(f"unipase weights unavailable: {exc}")


def test_unipase_enhances_real_speech(speech16k_path):
    import soundfile as sf

    clean, sr = sf.read(speech16k_path, dtype="float32")
    noisy = (clean + 0.05 * np.random.RandomState(42).randn(clean.size)).astype(np.float32)
    out, rate = _engine().denoise(noisy, sr)
    assert rate == _SR
    assert np.isfinite(out).all()
    assert float(np.max(np.abs(out))) > 1e-3


def test_unipase_preserves_length(speech16k_path):
    import soundfile as sf

    clean, sr = sf.read(speech16k_path, dtype="float32")
    out, _ = _engine().denoise(clean, sr)
    assert out.size == int(round(_SR / sr * clean.size))


def test_unipase_fp16_matches_fp32_on_real_speech(speech16k_path):
    """The fp16 variant must track fp32 closely on real speech (near-lossless: corr >0.999)."""
    import soundfile as sf

    clean, sr = sf.read(speech16k_path, dtype="float32")
    noisy = (clean + 0.03 * np.random.RandomState(7).randn(clean.size)).astype(np.float32)
    out32, _ = _engine(precision="fp32").denoise(noisy, sr)
    out16, _ = _engine(precision="fp16").denoise(noisy, sr)
    assert out16.shape == out32.shape
    assert np.isfinite(out16).all()
    n = min(out16.size, out32.size)
    corr = float(np.corrcoef(out16[:n], out32[:n])[0, 1])
    assert corr > 0.999, f"fp16 diverged from fp32: corr={corr:.6f}"


def test_unipase_crosses_the_window_boundary():
    """Longer than one 8 s window, exercising the overlap-add seam."""
    eng = _engine()
    n = _SEG_LEN + _SR * 5
    t = np.arange(n, dtype=np.float32) / _SR
    x = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    out, rate = eng.denoise(x, _SR)
    assert out.size == n and rate == _SR
    assert np.isfinite(out).all()


def test_unipase_is_deterministic():
    eng = _engine()
    x = (0.1 * np.random.RandomState(2).randn(_SR)).astype(np.float32)
    assert np.array_equal(eng.denoise(x, _SR)[0], eng.denoise(x, _SR)[0])


# --- adversarial --- #
def test_unipase_empty_input():
    out, _ = _engine().denoise(np.zeros(0, dtype=np.float32), _SR)
    assert out.size == 0


def test_unipase_silence_stays_finite():
    out, _ = _engine().denoise(np.zeros(_SR, dtype=np.float32), _SR)
    assert out.size == _SR
    assert np.isfinite(out).all()
    assert float(np.max(np.abs(out))) == 0.0


def test_unipase_non_finite_input_does_not_propagate():
    x = np.array([0.1, np.nan, np.inf, -np.inf] * 4000, dtype=np.float32)
    out, _ = _engine().denoise(x, _SR)
    assert np.isfinite(out).all()


def test_unipase_extremely_loud_input():
    x = (np.random.RandomState(1).randn(_SR) * 1e4).astype(np.float32)
    out, _ = _engine().denoise(x, _SR)
    assert np.isfinite(out).all()
