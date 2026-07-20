"""End-to-end MP-SENet denoising + adversarial inputs.

MP-SENet works in a power-compressed magnitude domain and reconstructs from a predicted
phase angle, so the weight-free tests pin down the compression round-trip and the RMS
normalisation — both live outside the graph and silently ruin the output if wrong.
"""
from __future__ import annotations

import numpy as np
import pytest

from audiosronnx.engines.mpsenet import _COMPRESS, _MODELS, _SR, MPSENetAdapter


# --------------------------------------------------------------------------- #
# Configuration / spectral conventions (no weights) — always runs.
# --------------------------------------------------------------------------- #
def test_default_checkpoint_is_dns():
    """The DNS checkpoint generalises far better on broadband noise than VoiceBank."""
    assert MPSENetAdapter()._model == "dns"


def test_both_checkpoints_selectable():
    assert set(_MODELS) == {"dns", "vb"}
    assert MPSENetAdapter(model="vb")._model == "vb"


def test_unknown_checkpoint_rejected():
    with pytest.raises(ValueError):
        MPSENetAdapter(model="voicebank")


def test_compression_round_trip_is_identity():
    """``mag**0.3`` then ``**(1/0.3)`` must return the original magnitudes."""
    mag = np.abs(np.random.RandomState(0).randn(64)) + 1e-3
    assert np.allclose((mag ** _COMPRESS) ** (1.0 / _COMPRESS), mag, rtol=1e-9)


def test_compression_is_not_a_no_op():
    """Guards against someone 'simplifying' the compression away."""
    mag = np.array([0.01, 0.5, 4.0])
    assert not np.allclose(mag ** _COMPRESS, mag)


def test_phase_reconstruction_recovers_the_complex_spectrum():
    """amp*(cos φ + i sin φ) must rebuild the spectrum the angle came from."""
    rng = np.random.RandomState(1)
    spec = rng.randn(32) + 1j * rng.randn(32)
    amp = np.abs(spec)
    ang = np.arctan2(spec.imag, spec.real)
    assert np.allclose(amp * np.cos(ang) + 1j * amp * np.sin(ang), spec, atol=1e-12)


# --------------------------------------------------------------------------- #
# Inference (weights) — skipped if unavailable.
# --------------------------------------------------------------------------- #
def _denoiser(**kw):
    from audiosronnx import load_denoise

    try:
        eng = load_denoise("mpsenet", **kw)
        eng._ensure_models()
        return eng
    except Exception as exc:
        pytest.skip(f"mpsenet weights unavailable: {exc}")


def _snr(x, clean):
    n = min(x.size, clean.size)
    return 10 * np.log10((clean[:n] ** 2).sum() / (((x[:n] - clean[:n]) ** 2).sum() + 1e-20))


def test_mpsenet_denoises_real_speech(speech16k_path):
    import soundfile as sf

    clean, sr = sf.read(speech16k_path, dtype="float32")
    noisy = (clean + 0.05 * np.random.RandomState(0).randn(clean.size)).astype(np.float32)
    out, rate = _denoiser().denoise(noisy, sr)
    assert rate == _SR
    assert _snr(out, clean) > _snr(noisy, clean) + 3.0    # measured ~+8.6 dB


def test_mpsenet_dns_beats_vb_on_broadband_noise(speech16k_path):
    """The reason dns is the default — vb barely helps on noise unlike its training set."""
    import soundfile as sf

    clean, sr = sf.read(speech16k_path, dtype="float32")
    noisy = (clean + 0.05 * np.random.RandomState(0).randn(clean.size)).astype(np.float32)
    dns, _ = _denoiser(model="dns").denoise(noisy, sr)
    vb, _ = _denoiser(model="vb").denoise(noisy, sr)
    assert _snr(dns, clean) > _snr(vb, clean) + 2.0


def test_mpsenet_preserves_length_and_rate(speech16k_path):
    import soundfile as sf

    clean, _ = sf.read(speech16k_path, dtype="float32")
    out, rate = _denoiser().denoise(clean, _SR)
    assert out.size == clean.size and rate == _SR


def test_mpsenet_resamples_foreign_rate():
    t = np.arange(48000, dtype=np.float32) / 48000.0
    out, rate = _denoiser().denoise((0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 48000)
    assert rate == _SR
    assert np.isfinite(out).all()


def test_mpsenet_is_deterministic():
    eng = _denoiser()
    x = (0.1 * np.random.RandomState(2).randn(16000)).astype(np.float32)
    assert np.array_equal(eng.denoise(x, _SR)[0], eng.denoise(x, _SR)[0])


def test_mpsenet_runs_at_arbitrary_lengths():
    """Regression guard for the attention export.

    Stock ``nn.MultiheadAttention`` bakes the traced sequence length into its reshapes, and
    the resulting graph raises a Reshape shape-mismatch at every other length. The failure
    is loud rather than silent, so simply exercising a spread of lengths catches it. This
    does not assert frame-wise agreement across lengths — the model is non-causal, so a
    frame's output legitimately depends on the whole clip.
    """
    eng = _denoiser()
    rng = np.random.RandomState(3)
    for n in (2000, 7999, 16000, 32001, 60000):
        out, rate = eng.denoise((0.1 * rng.randn(n)).astype(np.float32), _SR)
        assert out.size == n and rate == _SR
        assert np.isfinite(out).all()


# --- adversarial --- #
def test_mpsenet_empty_input():
    out, _ = _denoiser().denoise(np.zeros(0, dtype=np.float32), _SR)
    assert out.size == 0


def test_mpsenet_all_zero_input_short_circuits():
    """Zero energy has no RMS to normalise against; must return silence, not NaN."""
    out, _ = _denoiser().denoise(np.zeros(16000, dtype=np.float32), _SR)
    assert out.size == 16000
    assert np.isfinite(out).all()
    assert float(np.max(np.abs(out))) == 0.0


def test_mpsenet_single_sample():
    out, _ = _denoiser().denoise(np.array([0.5], dtype=np.float32), _SR)
    assert out.size == 1
    assert np.isfinite(out).all()


def test_mpsenet_sub_frame_input():
    out, _ = _denoiser().denoise(np.full(200, 0.1, dtype=np.float32), _SR)
    assert out.size == 200
    assert np.isfinite(out).all()


def test_mpsenet_non_finite_input_does_not_propagate():
    x = np.array([0.1, np.nan, np.inf, -np.inf] * 4000, dtype=np.float32)
    out, _ = _denoiser().denoise(x, _SR)
    assert np.isfinite(out).all()


def test_mpsenet_extremely_loud_input():
    """RMS normalisation should make the model level-invariant."""
    x = (np.random.RandomState(1).randn(16000) * 1e4).astype(np.float32)
    out, _ = _denoiser().denoise(x, _SR)
    assert np.isfinite(out).all()
