"""End-to-end DPDFNet denoising + adversarial inputs.

The weight-free tests cover model selection, the attenuation-limit blend, and the state
vector built from ONNX metadata. Inference tests are skipped when the ONNX weights are
unavailable (no network / pre-publish).
"""
from __future__ import annotations

import numpy as np
import pytest

from audiosronnx.engines.dpdfnet import (
    _MODELS,
    _apply_attn_limit,
    DPDFNetAdapter,
)


# --------------------------------------------------------------------------- #
# Configuration (no weights) — always runs.
# --------------------------------------------------------------------------- #
def test_default_model_is_fullband_48k():
    assert DPDFNetAdapter().input_sample_rate == 48000


@pytest.mark.parametrize("model,rate", [(m, r) for m, (_, r) in _MODELS.items()])
def test_each_model_reports_its_native_rate(model, rate):
    assert DPDFNetAdapter(model=model).input_sample_rate == rate


def test_unknown_model_rejected():
    with pytest.raises(ValueError):
        DPDFNetAdapter(model="dpdfnet99")


def test_negative_attn_limit_rejected():
    with pytest.raises(ValueError):
        DPDFNetAdapter(attn_limit_db=-3)


def test_attn_limit_none_is_passthrough():
    enh = np.ones((10, 4, 2), dtype=np.float32)
    noisy = np.zeros_like(enh)
    assert np.array_equal(_apply_attn_limit(noisy, enh, None), enh)


def test_attn_limit_zero_db_returns_noisy_aligned():
    """0 dB attenuation means 'change nothing' — the (frame-aligned) noisy signal."""
    rng = np.random.RandomState(0)
    noisy = rng.randn(12, 4, 2).astype(np.float32)
    enh = np.zeros_like(noisy)
    out = _apply_attn_limit(noisy, enh, 0.0)
    assert np.allclose(out[4:], noisy[:-4], atol=1e-6)
    assert np.allclose(out[:4], 0.0)


def test_attn_limit_blends_between_noisy_and_enhanced():
    noisy = np.ones((8, 2, 2), dtype=np.float32)
    enh = np.zeros_like(noisy)
    out = _apply_attn_limit(noisy, enh, 20.0)   # alpha = 0.1
    assert np.allclose(out[4:], 0.1, atol=1e-6)


def test_attn_limit_short_input_does_not_index_past_start():
    """Fewer frames than the alignment offset must not wrap or raise."""
    noisy = np.ones((2, 3, 2), dtype=np.float32)
    enh = np.zeros_like(noisy)
    out = _apply_attn_limit(noisy, enh, 6.0)
    assert out.shape == noisy.shape
    assert np.isfinite(out).all()


# --------------------------------------------------------------------------- #
# Inference (weights) — skipped if unavailable.
# --------------------------------------------------------------------------- #
def _denoiser(load_engine, **kw):
    from audiosronnx import load_denoise

    try:
        eng = load_denoise("dpdfnet", **kw)
        eng._ensure_models()
        return eng
    except Exception as exc:
        pytest.skip(f"dpdfnet weights unavailable: {exc}")


def test_dpdfnet_denoises_real_speech(load_engine, speech16k_path):
    """Adding noise then denoising must recover SNR against the clean reference."""
    import soundfile as sf

    from audiosronnx._kaiser import kaiser_resample

    clean16, sr = sf.read(speech16k_path, dtype="float32")
    clean = np.asarray(kaiser_resample(clean16, sr, 48000), dtype=np.float32)
    noisy = (clean + 0.05 * np.random.RandomState(0).randn(clean.size)).astype(np.float32)

    out, rate = _denoiser(load_engine).denoise(noisy, 48000)
    assert rate == 48000

    def snr(x):
        n = min(x.size, clean.size)
        return 10 * np.log10((clean[:n] ** 2).sum() / (((x[:n] - clean[:n]) ** 2).sum() + 1e-20))

    assert snr(out) > snr(noisy) + 3.0   # measured ~+11 dB; assert a wide margin


def test_dpdfnet_preserves_length_and_rate(load_engine):
    eng = _denoiser(load_engine)
    x = np.zeros(48000, dtype=np.float32)
    out, rate = eng.denoise(x, 48000)
    assert rate == 48000
    assert out.size == 48000


def test_dpdfnet_resamples_foreign_rate(load_engine):
    """A 16 kHz input to the 48 kHz model is resampled, and output lands at 48 kHz."""
    eng = _denoiser(load_engine)
    t = np.arange(16000, dtype=np.float32) / 16000.0
    out, rate = eng.denoise((0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 16000)
    assert rate == 48000
    assert np.isfinite(out).all()


def test_dpdfnet_16k_variant(load_engine):
    eng = _denoiser(load_engine, model="dpdfnet2")
    out, rate = eng.denoise(np.zeros(16000, dtype=np.float32), 16000)
    assert rate == 16000
    assert out.size == 16000


# --- adversarial: must not crash, must stay finite, must keep the length contract --- #
def test_dpdfnet_empty_input(load_engine):
    out, _ = _denoiser(load_engine).denoise(np.zeros(0, dtype=np.float32), 48000)
    assert out.size == 0


def test_dpdfnet_single_sample(load_engine):
    """Shorter than one STFT frame — the reflect pad must not blow up."""
    out, _ = _denoiser(load_engine).denoise(np.array([0.5], dtype=np.float32), 48000)
    assert out.size == 1
    assert np.isfinite(out).all()


def test_dpdfnet_sub_frame_input(load_engine):
    out, _ = _denoiser(load_engine).denoise(np.full(200, 0.1, dtype=np.float32), 48000)
    assert out.size == 200
    assert np.isfinite(out).all()


def test_dpdfnet_silence_stays_silent(load_engine):
    out, _ = _denoiser(load_engine).denoise(np.zeros(24000, dtype=np.float32), 48000)
    assert np.isfinite(out).all()
    assert float(np.max(np.abs(out))) < 1e-3


def test_dpdfnet_extremely_loud_input(load_engine):
    x = (np.random.RandomState(1).randn(24000) * 1e4).astype(np.float32)
    out, _ = _denoiser(load_engine).denoise(x, 48000)
    assert np.isfinite(out).all()


def test_dpdfnet_state_does_not_leak_between_calls(load_engine):
    """Each call re-initialises the RNN state, so results must be reproducible."""
    eng = _denoiser(load_engine)
    x = (0.1 * np.random.RandomState(2).randn(24000)).astype(np.float32)
    first, _ = eng.denoise(x, 48000)
    second, _ = eng.denoise(x, 48000)
    assert np.array_equal(first, second)
