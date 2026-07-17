"""End-to-end Sidon restoration + adversarial inputs.

The inference tests are skipped when the ONNX weights are unavailable (no network /
pre-publish). The numpy mel front-end tests need no weights and always run.
"""
from __future__ import annotations

import numpy as np
import pytest

from audiosronnx.engines.sidon import seamless_fbank


# --------------------------------------------------------------------------- #
# Front-end (no weights) — always runs.
# --------------------------------------------------------------------------- #
def test_seamless_fbank_shape_even_and_odd():
    # 16000 samples -> 98 mel frames (no centering) -> 49 stacked frames of 160 features
    feats = seamless_fbank(np.zeros(16000, dtype=np.float32))
    assert feats.shape == (49, 160)
    assert feats.dtype == np.float32
    # the adapter's ±160 edge pad yields the 50-frame windows the model expects
    padded = seamless_fbank(np.pad(np.zeros(16000, dtype=np.float32), (160, 160)))
    assert padded.shape == (50, 160)
    # an odd mel-frame count is padded up to the stride before stacking
    odd = seamless_fbank(np.zeros(16000 + 160, dtype=np.float32))
    assert odd.shape[1] == 160


def test_seamless_fbank_rejects_stereo():
    with pytest.raises(ValueError):
        seamless_fbank(np.zeros((2, 16000), dtype=np.float32))


def test_seamless_fbank_finite_on_signal():
    t = np.arange(16000, dtype=np.float32) / 16000.0
    feats = seamless_fbank(0.3 * np.sin(2 * np.pi * 220.0 * t))
    assert np.isfinite(feats).all()


# --------------------------------------------------------------------------- #
# Inference (weights) — skipped if unavailable.
# --------------------------------------------------------------------------- #
def test_sidon_restores_to_48k(load_engine, speech16k_path):
    model = load_engine("sidon")
    out, sr = model.upscale(speech16k_path)
    assert sr == 48000
    assert out.dtype == np.float32
    assert out.size > 0
    assert float(np.max(np.abs(out))) > 1e-3


def test_sidon_output_length_tracks_input(load_engine, speech16k_path):
    import soundfile as sf

    model = load_engine("sidon")
    audio, in_sr = sf.read(speech16k_path)
    out, sr = model.upscale(speech16k_path)
    expected = int(round(48000 / in_sr * len(audio)))
    assert abs(out.size - expected) <= 960  # within one feature frame


def test_sidon_resamples_8k_input(load_engine, speech8k_path):
    model = load_engine("sidon")
    out, sr = model.upscale(speech8k_path)
    assert sr == 48000
    assert np.isfinite(out).all()


# --- adversarial: must not crash, must stay finite, must keep the length contract --- #
def test_sidon_silence_returns_finite_zeros(load_engine):
    model = load_engine("sidon")
    out, sr = model.upscale(np.zeros(16000, dtype=np.float32), 16000)
    assert sr == 48000
    assert out.size == 48000
    assert np.isfinite(out).all()
    assert float(np.max(np.abs(out))) == 0.0


def test_sidon_empty_input(load_engine):
    model = load_engine("sidon")
    out, sr = model.upscale(np.zeros(0, dtype=np.float32), 16000)
    assert out.size == 0


def test_sidon_single_sample(load_engine):
    model = load_engine("sidon")
    out, sr = model.upscale(np.array([0.5], dtype=np.float32), 16000)
    assert np.isfinite(out).all()


def test_sidon_non_finite_input_does_not_propagate(load_engine):
    model = load_engine("sidon")
    x = np.array([0.1, np.nan, np.inf, -np.inf] * 4000, dtype=np.float32)
    out, sr = model.upscale(x, 16000)
    assert sr == 48000
    assert np.isfinite(out).all()  # non-finite peak -> silent, finite output


def test_sidon_extremely_loud_input(load_engine):
    model = load_engine("sidon")
    x = (np.random.RandomState(1).randn(16000) * 1e4).astype(np.float32)
    out, sr = model.upscale(x, 16000)
    assert np.isfinite(out).all()


def test_sidon_multi_chunk_seam(load_engine):
    """Input spanning several windows exercises the feature-cache seam.

    A tiny ``chunk_samples`` forces multiple windows over a short clip, so the seam is
    covered without the O(T^2) memory cost of a full 96 s window.
    """
    model = load_engine("sidon", chunk_samples=16000)  # 1 s windows
    t = np.arange(16000 * 4, dtype=np.float32) / 16000.0  # 4 s -> 4 windows
    x = (0.2 * np.sin(2 * np.pi * 180.0 * t)).astype(np.float32)
    out, sr = model.upscale(x, 16000)
    assert sr == 48000
    assert np.isfinite(out).all()
    assert abs(out.size - 48000 * 4) <= 960
