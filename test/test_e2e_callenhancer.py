"""End-to-end CallEnhancer restoration + adversarial inputs.

The inference tests are skipped when the ONNX weights are unavailable (no network /
pre-publish). CallEnhancer shares Sidon's numpy SeamlessM4T front-end, which is covered
in ``test_e2e_sidon.py``; here the weight-free tests exercise the adapter's own segment
bounds / chunking logic without any ONNX session.
"""
from __future__ import annotations

import numpy as np

from audiosronnx.engines.callenhancer import CallEnhancerAdapter


# --------------------------------------------------------------------------- #
# Segment bounds (no weights) — always runs.
# --------------------------------------------------------------------------- #
def test_single_pass_by_default():
    model = CallEnhancerAdapter()
    assert model._bounds(16000 * 30) == [(0, 16000 * 30)]


def test_chunking_windows_with_overlap():
    model = CallEnhancerAdapter(chunk_seconds=10.0)  # 10 s windows, 2 s overlap
    n = 16000 * 25
    bounds = model._bounds(n)
    assert len(bounds) > 1
    # windows cover the whole clip and never run past the end
    assert bounds[0][0] == 0
    assert bounds[-1][1] == n
    assert all(0 <= s < e <= n for s, e in bounds)
    # 8 s hop (10 s window - 2 s overlap)
    assert bounds[1][0] == 8 * 16000


def test_short_clip_is_single_pass_even_when_chunking():
    model = CallEnhancerAdapter(chunk_seconds=10.0)
    assert model._bounds(16000) == [(0, 16000)]


# --------------------------------------------------------------------------- #
# Inference (weights) — skipped if unavailable.
# --------------------------------------------------------------------------- #
def test_callenhancer_restores_to_48k(load_engine, speech16k_path):
    model = load_engine("callenhancer")
    out, sr = model.upscale(speech16k_path)
    assert sr == 48000
    assert out.dtype == np.float32
    assert out.size > 0
    assert float(np.max(np.abs(out))) > 1e-3


def test_callenhancer_output_length_tracks_input(load_engine, speech16k_path):
    import soundfile as sf

    model = load_engine("callenhancer")
    audio, in_sr = sf.read(speech16k_path)
    out, sr = model.upscale(speech16k_path)
    expected = int(round(48000 / in_sr * len(audio)))
    assert abs(out.size - expected) <= 960

def test_callenhancer_output_is_peak_normalised(load_engine, speech16k_path):
    model = load_engine("callenhancer")
    out, sr = model.upscale(speech16k_path)
    # upstream normalises the restored channel to a 0.97 peak
    assert float(np.max(np.abs(out))) <= 0.9701


def test_callenhancer_resamples_8k_input(load_engine, speech8k_path):
    model = load_engine("callenhancer")
    out, sr = model.upscale(speech8k_path)
    assert sr == 48000
    assert np.isfinite(out).all()


# --- adversarial: must not crash, must stay finite, must keep the length contract --- #
def test_callenhancer_silence_returns_finite_zeros(load_engine):
    model = load_engine("callenhancer")
    out, sr = model.upscale(np.zeros(16000, dtype=np.float32), 16000)
    assert sr == 48000
    assert out.size == 48000
    assert np.isfinite(out).all()
    assert float(np.max(np.abs(out))) == 0.0


def test_callenhancer_empty_input(load_engine):
    model = load_engine("callenhancer")
    out, sr = model.upscale(np.zeros(0, dtype=np.float32), 16000)
    assert out.size == 0


def test_callenhancer_single_sample(load_engine):
    model = load_engine("callenhancer")
    out, sr = model.upscale(np.array([0.5], dtype=np.float32), 16000)
    assert np.isfinite(out).all()


def test_callenhancer_non_finite_input_does_not_propagate(load_engine):
    model = load_engine("callenhancer")
    x = np.array([0.1, np.nan, np.inf, -np.inf] * 4000, dtype=np.float32)
    out, sr = model.upscale(x, 16000)
    assert sr == 48000
    assert np.isfinite(out).all()  # non-finite peak -> silent, finite output


def test_callenhancer_extremely_loud_input(load_engine):
    model = load_engine("callenhancer")
    x = (np.random.RandomState(1).randn(16000) * 1e4).astype(np.float32)
    out, sr = model.upscale(x, 16000)
    assert np.isfinite(out).all()


def test_callenhancer_multi_window_crossfade(load_engine):
    """Input spanning several windows exercises the crossfade seam."""
    model = load_engine("callenhancer", chunk_seconds=1.0)  # 1 s windows
    t = np.arange(16000 * 4, dtype=np.float32) / 16000.0  # 4 s -> multiple windows
    x = (0.2 * np.sin(2 * np.pi * 180.0 * t)).astype(np.float32)
    out, sr = model.upscale(x, 16000)
    assert sr == 48000
    assert np.isfinite(out).all()
    assert abs(out.size - 48000 * 4) <= 960
