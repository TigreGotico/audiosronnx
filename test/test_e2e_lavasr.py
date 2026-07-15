"""End-to-end LavaSR inference. Skipped when the ONNX weights are unavailable."""
from __future__ import annotations

import numpy as np


def test_lavasr_upscales_16k_to_48k(load_engine, speech16k_path):
    model = load_engine("lavasr")
    out, sr = model.upscale(speech16k_path)
    assert sr == 48000
    assert out.dtype == np.float32
    assert out.size > 0
    # output must not be silent
    assert float(np.max(np.abs(out))) > 1e-3


def test_lavasr_accepts_8k_input(load_engine, speech8k_path):
    model = load_engine("lavasr")
    out, sr = model.upscale(speech8k_path)
    assert sr == 48000
    assert float(np.max(np.abs(out))) > 1e-3


def test_lavasr_array_input(load_engine, sine_16k):
    model = load_engine("lavasr")
    out, sr = model.upscale(sine_16k, 16000)
    assert sr == 48000
    # a 300 Hz tone upscaled should retain energy near ~1s * 48k samples
    assert out.size > 40000
