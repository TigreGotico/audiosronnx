"""End-to-end NovaSR inference. Skipped when the ONNX weights are unavailable."""
from __future__ import annotations

import numpy as np


def test_novasr_upscales_16k_to_48k(load_engine, speech16k_path):
    model = load_engine("novasr")
    out, sr = model.upscale(speech16k_path)
    assert sr == 48000
    assert out.dtype == np.float32
    assert out.size > 0
    assert float(np.max(np.abs(out))) > 1e-3


def test_novasr_output_roughly_3x_length(load_engine, sine_16k):
    model = load_engine("novasr")
    out, sr = model.upscale(sine_16k, 16000)
    assert sr == 48000
    # 16k -> 48k is a 3x time-domain upsample
    assert abs(out.size - 3 * sine_16k.size) < sine_16k.size
