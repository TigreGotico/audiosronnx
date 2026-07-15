"""End-to-end HiFi-GAN+ inference. Skipped when the ONNX weights are unavailable."""
from __future__ import annotations

import numpy as np


def test_hifiganbwe_upscales_to_48k(load_engine, speech16k_path):
    model = load_engine("hifiganbwe")
    out, sr = model.upscale(speech16k_path)
    assert sr == 48000
    assert out.dtype == np.float32
    assert out.size > 0
    assert float(np.max(np.abs(out))) > 1e-3


def test_hifiganbwe_accepts_8k_input(load_engine, speech8k_path):
    model = load_engine("hifiganbwe")
    out, sr = model.upscale(speech8k_path)
    assert sr == 48000
    assert float(np.max(np.abs(out))) > 1e-3
