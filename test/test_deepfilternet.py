"""DeepFilterNet denoise engine: registry wiring + the DSP that lives outside the graph.

No weights required — the pieces exercised here are the ones the graphs do not carry.
"""
from __future__ import annotations

import numpy as np
import pytest

from audiosronnx import ENGINE_REGISTRY, available_denoisers, available_models, get_engine, load_denoise
from audiosronnx.engines.deepfilternet import DeepFilterNetAdapter


def test_registered_as_a_denoiser():
    assert "deepfilternet" in available_models()
    assert "deepfilternet" in available_denoisers()
    entry = get_engine("deepfilternet")
    assert entry.kind == "denoise"
    # a denoiser preserves the rate rather than upscaling to 48 kHz
    assert entry.input_sample_rate == entry.output_sample_rate == 48000


def test_load_denoise_rejects_an_sr_engine():
    with pytest.raises(ValueError, match="load_sr"):
        load_denoise("lavasr")


def test_sr_engines_are_not_denoisers():
    assert not {"lavasr", "novasr", "apbwe", "hifiganbwe"} & set(available_denoisers())


def test_pad_feat_applies_the_encoder_lookahead():
    """``pad_feat`` drops ``lookahead`` frames from the front and zero-fills the tail.

    The model does this before the encoder and it is not in the exported graph; skipping it
    puts every mask/coefficient out of step with the spectrum.
    """
    t = 6
    x = np.arange(t, dtype=np.float32).reshape(1, 1, t, 1)  # frames 0..5
    out = DeepFilterNetAdapter._pad_feat(x, lookahead=2)
    assert out.shape == x.shape                       # length preserved
    assert out[0, 0, :, 0].tolist() == [2, 3, 4, 5, 0, 0]


def test_pad_feat_is_a_noop_without_lookahead():
    x = np.arange(4, dtype=np.float32).reshape(1, 1, 4, 1)
    assert np.array_equal(DeepFilterNetAdapter._pad_feat(x, lookahead=0), x)


def test_deep_filter_is_freq_major_and_only_touches_low_bins():
    """``df_dec`` emits ``[B, T, F, O, 2]`` — freq-major. A delta-at-tap-2 filter with the
    lookahead must reproduce the current frame, and bins above ``nb_df`` stay untouched.
    """
    adapter = DeepFilterNetAdapter.__new__(DeepFilterNetAdapter)  # no weights needed
    frames, bins = 4, 481
    spec = (np.arange(frames * bins, dtype=np.float32).reshape(1, frames, bins)
            + 1j * np.zeros((1, frames, bins), np.float32))
    # identity filter: tap index == df_order-1-df_lookahead == 2 selects the current frame
    coefs = np.zeros((1, frames, 96, 5, 2), dtype=np.float32)
    coefs[:, :, :, 2, 0] = 1.0
    out = adapter._deep_filter(spec, coefs.reshape(1, frames, 96 * 5 * 2), frames)
    assert np.allclose(out[0, :, :96].real, spec[0, :, :96].real)   # low bins reproduced
    assert np.allclose(out[0, :, 96:], spec[0, :, 96:])             # high bins untouched
