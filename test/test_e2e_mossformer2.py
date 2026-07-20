"""End-to-end MossFormer2 denoising, the Kaldi front-end, and adversarial inputs.

The Kaldi filterbank is reproduced from ``torchaudio.compliance.kaldi.fbank``, so the
weight-free tests lock its numerics down with golden values — a silent drift there would
feed the model features it was not trained on.

The length-independence test guards a specific export hazard: exporting this graph with
``do_constant_folding=True`` bakes in the traced sequence length, and the result still
looks correct when checked at that one length.
"""
from __future__ import annotations

import numpy as np
import pytest

from audiosronnx._kaldi_fbank import deltas, fbank, hamming, mel_banks


# --------------------------------------------------------------------------- #
# Kaldi front-end (no weights) — always runs.
# --------------------------------------------------------------------------- #
def _noise(n=48000, seed=0):
    return (np.random.RandomState(seed).randn(n) * 0.1 * 32768).astype(np.float32)


def test_fbank_matches_golden_values():
    """Golden values captured against torchaudio (agreement was 3.2e-05)."""
    feats = fbank(_noise(), 48000, 1920, 384, 60)
    assert feats.shape == (121, 60)
    assert np.allclose(feats[0, :4],
                       [17.193676, 15.532068, 16.106571, 17.785894], atol=1e-4)
    assert np.isclose(feats.mean(), 23.107218, atol=1e-4)


def test_fbank_frame_count_follows_snip_edges():
    """Kaldi keeps only frames that fit entirely: 1 + (n - win) // hop."""
    for n in (19200, 48000, 50000):
        assert fbank(_noise(n), 48000, 1920, 384, 60).shape[0] == 1 + (n - 1920) // 384


def test_fbank_shorter_than_window_yields_no_frames():
    assert fbank(_noise(1000), 48000, 1920, 384, 60).shape == (0, 60)


def test_fbank_is_scale_sensitive():
    """log-mel is not scale-invariant — the 32768 pre-scale in the adapter matters."""
    x = _noise()
    assert not np.allclose(fbank(x, 48000, 1920, 384, 60),
                           fbank(x / 32768.0, 48000, 1920, 384, 60), atol=1e-3)


def test_fbank_on_silence_is_finite_at_the_floor():
    """Zero power must hit the float32-epsilon floor, not -inf."""
    feats = fbank(np.zeros(48000, dtype=np.float32), 48000, 1920, 384, 60)
    assert np.isfinite(feats).all()
    assert np.allclose(feats, np.log(np.finfo(np.float32).eps), atol=1e-3)


def test_fbank_dither_is_off_by_default_and_deterministic():
    x = _noise()
    assert np.array_equal(fbank(x, 48000, 1920, 384, 60), fbank(x, 48000, 1920, 384, 60))


def test_fbank_dither_perturbs_when_enabled():
    x = _noise()
    a = fbank(x, 48000, 1920, 384, 60, dither=1.0)
    b = fbank(x, 48000, 1920, 384, 60, dither=1.0)
    assert not np.array_equal(a, b)


def test_mel_banks_are_triangular_and_bounded():
    banks = mel_banks(60, 2048, 48000)
    assert banks.shape == (60, 1024)
    assert banks.min() >= 0.0 and banks.max() <= 1.0 + 1e-6
    # each filter peaks once — a triangle, not a plateau
    assert (banks.argmax(axis=1)[1:] > banks.argmax(axis=1)[:-1]).all()


def test_hamming_is_symmetric_and_matches_endpoints():
    w = hamming(1920)
    assert np.allclose(w, w[::-1])
    assert np.isclose(w[0], 0.54 - 0.46)      # symmetric window starts at alpha-beta


def test_deltas_of_constant_signal_are_zero():
    assert np.allclose(deltas(np.ones((20, 5), dtype=np.float32)), 0.0, atol=1e-7)


def test_deltas_of_linear_ramp_are_constant_in_the_interior():
    ramp = np.arange(20, dtype=np.float32)[:, None] * np.ones((1, 3), dtype=np.float32)
    d = deltas(ramp)
    assert np.allclose(d[3:-3], 1.0, atol=1e-5)   # slope 1 per frame; edges are padded


# --------------------------------------------------------------------------- #
# Inference (weights) — skipped if unavailable.
# --------------------------------------------------------------------------- #
def _denoiser(**kw):
    from audiosronnx import load_denoise

    try:
        eng = load_denoise("mossformer2", **kw)
        eng._ensure_models()
        return eng
    except Exception as exc:
        pytest.skip(f"mossformer2 weights unavailable: {exc}")


def test_mossformer2_mask_stays_sane_across_lengths():
    """Regression guard against a constant-folded export.

    Folding bakes in the traced sequence length; the graph then still looks correct at
    that one length while degrading badly elsewhere — masks blew up to ~4x their normal
    magnitude at 2x the traced length. This does not assert frame-wise equality (the model
    is a non-causal transformer, so a frame's mask legitimately depends on later context)
    — only that the mask distribution stays in the same band as the sequence grows.
    """
    eng = _denoiser()
    rng = np.random.RandomState(0)
    name = eng._sess.get_inputs()[0].name
    stats = []
    for frames in (100, 300, 600):
        mask = eng._sess.run(None, {name: rng.randn(1, frames, 180).astype(np.float32)})[0]
        assert np.isfinite(mask).all()
        stats.append((float(np.abs(mask).mean()), float(np.abs(mask).max())))
    means = [m for m, _ in stats]
    maxes = [x for _, x in stats]
    assert max(means) < 3 * min(means), f"mask magnitude drifts with length: {stats}"
    assert max(maxes) < 3 * min(maxes), f"mask peak drifts with length: {stats}"


def test_mossformer2_denoises_real_speech(speech16k_path):
    import soundfile as sf

    from audiosronnx._kaiser import kaiser_resample

    clean16, sr = sf.read(speech16k_path, dtype="float32")
    clean = np.asarray(kaiser_resample(clean16, sr, 48000), dtype=np.float32)
    noisy = (clean + 0.05 * np.random.RandomState(0).randn(clean.size)).astype(np.float32)
    out, rate = _denoiser().denoise(noisy, 48000)
    assert rate == 48000

    def snr(x):
        n = min(x.size, clean.size)
        return 10 * np.log10((clean[:n] ** 2).sum() / (((x[:n] - clean[:n]) ** 2).sum() + 1e-20))

    assert snr(out) > snr(noisy) + 4.0    # measured ~+11 dB


def test_mossformer2_preserves_length_and_rate():
    out, rate = _denoiser().denoise(np.zeros(96000, dtype=np.float32), 48000)
    assert out.size == 96000 and rate == 48000


def test_mossformer2_resamples_foreign_rate():
    t = np.arange(16000, dtype=np.float32) / 16000.0
    out, rate = _denoiser().denoise((0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 16000)
    assert rate == 48000
    assert np.isfinite(out).all()


def test_mossformer2_is_deterministic():
    eng = _denoiser()
    x = (0.1 * np.random.RandomState(2).randn(96000)).astype(np.float32)
    assert np.array_equal(eng.denoise(x, 48000)[0], eng.denoise(x, 48000)[0])


# --- adversarial --- #
def test_mossformer2_empty_input():
    out, _ = _denoiser().denoise(np.zeros(0, dtype=np.float32), 48000)
    assert out.size == 0


def test_mossformer2_shorter_than_one_window():
    """Fewer samples than the 1920-sample window yields no frames at all."""
    out, _ = _denoiser().denoise(np.full(500, 0.1, dtype=np.float32), 48000)
    assert out.size == 500
    assert np.isfinite(out).all()


def test_mossformer2_single_sample():
    out, _ = _denoiser().denoise(np.array([0.5], dtype=np.float32), 48000)
    assert out.size == 1
    assert np.isfinite(out).all()


def test_mossformer2_silence_stays_silent():
    out, _ = _denoiser().denoise(np.zeros(96000, dtype=np.float32), 48000)
    assert np.isfinite(out).all()
    assert float(np.max(np.abs(out))) < 1e-3


def test_mossformer2_extremely_loud_input():
    x = (np.random.RandomState(1).randn(96000) * 1e4).astype(np.float32)
    out, _ = _denoiser().denoise(x, 48000)
    assert np.isfinite(out).all()
