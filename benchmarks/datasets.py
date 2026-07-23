"""Gold sets and seeded degradations for the benchmark harness.

Every track pairs a *degraded* input with a *clean* reference. The reference is
never synthesised: it is the original recording. What varies is how the input is
degraded, and each degradation is seeded per file (``rng(seed + file_index)``) so
a given file gets the same corruption on every run and across engines — the whole
point of a scoreboard is that every engine sees the identical input.

Datasets stream from HuggingFace, pinned to a fixed revision so the gold set does
not drift under us. Nothing large is materialised: rows arrive one at a time and
only the requested ``--limit`` are ever pulled.

Tracks
------
denoise
    `VoiceBank-DEMAND <https://huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k>`_
    test set — 824 paired ``(noisy, clean)`` files at 16 kHz. The degradation is
    real recorded noise, so nothing is added here.
sr
    `VCTK <https://huggingface.co/datasets/sanchit-gandhi/vctk>`_ clean speech at
    48 kHz, downsampled to a narrowband input rate (8 or 16 kHz). The clean 48 kHz
    signal is the reference; the task is to reconstruct the missing high band.
enhance
    VCTK clean speech with a seeded *compound* degradation — additive colored
    noise at a target SNR, a synthetic exponential-decay reverb, and a lowpass —
    stacked to mimic a damaged recording. The clean 48 kHz signal is the reference.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

from audiosronnx._kaiser import kaiser_resample

# --------------------------------------------------------------------------- #
# Pinned gold sets
# --------------------------------------------------------------------------- #
#: VoiceBank+DEMAND test set, 16 kHz, columns ``id`` / ``clean`` / ``noisy``.
DENOISE_DATASET = "JacobLinCool/VoiceBank-DEMAND-16k"
DENOISE_REVISION = "4497db342d7312978c45690591fda86117831940"
DENOISE_SPLIT = "test"

#: VCTK, 48 kHz clean speech. No official splits — the ``train`` split is the corpus.
VCTK_DATASET = "sanchit-gandhi/vctk"
VCTK_REVISION = "73ef4ee7d49a6fed4ea1efd65f82b4c95faeb9de"
VCTK_SPLIT = "train"


@dataclass
class Sample:
    """One benchmark item: a degraded input and its clean reference."""

    id: str
    degraded: np.ndarray  # mono float32
    degraded_sr: int
    ref: Optional[np.ndarray]  # mono float32, or None for a no-reference-only item
    ref_sr: Optional[int]


# --------------------------------------------------------------------------- #
# Row → (array, sr) extraction (tolerant of datasets versions)
# --------------------------------------------------------------------------- #
def _to_mono_f32(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        # (channels, N) or (N, channels) → average to mono
        x = x.mean(axis=0) if x.shape[0] < x.shape[-1] else x.mean(axis=-1)
    return np.ascontiguousarray(x, dtype=np.float32)


def _extract_audio(cell) -> tuple[np.ndarray, int]:
    """Pull ``(mono float32, sample_rate)`` out of a HF audio cell.

    Handles the ``datasets`` 3.x dict form ``{"array", "sampling_rate"}`` and the
    4.x/5.x ``AudioDecoder`` object exposing ``get_all_samples()``.
    """
    if hasattr(cell, "get_all_samples"):
        s = cell.get_all_samples()
        return _to_mono_f32(np.asarray(s.data)), int(s.sample_rate)
    if isinstance(cell, dict) and "array" in cell:
        return _to_mono_f32(cell["array"]), int(cell["sampling_rate"])
    raise TypeError(f"unrecognised audio cell type: {type(cell)!r}")


def _stream(dataset: str, revision: str, split: str):
    from datasets import load_dataset

    return load_dataset(dataset, split=split, streaming=True, revision=revision)


# --------------------------------------------------------------------------- #
# Seeded degradations
# --------------------------------------------------------------------------- #
def _colored_noise(rng: np.random.Generator, n: int, exponent: float = 1.0) -> np.ndarray:
    """1/f**exponent (pink-ish) noise, unit-variance, mono float32.

    Colored noise is a far more speech-like corruption than white noise: its
    energy concentrates where speech lives, so an engine cannot win by simply
    lowpassing.
    """
    white = rng.standard_normal(n).astype(np.float32)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.power(freqs[1:], exponent / 2.0)
    shaped = np.fft.irfft(spec * scale, n=n).astype(np.float32)
    std = float(shaped.std()) or 1.0
    return shaped / std


def _mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    clean_p = float((clean ** 2).mean()) + 1e-12
    noise_p = float((noise ** 2).mean()) + 1e-12
    gain = math.sqrt(clean_p / (noise_p * (10.0 ** (snr_db / 10.0))))
    return (clean + gain * noise).astype(np.float32)


def _synthetic_reverb(rng: np.random.Generator, x: np.ndarray, sr: int, rt60: float = 0.4) -> np.ndarray:
    """Convolve with a seeded exponential-decay noise burst — a cheap synthetic room."""
    length = max(1, int(rt60 * sr))
    t = np.arange(length, dtype=np.float32) / sr
    ir = rng.standard_normal(length).astype(np.float32) * np.exp(-6.9 * t / rt60)
    ir[0] += 1.0  # keep the direct path dominant
    ir /= np.abs(ir).sum() + 1e-9
    wet = np.convolve(x, ir)[: x.shape[0]].astype(np.float32)
    return wet


def _lowpass(x: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    """Brick-wall lowpass in the STFT/rFFT domain — models band loss."""
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.shape[0], d=1.0 / sr)
    spec[freqs > cutoff_hz] = 0.0
    return np.fft.irfft(spec, n=x.shape[0]).astype(np.float32)


def _peak_norm(x: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = float(np.abs(x).max()) or 1.0
    return (x * (peak / m)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Track iterators
# --------------------------------------------------------------------------- #
def denoise_samples(limit: int, seed: int) -> Iterator[Sample]:
    """VoiceBank+DEMAND test set — recorded noisy input, clean reference (16 kHz)."""
    ds = _stream(DENOISE_DATASET, DENOISE_REVISION, DENOISE_SPLIT)
    for i, row in enumerate(ds):
        if i >= limit:
            break
        noisy, nsr = _extract_audio(row["noisy"])
        clean, csr = _extract_audio(row["clean"])
        yield Sample(str(row.get("id", i)), noisy, nsr, clean, csr)


def sr_samples(limit: int, seed: int, input_rate: int) -> Iterator[Sample]:
    """VCTK 48 kHz clean, downsampled to ``input_rate`` — narrowband → fullband task."""
    ds = _stream(VCTK_DATASET, VCTK_REVISION, VCTK_SPLIT)
    for i, row in enumerate(ds):
        if i >= limit:
            break
        clean, sr = _extract_audio(row["audio"])
        clean = _peak_norm(clean)
        narrow = kaiser_resample(clean, sr, input_rate)
        uid = f"{row.get('speaker_id', 'spk')}_{row.get('text_id', i)}"
        yield Sample(uid, narrow, input_rate, clean, sr)


def enhance_samples(limit: int, seed: int, snr_db: float = 10.0) -> Iterator[Sample]:
    """VCTK 48 kHz clean + seeded compound degradation (noise + reverb + lowpass).

    ``snr_db`` alternates 5/15 dB across items so the set spans mild and severe
    corruption without a second pass. Clean 48 kHz is the reference.
    """
    ds = _stream(VCTK_DATASET, VCTK_REVISION, VCTK_SPLIT)
    for i, row in enumerate(ds):
        if i >= limit:
            break
        clean, sr = _extract_audio(row["audio"])
        clean = _peak_norm(clean)
        rng = np.random.default_rng(seed + i)
        this_snr = 5.0 if i % 2 else 15.0
        wet = _synthetic_reverb(rng, clean, sr, rt60=0.35)
        noise = _colored_noise(rng, wet.shape[0], exponent=1.0)
        noisy = _mix_at_snr(wet, noise, this_snr)
        degraded = _lowpass(noisy, sr, cutoff_hz=4000.0)
        degraded = _peak_norm(degraded)
        uid = f"{row.get('speaker_id', 'spk')}_{row.get('text_id', i)}"
        yield Sample(uid, degraded, sr, clean, sr)


def iter_samples(kind: str, limit: int, seed: int, input_rate: int) -> Iterator[Sample]:
    """Dispatch to the per-track iterator."""
    if kind == "denoise":
        return denoise_samples(limit, seed)
    if kind == "sr":
        return sr_samples(limit, seed, input_rate)
    if kind == "enhance":
        return enhance_samples(limit, seed)
    raise ValueError(f"unknown track kind {kind!r}")


#: Provenance recorded into every summary so a table can never lose its source.
DATASET_PROVENANCE = {
    "denoise": {"dataset": DENOISE_DATASET, "revision": DENOISE_REVISION, "split": DENOISE_SPLIT},
    "sr": {"dataset": VCTK_DATASET, "revision": VCTK_REVISION, "split": VCTK_SPLIT},
    "enhance": {"dataset": VCTK_DATASET, "revision": VCTK_REVISION, "split": VCTK_SPLIT,
                "degradation": "seeded colored-noise + synthetic reverb + 4 kHz lowpass"},
}
