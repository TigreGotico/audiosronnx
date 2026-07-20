"""VoiceFixer restoration adapter: ``voicefixer``.

VoiceFixer (Liu et al.) is general speech **restoration** at 44.1 kHz. Where the other
engines each target one defect, this one is trained to undo noise, reverberation, clipping
and bandwidth loss *together* — which makes it the engine to reach for when a recording is
damaged in several ways at once and you do not want to chain three others.

Two graphs, the same shape as :mod:`~audiosronnx.engines.sidon` — a predictor followed by a
neural vocoder::

    analysis : mel_orig[1, 1, 501, 128]  ->  mel[1, 1, 501, 128]
    vocoder  : mel[1, 1, 501, 128]       ->  wav[1, 1, 501*441]

The STFT (n_fft 2048, hop 441), the mel projection and the ``10**clip(x, max=5)`` log
inverse all run in numpy here, so nothing pulls torch at inference.

Being generative, it *resynthesises* speech rather than filtering it. Treat the result as
enhanced-real: good to listen to and reasonable as an acoustic-model target, but the detail
it restores is invented, not recovered.

Fixed window
------------
The analysis graph is length-specialised, so it takes a fixed **5 s (501-frame)** window and
this adapter slides it with a crossfaded overlap. Upstream's own ``restore_inmem`` segments
long audio too.

ONNX artifact: ``TigreGotico/audiosronnx-voicefixer`` (MIT).

Reference
---------
- https://github.com/haoheliu/voicefixer  (MIT)
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from audiosronnx._kaiser import kaiser_resample
from audiosronnx._stft import istft, stft
from audiosronnx.base import Denoiser, EngineEntry, register_engine
from audiosronnx.resolver import resolve

_HF_REPO = "TigreGotico/audiosronnx-voicefixer"
_HF_REVISION: Optional[str] = "aa259fa578b9d5f4c76fda3cd967f73542b0e15b"
_ANALYSIS = "analysis.onnx"
_VOCODER = "vocoder.onnx"
_MEL_FILTERS = "mel_filters.npy"

_SR = 44100
_N_FFT = 2048
_HOP = 441
_N_MELS = 128
_FRAMES = 501                 # the window the analysis graph was traced at
_OVERLAP_FRAMES = 100         # crossfade between windows
_WINDOW_SAMPLES = _FRAMES * _HOP
_LOG_CLIP = 5.0               # upstream clips the log-mel before 10**x


class VoiceFixerAdapter(Denoiser):
    """VoiceFixer 44.1 kHz general restoration, run over a fixed 5 s window.

    Parameters
    ----------
    providers:
        onnxruntime execution providers (default CPU).
    cache_dir:
        Override the model download cache directory.
    """

    input_sample_rate = _SR

    def __init__(
        self,
        *,
        providers: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
        revision: Optional[str] = None,
        analysis_path: Optional[str] = None,
        vocoder_path: Optional[str] = None,
        mel_filters_path: Optional[str] = None,
        **cfg,
    ):
        super().__init__(**cfg)
        self._providers = providers
        self._cache_dir = cache_dir
        self._revision = revision if revision is not None else _HF_REVISION
        self._analysis_path = analysis_path
        self._vocoder_path = vocoder_path
        self._mel_filters_path = mel_filters_path
        self._analysis = None
        self._vocoder = None
        self._mel_fb: Optional[np.ndarray] = None

    def _ensure_models(self) -> None:
        if self._analysis is not None:
            return
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = os.cpu_count() or 4
        providers = self._providers or ["CPUExecutionProvider"]

        def _fetch(explicit, name):
            return explicit or resolve(
                name, hf_repo=_HF_REPO, revision=self._revision, cache_dir=self._cache_dir)

        self._analysis = ort.InferenceSession(
            _fetch(self._analysis_path, _ANALYSIS), sess_options=opts, providers=providers)
        self._vocoder = ort.InferenceSession(
            _fetch(self._vocoder_path, _VOCODER), sess_options=opts, providers=providers)
        self._mel_fb = np.load(_fetch(self._mel_filters_path, _MEL_FILTERS))  # [1025, 128]

    def _log_mel(self, wave: np.ndarray) -> np.ndarray:
        """Magnitude STFT projected onto the htk mel filterbank -> ``[1, 1, T, 128]``."""
        spec = stft(np.asarray(wave, dtype=np.float64), n_fft=_N_FFT, hop_size=_HOP,
                    win_size=_N_FFT, center=True)                     # [F, T]
        mag = np.abs(spec).T                                          # [T, F]
        return (mag @ self._mel_fb)[None, None]                       # [1, 1, T, mels]

    def _restore_window(self, wave: np.ndarray) -> np.ndarray:
        """Run one fixed-length window through analysis + vocoder."""
        padded = wave
        if padded.size < _WINDOW_SAMPLES:
            padded = np.pad(padded, (0, _WINDOW_SAMPLES - padded.size))

        mel_orig = self._log_mel(padded).astype(np.float32)
        if mel_orig.shape[2] != _FRAMES:                              # trim STFT edge frame
            mel_orig = mel_orig[:, :, :_FRAMES]
        predicted = self._analysis.run(None, {"mel_orig": mel_orig})[0]
        # upstream's from_log: 10 ** clip(x, max=5)
        decompressed = np.power(10.0, np.clip(predicted, None, _LOG_CLIP)).astype(np.float32)
        out = self._vocoder.run(None, {"mel": decompressed})[0].reshape(-1)
        return out[:wave.size] if wave.size < out.size else out

    def _denoise_array(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        self._ensure_models()
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sample_rate != _SR:
            x = np.asarray(kaiser_resample(x, sample_rate, _SR), dtype=np.float32)
        n_out = x.size
        if n_out == 0:
            return np.zeros(0, dtype=np.float32)
        if not np.isfinite(x).all():
            return np.zeros(n_out, dtype=np.float32)

        if n_out <= _WINDOW_SAMPLES:
            out = self._restore_window(x)
            result = np.zeros(n_out, dtype=np.float32)
            result[:min(n_out, out.size)] = out[:n_out]
            return result

        stride = _WINDOW_SAMPLES - _OVERLAP_FRAMES * _HOP
        acc = np.zeros(n_out, dtype=np.float64)
        weight = np.zeros(n_out, dtype=np.float64)
        overlap = _OVERLAP_FRAMES * _HOP
        ramp = np.linspace(0.0, 1.0, overlap, dtype=np.float64)
        start = 0
        while start < n_out:
            chunk = x[start:start + _WINDOW_SAMPLES]
            restored = self._restore_window(chunk)[:chunk.size]
            envelope = np.ones(restored.size, dtype=np.float64)
            if start > 0:
                span = min(overlap, restored.size)
                envelope[:span] = ramp[:span]
            if start + _WINDOW_SAMPLES < n_out and restored.size >= overlap:
                envelope[-overlap:] = ramp[::-1]
            acc[start:start + restored.size] += restored * envelope
            weight[start:start + restored.size] += envelope
            start += stride

        return np.ascontiguousarray(
            acc / np.where(weight > 1e-8, weight, 1.0), dtype=np.float32)


register_engine(EngineEntry(
    alias="voicefixer",
    adapter_class=VoiceFixerAdapter,
    description=(
        "VoiceFixer (Liu et al.): general 44.1 kHz speech restoration — noise, reverb, "
        "clipping and bandwidth loss together, rather than any one of them. ResUNet mel "
        "predictor plus a TFGAN vocoder, over a fixed 5 s window. Generative, so the "
        "detail it restores is invented. ONNX from TigreGotico/audiosronnx-voicefixer. "
        "(MIT)"
    ),
    input_sample_rate=_SR,
    output_sample_rate=_SR,
    license="MIT",
    extras="",
    kind="enhance",
))
