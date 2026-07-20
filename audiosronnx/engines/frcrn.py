"""FRCRN denoise adapter: ``frcrn``.

FRCRN (*Frequency Recurrent CRN*, Alibaba / ClearerVoice-Studio) is the strongest-scoring
denoiser here on the standard benchmark — **PESQ 3.23 / STOI 0.95 / SI-SDR 19.22** on
VoiceBank+DEMAND, and 2nd place in the 2022 DNS Challenge. It runs at 16 kHz and predicts
a complex mask through two stacked UNets with frequency-recurrent layers.

Unlike the other denoise engines here it is **not** a per-frame streaming graph: the whole
model, including its STFT and ISTFT, exports as one waveform-to-waveform graph with a
dynamic length axis::

    noisy[1, T]  ->  enhanced[1, T]

That is possible because its ``ConvSTFT``/``ConviSTFT`` are Conv1d layers with Fourier
kernels rather than ``torch.stft``, so they are ordinary convolutions to onnxruntime and no
spectral op has to be replicated outside the graph.

What *does* live outside the graph is upstream's two-stage RMS normalisation and its
padding rule, both reproduced here in numpy — the model is sensitive to input level, and
the padding changes how much context the recurrence sees, so skipping either measurably
changes the output.

ONNX artifact: ``TigreGotico/audiosronnx-frcrn`` (Apache-2.0).

Reference
---------
- https://github.com/modelscope/ClearerVoice-Studio  (Apache-2.0)
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

from audiosronnx._kaiser import kaiser_resample
from audiosronnx.base import Denoiser, EngineEntry, register_engine
from audiosronnx.resolver import resolve

_HF_REPO = "TigreGotico/audiosronnx-frcrn"
_HF_REVISION: Optional[str] = "8d80ae92591dce122c8d59b02886095cd4192c29"
_ONNX = "frcrn.onnx"

_SR = 16000
_EPS = 1e-6
_TARGET_DB = -25.0            # upstream normalises to a -25 dBFS RMS target
_DECODE_WINDOW = 1.0          # seconds; upstream `decode_window`
_SEGMENT_AFTER_S = 120.0      # upstream `one_time_decode_length`


def _audio_norm(x: np.ndarray) -> Tuple[np.ndarray, float]:
    """Upstream's two-stage RMS normalisation; returns the scaled signal and its inverse.

    The second stage re-measures RMS over only the above-average-power samples, so quiet
    passages do not drag the level up. The returned scalar restores the original loudness
    after enhancement.
    """
    target = 10.0 ** (_TARGET_DB / 20.0)
    x = np.asarray(x, dtype=np.float64)
    scalar = target / (float((x ** 2).mean() ** 0.5) + _EPS)
    x = x * scalar

    power = x ** 2
    loud = power[power > power.mean()]
    rms_loud = float(loud.mean() ** 0.5) if loud.size else 0.0
    scalar_x = target / (rms_loud + _EPS)
    return x * scalar_x, 1.0 / (scalar * scalar_x + _EPS)


def _pad_for_decode(x: np.ndarray, sample_rate: int) -> np.ndarray:
    """Upstream's zero-padding rule, which fixes how much context the model sees."""
    window = int(sample_rate * _DECODE_WINDOW)
    stride = int(window * 0.75)
    t = x.size
    if t < window:
        pad = window - t
    elif t < window + stride:
        pad = window + stride - t
    elif (t - window) % stride:
        pad = t - (t - window) // stride * stride
    else:
        pad = 0
    return np.concatenate([x, np.zeros(pad, dtype=x.dtype)]) if pad else x


class FRCRNAdapter(Denoiser):
    """FRCRN 16 kHz denoiser — one waveform-to-waveform ONNX graph.

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
        onnx_path: Optional[str] = None,
        **cfg,
    ):
        super().__init__(**cfg)
        self._providers = providers
        self._cache_dir = cache_dir
        self._revision = revision if revision is not None else _HF_REVISION
        self._onnx_path = onnx_path
        self._sess = None

    def _ensure_models(self) -> None:
        if self._sess is not None:
            return
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = os.cpu_count() or 4
        path = self._onnx_path or resolve(
            _ONNX, hf_repo=_HF_REPO, revision=self._revision, cache_dir=self._cache_dir)
        self._sess = ort.InferenceSession(
            path, sess_options=opts, providers=self._providers or ["CPUExecutionProvider"])

    def _run(self, x: np.ndarray) -> np.ndarray:
        name = self._sess.get_inputs()[0].name
        out = self._sess.run(None, {name: np.ascontiguousarray(x[None], dtype=np.float32)})[0]
        return np.asarray(out, dtype=np.float32).reshape(-1)

    def _denoise_array(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        self._ensure_models()
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sample_rate != _SR:
            x = np.asarray(kaiser_resample(x, sample_rate, _SR), dtype=np.float32)
        n_out = x.size
        if n_out == 0:
            return np.zeros(0, dtype=np.float32)

        peak = float(np.max(np.abs(x)))
        if not np.isfinite(peak) or peak <= 0:
            return np.zeros(n_out, dtype=np.float32)

        normed, scalar = _audio_norm(x)
        padded = _pad_for_decode(normed, _SR)

        window = int(_SR * _DECODE_WINDOW)
        stride = int(window * 0.75)
        if padded.size > _SR * _SEGMENT_AFTER_S:
            # very long input: slide a window and drop the transient edges of each segment
            give_up = (window - stride) // 2
            out = np.zeros(padded.size, dtype=np.float32)
            idx = 0
            while idx + window <= padded.size:
                seg = self._run(padded[idx:idx + window])
                if idx == 0:
                    out[idx:idx + window - give_up] = seg[:window - give_up]
                else:
                    out[idx + give_up:idx + window - give_up] = seg[give_up:window - give_up]
                idx += stride
        else:
            out = self._run(padded)

        out = out[:n_out] * scalar
        result = np.zeros(n_out, dtype=np.float32)
        result[:out.size] = out
        return result


register_engine(EngineEntry(
    alias="frcrn",
    adapter_class=FRCRNAdapter,
    description=(
        "FRCRN (ClearerVoice-Studio): 16 kHz complex-mask denoiser, two stacked UNets "
        "with frequency-recurrent layers. Highest benchmark scores of the shipped "
        "denoisers (PESQ 3.23 on VoiceBank+DEMAND). One waveform-to-waveform ONNX graph. "
        "ONNX from TigreGotico/audiosronnx-frcrn. (Alibaba, Apache-2.0)"
    ),
    input_sample_rate=_SR,
    output_sample_rate=_SR,
    license="Apache-2.0",
    extras="",
    kind="denoise",
))
