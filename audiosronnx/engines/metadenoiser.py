"""Meta Denoiser adapter: ``metadenoiser``.

The Facebook/Meta Research *denoiser* (Defossez et al.) is a **causal Demucs** working
directly on the waveform: a convolutional encoder/decoder around an LSTM bottleneck, with
no spectral front-end at all. It is the only time-domain denoiser here — every other one
masks or predicts a spectrum.

The graph is waveform-to-waveform at 16 kHz::

    noisy[1, 1, 160000]  ->  enhanced[1, 1, 160000]

Amplitude normalisation and the internal length padding live inside the model's own
``forward``, so the graph is self-contained and the adapter only has to window the input.

**License:** the upstream weights are **CC-BY-NC-4.0** — research and non-commercial use.
That covers the model, not audio processed with it. Every other denoise engine here is
MIT or Apache-2.0, so reach for this one deliberately.

Fixed window
------------
Demucs computes its padding from the input length with Python arithmetic, which the tracer
bakes in: a dynamic-length export is correct only at the length it was traced at. The
published graphs therefore take a **fixed 10 s window**, and this adapter slides that window
with a crossfaded overlap, the same approach ``dpdfnet`` uses for long audio.

ONNX artifact: ``TigreGotico/audiosronnx-metadenoiser`` (CC-BY-NC-4.0).

Reference
---------
- https://github.com/facebookresearch/denoiser  (CC-BY-NC-4.0)
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from audiosronnx._kaiser import kaiser_resample
from audiosronnx.base import Denoiser, EngineEntry, register_engine
from audiosronnx.resolver import resolve

_HF_REPO = "TigreGotico/audiosronnx-metadenoiser"
_HF_REVISION: Optional[str] = "9a648aa278202de97d909dc82595e0899ccf4c66"
#: ``model`` alias -> onnx filename. dns64 is the larger, higher-quality variant.
_MODELS: Dict[str, str] = {"dns64": "dns64.onnx", "dns48": "dns48.onnx"}
_DEFAULT_MODEL = "dns64"

_SR = 16000
_WINDOW = 160000              # 10 s — the length the graphs were traced at
_OVERLAP = 16000              # 1 s crossfade between windows


class MetaDenoiserAdapter(Denoiser):
    """Meta's causal-Demucs waveform denoiser, run over a fixed 10 s window.

    Parameters
    ----------
    model:
        ``"dns64"`` (default, 33.5M params) or ``"dns48"`` (18.9M, roughly half the size).
    providers:
        onnxruntime execution providers (default CPU).
    cache_dir:
        Override the model download cache directory.
    """

    input_sample_rate = _SR

    def __init__(
        self,
        *,
        model: str = _DEFAULT_MODEL,
        providers: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
        revision: Optional[str] = None,
        onnx_path: Optional[str] = None,
        **cfg,
    ):
        super().__init__(**cfg)
        if model not in _MODELS:
            raise ValueError(
                f"unknown metadenoiser model {model!r}; choose one of {sorted(_MODELS)}")
        self._model = model
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
            _MODELS[self._model], hf_repo=_HF_REPO, revision=self._revision,
            cache_dir=self._cache_dir)
        self._sess = ort.InferenceSession(
            path, sess_options=opts, providers=self._providers or ["CPUExecutionProvider"])

    def _run_window(self, chunk: np.ndarray) -> np.ndarray:
        """Enhance exactly one ``_WINDOW``-long chunk, zero-padding a short tail."""
        padded = chunk
        if padded.size < _WINDOW:
            padded = np.pad(padded, (0, _WINDOW - padded.size))
        out = self._sess.run(
            None, {"noisy": np.ascontiguousarray(padded[None, None], dtype=np.float32)})[0]
        return out.reshape(-1)[:chunk.size] if chunk.size < _WINDOW else out.reshape(-1)

    def _denoise_array(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        self._ensure_models()
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sample_rate != _SR:
            x = np.asarray(kaiser_resample(x, sample_rate, _SR), dtype=np.float32)
        n_out = x.size
        if n_out == 0:
            return np.zeros(0, dtype=np.float32)
        if not np.isfinite(x).all():
            # the model normalises by the signal's own std, so a single non-finite sample
            # would poison the whole window
            return np.zeros(n_out, dtype=np.float32)

        if n_out <= _WINDOW:
            return np.ascontiguousarray(self._run_window(x)[:n_out], dtype=np.float32)

        # slide the fixed window, crossfading the overlap so seams do not click
        stride = _WINDOW - _OVERLAP
        out = np.zeros(n_out, dtype=np.float64)
        weight = np.zeros(n_out, dtype=np.float64)
        ramp = np.linspace(0.0, 1.0, _OVERLAP, dtype=np.float64)
        start = 0
        while start < n_out:
            chunk = x[start:start + _WINDOW]
            enhanced = self._run_window(chunk)[:chunk.size]
            envelope = np.ones(chunk.size, dtype=np.float64)
            if start > 0:
                envelope[:min(_OVERLAP, chunk.size)] = ramp[:min(_OVERLAP, chunk.size)]
            if start + _WINDOW < n_out:
                envelope[-_OVERLAP:] = ramp[::-1]
            out[start:start + chunk.size] += enhanced * envelope
            weight[start:start + chunk.size] += envelope
            start += stride

        return np.ascontiguousarray(
            out / np.where(weight > 1e-8, weight, 1.0), dtype=np.float32)


register_engine(EngineEntry(
    alias="metadenoiser",
    adapter_class=MetaDenoiserAdapter,
    description=(
        "Meta Denoiser (Defossez et al.): causal Demucs operating directly on the "
        "waveform — the only time-domain denoiser here. dns64 / dns48 variants, run over "
        "a fixed 10 s window. Weights are CC-BY-NC-4.0. ONNX from "
        "TigreGotico/audiosronnx-metadenoiser."
    ),
    input_sample_rate=_SR,
    output_sample_rate=_SR,
    license="CC-BY-NC-4.0",
    extras="",
    kind="denoise",
))
