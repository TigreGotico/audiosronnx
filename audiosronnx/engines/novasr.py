"""NovaSR adapter for audiosronnx.

NovaSR (``YatharthS/NovaSR``, Apache-2.0) is a tiny (~52 KB) conv1d / BigVGAN-snake
generator that upsamples 16 kHz speech to 48 kHz in a single time-domain pass. There
is no spectral front-end: the whole model is one ONNX graph that maps a 16 kHz
waveform ``[1, 1, T]`` to a 48 kHz waveform ``[1, 1, 3T]``.

It is extremely fast and memory-light, at lower fidelity than LavaSR. Input other than
16 kHz is resampled to 16 kHz first.

ONNX artifact: ``TigreGotico/audiosronnx-novasr`` (Apache-2.0).

Reference
---------
- https://github.com/ysharma3501/NovaSR
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from audiosronnx.audio import resample
from audiosronnx.base import EngineEntry, SRModel, register_engine
from audiosronnx.resolver import resolve

_HF_REPO = "TigreGotico/audiosronnx-novasr"
_HF_REVISION = "5d92bf1488aae6d92356ef5951c7ef9bf9f9801b"
_MODEL = "novasr.onnx"
_IN_SR = 16000
_OUT_SR = 48000


class NovaSRAdapter(SRModel):
    """Super-resolution adapter backed by the NovaSR ONNX graph.

    Parameters
    ----------
    providers:
        onnxruntime execution providers (default CPU).
    cache_dir:
        Override the model download cache directory.
    """

    input_sample_rate = _IN_SR
    output_sample_rate = _OUT_SR

    def __init__(
        self,
        *,
        providers: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
        revision: Optional[str] = None,
        **cfg,
    ):
        super().__init__(**cfg)
        self._providers = providers
        self._cache_dir = cache_dir
        self._revision = revision or _HF_REVISION
        self._sess = None

    def _ensure_models(self) -> None:
        if self._sess is not None:
            return
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = os.cpu_count() or 4
        path = resolve(
            _MODEL, hf_repo=_HF_REPO, revision=self._revision, cache_dir=self._cache_dir
        )
        self._sess = ort.InferenceSession(
            path, sess_options=opts, providers=self._providers or ["CPUExecutionProvider"]
        )

    def _upscale_array(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        self._ensure_models()
        x = resample(audio, sample_rate, _IN_SR)
        inp = x[None, None, :].astype(np.float32)  # [1, 1, T]
        out = self._sess.run(None, {self._sess.get_inputs()[0].name: inp})[0]
        return np.asarray(out).reshape(-1).astype(np.float32)


register_engine(
    EngineEntry(
        alias="novasr",
        adapter_class=NovaSRAdapter,
        description=(
            "NovaSR: ~52 KB conv1d/BigVGAN-snake generator, 16 kHz -> 48 kHz in a "
            "single time-domain pass. Extremely fast and light, lower fidelity than "
            "LavaSR. ONNX from TigreGotico/audiosronnx-novasr."
        ),
        input_sample_rate=_IN_SR,
        output_sample_rate=_OUT_SR,
        license="Apache-2.0",
        extras="",
    )
)
