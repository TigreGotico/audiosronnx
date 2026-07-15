"""HiFi-GAN+ bandwidth-extension adapter for audiosronnx.

HiFi-GAN+ (Su et al., ICASSP 2021; brentspell/hifi-gan-bwe) upsamples band-limited
speech to 48 kHz and reconstructs the missing high band with a non-causal WaveNet
(stacked gated dilated 1-D convolutions). It is a time-domain model — no spectral
front-end — so the whole neural part is one ONNX graph.

The adapter does bandlimited (kaiser) interpolation to 48 kHz in numpy, reflect-free
zero-padding by half the receptive field, then runs the WaveNet ONNX graph and a
``tanh``. Input may be any sample rate; output is 48 kHz.

ONNX artifact: ``TigreGotico/audiosronnx-hifiganbwe`` (MIT).

Reference
---------
- https://github.com/brentspell/hifi-gan-bwe
- https://pixl.cs.princeton.edu/pubs/Su_2021_BEI/ICASSP2021_Su_Wang_BWE.pdf
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from audiosronnx._kaiser import kaiser_resample
from audiosronnx.base import EngineEntry, SRModel, register_engine
from audiosronnx.resolver import resolve

_HF_REPO = "TigreGotico/audiosronnx-hifiganbwe"
_HF_REVISION = "d617931bce0671703b0ac0fbe14cf37d243e5f8c"
_MODEL = "hifiganbwe_wavenet.onnx"
_OUT_SR = 48000
# WaveNet receptive field for the shipped model (2 stacks x 8 layers, dilation base 3,
# kernel 3): (3-1) * 2 * sum(3**i for i in range(8)) = 13120.
_RECEPTIVE_FIELD = 13120


class HiFiGANBWEAdapter(SRModel):
    """Bandwidth-extension adapter backed by the HiFi-GAN+ WaveNet ONNX graph.

    Parameters
    ----------
    providers:
        onnxruntime execution providers (default CPU).
    cache_dir:
        Override the model download cache directory.
    """

    input_sample_rate = 0  # flexible; internally upsampled to 48 kHz
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
        up = kaiser_resample(audio, sample_rate, _OUT_SR)
        pad = _RECEPTIVE_FIELD // 2
        p = np.pad(up, (pad, pad))[None, None, :].astype(np.float32)
        y = self._sess.run(None, {self._sess.get_inputs()[0].name: p})[0]
        y = np.tanh(y)[0, 0]
        return y[pad:-pad].astype(np.float32)


register_engine(
    EngineEntry(
        alias="hifiganbwe",
        adapter_class=HiFiGANBWEAdapter,
        description=(
            "HiFi-GAN+: non-causal WaveNet bandwidth extension, any input -> 48 kHz "
            "with bandlimited (kaiser) interpolation. ~1M params, time-domain. "
            "ONNX from TigreGotico/audiosronnx-hifiganbwe. (Su et al., ICASSP 2021, MIT)"
        ),
        input_sample_rate=0,
        output_sample_rate=_OUT_SR,
        license="MIT",
        extras="",
    )
)
