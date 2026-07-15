"""AP-BWE adapter for audiosronnx.

AP-BWE (Lu et al.; yxlu-0102/AP-BWE) is an amplitude-phase bandwidth extender: two
parallel ConvNeXt stacks take the log-amplitude and phase spectra of a band-limited
signal and predict the wideband log-amplitude and phase, which are recombined and
inverted back to a 48 kHz waveform. The neural part is one ONNX graph
``(log_amp, pha) -> (log_amp_wb, pha_wb)``; STFT/ISTFT run in numpy.

The shipped model is the ``12kto48k`` checkpoint: it treats the input as ~12 kHz-band
content, so an input at any rate is bandlimited to 12 kHz and interpolated to 48 kHz
before analysis. Output is 48 kHz.

ONNX artifact: ``TigreGotico/audiosronnx-apbwe`` (MIT).

Reference
---------
- https://github.com/yxlu-0102/AP-BWE
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from audiosronnx._kaiser import kaiser_resample
from audiosronnx._stft import istft, stft
from audiosronnx.base import EngineEntry, SRModel, register_engine
from audiosronnx.resolver import resolve

_HF_REPO = "TigreGotico/audiosronnx-apbwe"
_HF_REVISION = "22d116972f09802273d9e26851f004da0bfbd58a"
_MODEL = "apbwe.onnx"
_OUT_SR = 48000
_LR_SR = 12000
_N_FFT = 1024
_HOP = 80
_WIN = 320


class APBWEAdapter(SRModel):
    """Amplitude-phase bandwidth-extension adapter backed by the AP-BWE ONNX graph.

    Parameters
    ----------
    providers:
        onnxruntime execution providers (default CPU).
    cache_dir:
        Override the model download cache directory.
    """

    input_sample_rate = 0  # flexible; bandlimited to 12 kHz then interpolated to 48 kHz
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
        # bandlimit to the model's low rate, then interpolate up to 48 kHz
        lr = kaiser_resample(audio, sample_rate, _LR_SR)
        nb = kaiser_resample(lr, _LR_SR, _OUT_SR)

        spec = stft(nb, _N_FFT, _HOP, _WIN)          # complex [F, T]
        log_amp = np.log(np.abs(spec) + 1e-4).astype(np.float32)[None]
        pha = np.angle(spec).astype(np.float32)[None]

        names = [i.name for i in self._sess.get_inputs()]
        out = self._sess.run(None, {names[0]: log_amp, names[1]: pha})
        log_amp_wb, pha_wb = out[0][0], out[1][0]

        amp = np.exp(log_amp_wb)
        com = (amp * np.cos(pha_wb) + 1j * amp * np.sin(pha_wb)).astype(np.complex64)
        wav = istft(com, _N_FFT, _HOP, _WIN, length=nb.shape[0])
        return wav.astype(np.float32)


register_engine(
    EngineEntry(
        alias="apbwe",
        adapter_class=APBWEAdapter,
        description=(
            "AP-BWE: dual-ConvNeXt amplitude-phase bandwidth extension in the STFT "
            "domain, input -> 48 kHz (12 kHz-band model). Highest LSD accuracy of the "
            "shipped engines. ONNX from TigreGotico/audiosronnx-apbwe. (Lu et al., MIT)"
        ),
        input_sample_rate=0,
        output_sample_rate=_OUT_SR,
        license="MIT",
        extras="",
    )
)
