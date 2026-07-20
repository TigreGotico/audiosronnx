"""GTCRN denoise adapter: ``gtcrn``.

GTCRN (*Grouped Temporal Convolutional Recurrent Network*, Rong et al.) is an
ultra-lightweight 16 kHz speech denoiser — **23.7 K parameters, 33 MMACs/s**, roughly
half a megabyte of ONNX. It is by far the smallest engine here, which makes it the one to
reach for on embedded/on-device targets where DeepFilterNet or DPDFNet are too heavy.

The published graph is the streaming variant: it consumes **one STFT frame at a time** and
threads three recurrent caches through the call, so the model itself is stateful while the
graph stays static::

    mix[1, 257, 1, 2], conv_cache, tra_cache, inter_cache
        -> enh[1, 257, 1, 2], conv_cache_out, tra_cache_out, inter_cache_out

Everything model-specific — the ERB filterbank, subband feature extraction, the grouped
temporal recurrent blocks — lives inside the graph, so the only work outside it is a
sqrt-Hann STFT/ISTFT, which runs in numpy here. Cache shapes and the STFT geometry are read
back from the graph's own metadata rather than hard-coded.

This reproduces the upstream offline recipe, which is a plain centred ``torch.stft`` /
``torch.istft`` pair with a ``hann_window(512).pow(0.5)`` window::

    input = torch.stft(mix, 512, 256, 512, torch.hann_window(512).pow(0.5))
    enh   = torch.istft(model(input[None])[0], 512, 256, 512, torch.hann_window(512).pow(0.5))

ONNX artifact: ``TigreGotico/audiosronnx-gtcrn`` (MIT).

Reference
---------
- https://github.com/Xiaobin-Rong/gtcrn  (MIT)
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from audiosronnx._kaiser import kaiser_resample
from audiosronnx._stft import istft, stft
from audiosronnx.base import Denoiser, EngineEntry, register_engine
from audiosronnx.resolver import resolve

_HF_REPO = "TigreGotico/audiosronnx-gtcrn"
_HF_REVISION: Optional[str] = "eeba5e9636e7e0aad93b25fad17471a9646d2640"
_ONNX = "gtcrn_simple.onnx"

_SR = 16000
_N_FFT = 512
_HOP = 256


def _sqrt_hann(n: int) -> np.ndarray:
    """``torch.hann_window(n).pow(0.5)`` — periodic Hann, square-rooted."""
    k = np.arange(n, dtype=np.float64)
    return np.sqrt(0.5 * (1.0 - np.cos(2.0 * np.pi * k / n)))


class GTCRNAdapter(Denoiser):
    """GTCRN 16 kHz denoiser, driven frame-by-frame through one stateful ONNX graph.

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

    def _zero_caches(self) -> dict:
        """Fresh zero caches, shaped from the graph's own declared inputs."""
        return {
            i.name: np.zeros([int(d) for d in i.shape], dtype=np.float32)
            for i in self._sess.get_inputs()[1:]
        }

    def _denoise_array(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        self._ensure_models()
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sample_rate != _SR:
            x = np.asarray(kaiser_resample(x, sample_rate, _SR), dtype=np.float32)
        n_out = x.size
        if n_out == 0:
            return np.zeros(0, dtype=np.float32)

        window = _sqrt_hann(_N_FFT)
        spec = stft(x, n_fft=_N_FFT, hop_size=_HOP, win_size=_N_FFT,
                    center=True, window=window)            # [F, T] complex
        noisy = np.stack([spec.real, spec.imag], axis=-1).astype(np.float32)  # [F,T,2]

        mix_name = self._sess.get_inputs()[0].name
        out_names = [o.name for o in self._sess.get_outputs()]
        caches = self._zero_caches()
        cache_in = list(caches)
        enhanced = np.empty_like(noisy)
        for t in range(noisy.shape[1]):
            feeds = {mix_name: np.ascontiguousarray(noisy[:, t: t + 1][None])}
            feeds.update(caches)
            enh, *new_caches = self._sess.run(out_names, feeds)
            enhanced[:, t] = enh[0, :, 0]
            caches = dict(zip(cache_in, new_caches))

        wav = istft((enhanced[..., 0] + 1j * enhanced[..., 1]), n_fft=_N_FFT,
                    hop_size=_HOP, win_size=_N_FFT, center=True, length=n_out,
                    window=window)
        return np.ascontiguousarray(wav, dtype=np.float32)


register_engine(EngineEntry(
    alias="gtcrn",
    adapter_class=GTCRNAdapter,
    description=(
        "GTCRN: ultra-light 16 kHz speech denoiser (23.7 K params, 33 MMACs, ~0.5 MB "
        "ONNX) as a single stateful graph; sqrt-Hann STFT in numpy. The smallest engine "
        "here — for embedded / on-device use. ONNX from TigreGotico/audiosronnx-gtcrn. "
        "(Rong et al., MIT)"
    ),
    input_sample_rate=_SR,
    output_sample_rate=_SR,
    license="MIT",
    extras="",
    kind="denoise",
))
