"""CMGAN denoise adapter: ``cmgan``.

CMGAN (*Conformer-based Metric GAN*, Cao et al.) denoises in the complex STFT domain: a
dense encoder feeds four two-stage conformer blocks that attend along time and frequency in
turn, and two decoders emit a magnitude mask and a complex residual. Combining a masked
magnitude with an additive complex term lets it correct phase, which a magnitude-only mask
cannot.

At **1.83 M parameters / 7.8 MB** it is among the smallest engines here, and the only one
built on conformers.

The graph is a clean spectral transform, with the STFT outside it::

    spec[1, 2, T, F]  ->  enhanced[1, 2, T, F]      (real, imag)

Magnitudes are **power-compressed** by 0.3 before the model and decompressed after. Upstream
also RMS-normalises the utterance and pads it to a whole number of hops by wrapping its own
start — both reproduced here, since the model is level-sensitive.

Only the generator is exported. CMGAN's discriminator exists to supply the metric-GAN
training loss and has no role at inference.

Being a *metric* GAN, it optimises PESQ rather than waveform fidelity, and the two
diverge: on speech at 11 dB input SNR it gains **+0.76 PESQ while losing 1.5 dB SNR**. It
makes audio sound better and match the reference waveform less. Every engine here trained
on an SNR-shaped objective gains on both, so reach for cmgan only when perceptual quality
is the target and waveform fidelity is not.

ONNX artifact: ``TigreGotico/audiosronnx-cmgan`` (MIT).

Export note
-----------
``TSCNet.forward`` derives the noisy phase with ``torch.angle(torch.complex(re, im))``, and
``torch.complex`` has no ONNX operator. The exported graph rewrites that as the identical
``torch.atan2(im, re)``; the rewrite is bit-for-bit equal to the original.

Reference
---------
- https://github.com/ruizhecao96/CMGAN  (MIT)
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from audiosronnx._kaiser import kaiser_resample
from audiosronnx._stft import hamming_window, istft, stft
from audiosronnx.base import Denoiser, EngineEntry, register_engine
from audiosronnx.resolver import resolve

_HF_REPO = "TigreGotico/audiosronnx-cmgan"
_HF_REVISION: Optional[str] = "fbd39dc56f3dcaae3934873a2a8c3bf8ef1a1d71"
_ONNX = "cmgan.onnx"

_SR = 16000
_N_FFT = 400
_HOP = 100
_COMPRESS = 0.3


def _compress(spec: np.ndarray, power: float) -> np.ndarray:
    """Raise magnitudes to ``power`` while preserving phase."""
    mag = np.abs(spec) ** power
    ang = np.angle(spec)
    return mag * np.cos(ang) + 1j * mag * np.sin(ang)


class CMGANAdapter(Denoiser):
    """CMGAN 16 kHz conformer denoiser — masked magnitude plus a complex residual.

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

    def _denoise_array(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        self._ensure_models()
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sample_rate != _SR:
            x = np.asarray(kaiser_resample(x, sample_rate, _SR), dtype=np.float32)
        n_out = x.size
        if n_out == 0:
            return np.zeros(0, dtype=np.float32)

        x64 = np.asarray(x, dtype=np.float64)
        energy = float((x64 ** 2).sum())
        if not np.isfinite(energy) or energy <= 0.0:
            return np.zeros(n_out, dtype=np.float32)
        norm = np.sqrt(x64.size / energy)
        x64 = x64 * norm

        # upstream pads to a whole number of hops by wrapping the clip's own start
        pad = (-x64.size) % _HOP
        if pad:
            x64 = np.concatenate([x64, x64[:pad]])

        window = hamming_window(_N_FFT)
        spec = stft(x64, n_fft=_N_FFT, hop_size=_HOP, win_size=_N_FFT,
                    center=True, window=window)                   # [F, T]
        comp = _compress(spec, _COMPRESS)
        feed = np.stack([comp.real, comp.imag])[None].transpose(0, 1, 3, 2)  # [1,2,T,F]

        out = self._sess.run(None, {"spec": np.ascontiguousarray(feed, dtype=np.float32)})[0]
        est = out[0, 0].T + 1j * out[0, 1].T                       # [F, T]
        enhanced = _compress(est, 1.0 / _COMPRESS)

        wav = istft(enhanced, n_fft=_N_FFT, hop_size=_HOP, win_size=_N_FFT,
                    center=True, length=x64.size, window=window)
        return np.ascontiguousarray(wav[:n_out] / norm, dtype=np.float32)


register_engine(EngineEntry(
    alias="cmgan",
    adapter_class=CMGANAdapter,
    description=(
        "CMGAN (Cao et al.): conformer-based metric-GAN denoiser, 16 kHz. Masked magnitude "
        "plus an additive complex residual, so it corrects phase rather than reusing it. "
        "1.83M params / 7.8 MB. ONNX from TigreGotico/audiosronnx-cmgan. (MIT)"
    ),
    input_sample_rate=_SR,
    output_sample_rate=_SR,
    license="MIT",
    extras="",
    kind="denoise",
))
