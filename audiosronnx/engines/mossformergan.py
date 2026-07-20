"""MossFormerGAN denoise adapter: ``mossformergan``.

MossFormerGAN (Alibaba / ClearerVoice-Studio) pairs the MossFormer attention backbone with
a metric-GAN objective, and holds the **highest published PESQ of any model surveyed for
this library — 3.47** on VoiceBank+DEMAND.

Like :mod:`~audiosronnx.engines.cmgan` it predicts a masked magnitude plus an additive
complex residual, so it corrects phase rather than reusing the noisy phase::

    spec[1, 2, T, 201]  ->  enhanced[1, 2, T, 201]      (real, imag)

Only the generator is exported; the discriminator supplies the training loss.

Fixed window
------------
MossFormer's group attention reshapes the sequence into fixed-size groups, and that reshape
captures the traced length — a dynamic-length graph runs at its trace size and fails
elsewhere, even with constant folding disabled. The published graph therefore takes a fixed
**401-frame** window (about 2.5 s at 16 kHz with a 100-sample hop), and this adapter slides
it with a crossfaded overlap. Upstream's own decode segments long audio for the same reason.

ONNX artifact: ``TigreGotico/audiosronnx-mossformergan`` (Apache-2.0).

Export note
-----------
Two operators needed rewriting first, each bit-for-bit equivalent: ``torch.complex`` (no
ONNX operator) became ``atan2``, and ``torch.eye(dtype=bool)``, which exports to
``EyeLike(bool)`` and has no onnxruntime kernel, became an arange equality.

Reference
---------
- https://github.com/modelscope/ClearerVoice-Studio  (Apache-2.0)
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from audiosronnx._kaiser import kaiser_resample
from audiosronnx._stft import hamming_window, istft, stft
from audiosronnx.base import Denoiser, EngineEntry, register_engine
from audiosronnx.resolver import resolve

_HF_REPO = "TigreGotico/audiosronnx-mossformergan"
_HF_REVISION: Optional[str] = "5f471c9d92dda80c0883cc850227fd878b334b08"
_ONNX = "mossformergan.onnx"

_SR = 16000
_N_FFT = 400
_HOP = 100
_COMPRESS = 0.3
_FRAMES = 401                 # the window the graph was traced at
_OVERLAP_FRAMES = 80          # crossfade between windows


def _compress(spec: np.ndarray, power: float) -> np.ndarray:
    """Raise magnitudes to ``power`` while preserving phase."""
    mag = np.abs(spec) ** power
    ang = np.angle(spec)
    return mag * np.cos(ang) + 1j * mag * np.sin(ang)


class MossFormerGANAdapter(Denoiser):
    """MossFormerGAN 16 kHz denoiser, run over a fixed 401-frame window.

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

    def _run_window(self, block: np.ndarray) -> np.ndarray:
        """Enhance one ``[2, _FRAMES, F]`` block, zero-padding a short tail."""
        frames = block.shape[1]
        if frames < _FRAMES:
            block = np.pad(block, ((0, 0), (0, _FRAMES - frames), (0, 0)))
        out = self._sess.run(
            None, {"spec": np.ascontiguousarray(block[None], dtype=np.float32)})[0]
        return out[0][:, :frames]                      # [2, frames, F]

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
        pad = (-x64.size) % _HOP
        if pad:
            x64 = np.concatenate([x64, x64[:pad]])

        window = hamming_window(_N_FFT)
        spec = stft(x64, n_fft=_N_FFT, hop_size=_HOP, win_size=_N_FFT,
                    center=True, window=window)                        # [F, T]
        comp = _compress(spec, _COMPRESS)
        blocks = np.stack([comp.real, comp.imag]).transpose(0, 2, 1)   # [2, T, F]

        total = blocks.shape[1]
        stride = _FRAMES - _OVERLAP_FRAMES
        acc = np.zeros_like(blocks)
        weight = np.zeros((1, total, 1), dtype=np.float64)
        ramp = np.linspace(0.0, 1.0, _OVERLAP_FRAMES, dtype=np.float64)
        start = 0
        while start < total:
            block = blocks[:, start:start + _FRAMES]
            enhanced = self._run_window(block)
            envelope = np.ones((1, block.shape[1], 1), dtype=np.float64)
            if start > 0:
                span = min(_OVERLAP_FRAMES, block.shape[1])
                envelope[0, :span, 0] = ramp[:span]
            if start + _FRAMES < total:
                envelope[0, -_OVERLAP_FRAMES:, 0] = ramp[::-1]
            acc[:, start:start + block.shape[1]] += enhanced * envelope
            weight[:, start:start + block.shape[1]] += envelope
            start += stride

        acc /= np.where(weight > 1e-8, weight, 1.0)
        est = acc[0].T + 1j * acc[1].T                                  # [F, T]
        enhanced_spec = _compress(est, 1.0 / _COMPRESS)
        wav = istft(enhanced_spec, n_fft=_N_FFT, hop_size=_HOP, win_size=_N_FFT,
                    center=True, length=x64.size, window=window)
        return np.ascontiguousarray(wav[:n_out] / norm, dtype=np.float32)


register_engine(EngineEntry(
    alias="mossformergan",
    adapter_class=MossFormerGANAdapter,
    description=(
        "MossFormerGAN (ClearerVoice-Studio): MossFormer attention with a metric-GAN "
        "objective — the highest published PESQ (3.47) of any model surveyed here. Masked "
        "magnitude plus an additive complex residual, run over a fixed 401-frame window. "
        "ONNX from TigreGotico/audiosronnx-mossformergan. (Alibaba, Apache-2.0)"
    ),
    input_sample_rate=_SR,
    output_sample_rate=_SR,
    license="Apache-2.0",
    extras="",
    kind="denoise",
))
