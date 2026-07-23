"""UniPASE universal speech-enhancement adapter: ``unipase``.

UniPASE (Rong et al., *IEEE TASLP* 2026) is a **generative** universal speech enhancement
model: a single feed-forward pass removes noise and reverberation and reconstructs a clean
**16 kHz** signal. It is built on a de-noised WavLM (``DeWavLM-Omni``) SSL encoder whose L1
and L24 representations are fused by a Vocos adapter and resynthesised by a Vocos vocoder::

    encoder_adapter : wav[1, 128000]      ->  feat[1, 400, 1024]
    vocoder         : feat[1, 1024, T]    ->  spec[1, 1282, T]   (pre-ISTFT)

The encoder + adapter are fused into one graph (WavLM's conv front-end, transformer with
gated relative-position bias, and the Vocos adapter), shipped **fp32** — the 24-layer
WavLM-Large encoder loses too much to int8 (end-to-end corr drops to ~0.85), the same
depth-driven degradation that makes CallEnhancer default to fp32. The vocoder graph is cut
just before its ISTFT — ``torch.fft.irfft`` has no ONNX equivalent, so
the Vocos "same"-padding inverse STFT (n_fft 1280, hop 320) runs in numpy here, matching
upstream bit-for-bit. Nothing pulls torch at inference.

Being generative, it *resynthesises* speech rather than filtering it: treat the detail it
restores as enhanced-real (good to listen to, reasonable as an acoustic-model target) but
invented, not recovered.

Fixed window
------------
The encoder graph is length-specialised, so this adapter runs a fixed **8 s** window and
slides it with a **4 s hop**, stitching the 2 s-trimmed centres exactly as upstream's
``inference_long.py`` overlap-add.

Scope
-----
This engine ships UniPASE's 16 kHz core (denoise + dereverb). Upstream's optional PLC
(packet-loss concealment) and PostNet bandwidth-extension to 48 kHz are **not** included:
PLC needs a data-dependent CNN-output mask that does not trace, and PostNet depends on
``espnet2``. Both are candidate follow-ups. Input at any rate is resampled to 16 kHz.

ONNX artifact: ``TigreGotico/audiosronnx-unipase`` (MIT).

Reference
---------
- https://github.com/Xiaobin-Rong/unipase  (MIT; WavLM redistributed under MIT, see the
  upstream ``licenses/`` directory: ``LICENSE_wavlm``, ``LICENSE_pase``, ...)
- https://arxiv.org/abs/2604.14606
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from audiosronnx._kaiser import kaiser_resample
from audiosronnx.base import Denoiser, EngineEntry, register_engine
from audiosronnx.resolver import resolve

_HF_REPO = "TigreGotico/audiosronnx-unipase"
_HF_REVISION: Optional[str] = "af5ac49c86fba1240d2b6c2ba04a2983f2c36e6a"
_ENCODER_ADAPTER = "encoder_adapter.onnx"
_VOCODER = "vocoder.onnx"

_SR = 16000
_SEG_LEN = _SR * 8            # 8 s window the encoder graph was traced at
_HOP_LEN = _SR * 4            # 4 s hop between windows
_OLP_HALF = _SR * 2          # half the 4 s overlap, trimmed at each seam (upstream)

_N_FFT = 1280                 # Vocos vocoder ISTFT
_HOP = 320
_MAG_CLIP = (-20.0, 5.0)     # upstream ISTFTHead magnitude clamp


def _vocos_istft(spec_params: np.ndarray) -> np.ndarray:
    """Vocos "same"-padding inverse STFT in numpy: ``[1, n_fft+2, T]`` -> waveform.

    Reproduces upstream ``ISTFTHead``/``ISTFT`` (n_fft 1280, hop 320, periodic Hann, pad
    ``(n_fft - hop) // 2`` trimmed each side) — ``torch.fft.irfft`` is not ONNX-exportable,
    so the graph is cut before it and this closes the pipeline.
    """
    x = np.asarray(spec_params[0], dtype=np.float64)              # (n_fft+2, T)
    mag = np.exp(np.clip(x[: _N_FFT // 2 + 1], *_MAG_CLIP))
    p = x[_N_FFT // 2 + 1:]
    spec = mag * (np.cos(p) + 1j * np.sin(p))                     # (F, T)
    win = np.hanning(_N_FFT + 1)[:-1]                             # torch.hann_window (periodic)
    t = spec.shape[1]
    frames = np.fft.irfft(spec.T, n=_N_FFT, axis=1) * win         # (T, n_fft)
    out_size = (t - 1) * _HOP + _N_FFT
    ola = np.zeros(out_size, dtype=np.float64)
    env = np.zeros(out_size, dtype=np.float64)
    w2 = win ** 2
    for i in range(t):
        s = i * _HOP
        ola[s:s + _N_FFT] += frames[i]
        env[s:s + _N_FFT] += w2
    pad = (_N_FFT - _HOP) // 2
    ola, env = ola[pad:-pad], env[pad:-pad]
    return (ola / np.where(env > 1e-11, env, 1.0)).astype(np.float32)


class UniPASEAdapter(Denoiser):
    """UniPASE universal speech enhancement (16 kHz), backed by two ONNX graphs.

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
        encoder_adapter_path: Optional[str] = None,
        vocoder_path: Optional[str] = None,
        **cfg,
    ):
        super().__init__(**cfg)
        self._providers = providers
        self._cache_dir = cache_dir
        self._revision = revision if revision is not None else _HF_REVISION
        self._encoder_adapter_path = encoder_adapter_path
        self._vocoder_path = vocoder_path
        self._enc = None
        self._voc = None

    def _ensure_models(self) -> None:
        if self._enc is not None:
            return
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = os.cpu_count() or 4
        providers = self._providers or ["CPUExecutionProvider"]

        def _sess(explicit, name):
            path = explicit or resolve(
                name, hf_repo=_HF_REPO, revision=self._revision, cache_dir=self._cache_dir)
            return ort.InferenceSession(path, sess_options=opts, providers=providers)

        self._enc = _sess(self._encoder_adapter_path, _ENCODER_ADAPTER)
        self._voc = _sess(self._vocoder_path, _VOCODER)

    def _enhance_window(self, wave: np.ndarray) -> np.ndarray:
        """Run one fixed 8 s window (padded if short) and return ``wave.size`` samples."""
        n = wave.size
        padded = wave if n >= _SEG_LEN else np.pad(wave, (0, _SEG_LEN - n))
        feat = self._enc.run(None, {"wav": padded[None].astype(np.float32)})[0]  # [1,T,1024]
        spec = self._voc.run(
            None, {"feat": np.ascontiguousarray(feat.transpose(0, 2, 1))})[0]    # [1,1282,T]
        out = _vocos_istft(spec)
        if out.size < n:
            out = np.pad(out, (0, n - out.size))
        return out[:n]

    def _denoise_array(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        self._ensure_models()
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sample_rate != _SR:
            x = np.asarray(kaiser_resample(x, sample_rate, _SR), dtype=np.float32)
        n_out = x.size
        if n_out == 0:
            return np.zeros(0, dtype=np.float32)
        scale = float(np.max(np.abs(x)))
        if not np.isfinite(scale) or scale <= 0:
            return np.zeros(n_out, dtype=np.float32)

        # Segment into 8 s windows with a 4 s hop (upstream inference_long.py).
        patches: List[np.ndarray] = []
        start = 0
        while start < n_out:
            end = start + _SEG_LEN
            if end >= n_out:
                patches.append(x[start:n_out])
                break
            patches.append(x[start:end])
            start += _HOP_LEN

        if len(patches) == 1:
            enhanced = self._enhance_window(patches[0])
        else:
            parts = [self._enhance_window(patches[0])[: _SEG_LEN - _OLP_HALF]]
            for mid in patches[1:-1]:
                parts.append(self._enhance_window(mid)[_OLP_HALF: _SEG_LEN - _OLP_HALF])
            parts.append(self._enhance_window(patches[-1])[_OLP_HALF:])
            enhanced = np.concatenate(parts)

        enhanced = enhanced[:n_out]
        if enhanced.size < n_out:
            enhanced = np.pad(enhanced, (0, n_out - enhanced.size))
        peak = float(np.max(np.abs(enhanced)))
        if peak > 0:
            enhanced = enhanced / peak * scale
        return np.ascontiguousarray(enhanced, dtype=np.float32)


register_engine(EngineEntry(
    alias="unipase",
    adapter_class=UniPASEAdapter,
    description=(
        "UniPASE (Rong et al., IEEE TASLP): generative universal speech enhancement — "
        "noise and reverberation removed in one feed-forward pass at 16 kHz. DeWavLM-Omni "
        "SSL encoder + Vocos adapter + Vocos vocoder, over a fixed 8 s window. Generative, "
        "so the detail it restores is invented. fp32 WavLM encoder; ISTFT in numpy. ONNX from "
        "TigreGotico/audiosronnx-unipase. (MIT)"
    ),
    input_sample_rate=_SR,
    output_sample_rate=_SR,
    license="MIT",
    extras="",
    kind="enhance",
))
