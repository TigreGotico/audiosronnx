"""FLowHigh bandwidth-extension adapter: ``flowhigh``.

FLowHigh (Yun et al., ICASSP 2025) upscales speech to 48 kHz with **conditional flow
matching**. Rather than denoising over tens of diffusion steps, it learns a velocity field
and integrates it over a handful of Euler steps — four by default — which makes a
generative super-resolution model practical on CPU.

Two ONNX graphs do the neural work::

    flow field : x[1,T,256], times[1], cond[1,T,256] -> velocity[1,T,256]
    vocoder    : mel[1,256,T]                        -> waveform[1,1,480*T]

Everything else runs in numpy: the log-mel front-end, the Euler loop, and a post-processing
merge that keeps the input's own low band and takes only the reconstructed high band from
the model — the same idea as :mod:`~audiosronnx.engines.lavasr`'s spectral merge, so the
original signal is never resynthesised away.

Sampling starts from ``y0 = mel(cond) + eps``. Upstream draws ``eps`` freshly each run, so
its output is not reproducible; this adapter seeds the draw (``seed=0``) and is
deterministic by default. Pass ``seed=None`` for upstream's behaviour.

More Euler steps trade compute for fidelity; ``steps=4`` is upstream's default and the
setting its published results use.

ONNX artifacts: ``TigreGotico/audiosronnx-flowhigh`` (MIT), which also carries the BigVGAN
48 kHz vocoder (NVIDIA, MIT).

Export note
-----------
BigVGAN's alias-free resamplers build their depthwise kernel at call time with
``filter.expand(C, -1, -1)``, leaving the tracer with a convolution of unknown kernel
shape. The channel count is fixed per layer, so the published graph materialises each
expanded kernel as a buffer — 182 of them — which is bit-for-bit equivalent.

Reference
---------
- https://github.com/jjunak-yun/FLowHigh_code  (MIT)
- https://github.com/NVIDIA/BigVGAN  (MIT)
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from audiosronnx._stft import istft, stft
from audiosronnx.base import EngineEntry, SRModel, register_engine
from audiosronnx.resolver import resolve

_HF_REPO = "TigreGotico/audiosronnx-flowhigh"
_HF_REVISION: Optional[str] = "0e9984c96ee5e61188c4506ebb4743e061c557cb"
_FLOW = "flowfield.onnx"
_VOCODER = "bigvgan.onnx"

_SR = 48000
_N_FFT = 2048
_HOP = 480
_N_MELS = 256
_F_MIN = 20.0
_F_MAX = 24000.0
_LOG_FLOOR = 1e-5           # upstream clamps magnitudes here before the log
_MAG_EPS = 1e-9
_STEPS = 4
#: fraction of cumulative spectral energy that defines the input's own band
_ENERGY_THRESHOLD = 0.99

_MEL_FB: Optional[np.ndarray] = None
_WINDOW: Optional[np.ndarray] = None


def _mel_filterbank() -> np.ndarray:
    """Slaney-normalised mel filterbank, matching ``librosa.filters.mel``."""
    global _MEL_FB
    if _MEL_FB is None:
        from audiosronnx.engines.lavasr import _build_mel_filterbank

        _MEL_FB = _build_mel_filterbank(_SR, _N_FFT, _N_MELS, _F_MIN, _F_MAX)
    return _MEL_FB


def _hann() -> np.ndarray:
    """``torch.hann_window(n)`` — periodic."""
    global _WINDOW
    if _WINDOW is None:
        _WINDOW = np.hanning(_N_FFT + 1)[:-1]
    return _WINDOW


def log_mel(audio: np.ndarray) -> np.ndarray:
    """Log-mel spectrogram ``[1, frames, 256]`` matching the upstream front-end.

    Reflect-pads by ``(n_fft - hop) / 2`` and runs an uncentred STFT, which is the
    HiFi-GAN/BigVGAN convention rather than a centred one.
    """
    pad = (_N_FFT - _HOP) // 2
    x = np.pad(np.asarray(audio, dtype=np.float64), (pad, pad), mode="reflect")
    frames = max(0, 1 + (x.size - _N_FFT) // _HOP)
    idx = np.arange(_N_FFT)[None, :] + _HOP * np.arange(frames)[:, None]
    spec = np.fft.rfft(x[idx] * _hann(), n=_N_FFT, axis=1).T           # [F, T]
    mag = np.sqrt(spec.real ** 2 + spec.imag ** 2 + _MAG_EPS)
    mel = _mel_filterbank() @ mag
    return np.log(np.clip(mel, _LOG_FLOOR, None)).T[None]              # [1, T, mels]


def _cutoff_bin(spec: np.ndarray) -> int:
    """Highest bin holding ``_ENERGY_THRESHOLD`` of the input's cumulative energy.

    Above it the source carries essentially nothing, so that is where the model's
    reconstruction should take over.
    """
    energy = np.cumsum(np.abs(spec).sum(axis=-1))
    if energy[-1] <= 0:
        return 0
    limit = energy[-1] * _ENERGY_THRESHOLD
    below = np.nonzero(energy < limit)[0]
    return int(below[-1]) + 1 if below.size else 0


class FLowHighAdapter(SRModel):
    """FLowHigh flow-matching bandwidth extension to 48 kHz.

    Parameters
    ----------
    steps:
        Euler steps for the ODE (default 4, upstream's setting). More steps cost
        proportionally more compute.
    seed:
        Seed for the noise the sampler starts from. ``None`` draws freshly each call, as
        upstream does, making output non-reproducible.
    merge:
        Keep the input's own low band and take only the reconstructed high band
        (default). Disable to return the vocoder output unmodified.
    providers:
        onnxruntime execution providers (default CPU).
    cache_dir:
        Override the model download cache directory.
    """

    input_sample_rate = 0        # accepts any rate; resampled to 48 kHz
    output_sample_rate = _SR

    def __init__(
        self,
        *,
        steps: int = _STEPS,
        seed: Optional[int] = 0,
        merge: bool = True,
        providers: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
        revision: Optional[str] = None,
        flow_path: Optional[str] = None,
        vocoder_path: Optional[str] = None,
        **cfg,
    ):
        super().__init__(**cfg)
        if int(steps) < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        self._steps = int(steps)
        self._seed = seed
        self._merge = bool(merge)
        self._providers = providers
        self._cache_dir = cache_dir
        self._revision = revision if revision is not None else _HF_REVISION
        self._flow_path = flow_path
        self._vocoder_path = vocoder_path
        self._flow = None
        self._voc = None

    def _ensure_models(self) -> None:
        if self._flow is not None:
            return
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = os.cpu_count() or 4
        providers = self._providers or ["CPUExecutionProvider"]

        def _sess(explicit, name):
            path = explicit or resolve(
                name, hf_repo=_HF_REPO, revision=self._revision, cache_dir=self._cache_dir)
            return ort.InferenceSession(path, sess_options=opts, providers=providers)

        self._flow = _sess(self._flow_path, _FLOW)
        self._voc = _sess(self._vocoder_path, _VOCODER)

    def _integrate(self, cond: np.ndarray) -> np.ndarray:
        """Euler-integrate the velocity field from ``cond + eps`` over ``[0, 1]``."""
        rng = np.random.RandomState(self._seed) if self._seed is not None else np.random
        y = cond + rng.randn(*cond.shape).astype(np.float32)
        times = np.linspace(0.0, 1.0, self._steps + 1, dtype=np.float32)
        for i in range(self._steps):
            velocity = self._flow.run(None, {
                "x": np.ascontiguousarray(y, dtype=np.float32),
                "times": times[i:i + 1],
                "cond": cond,
            })[0]
            y = y + (times[i + 1] - times[i]) * velocity
        return y

    def _blend(self, generated: np.ndarray, source: np.ndarray) -> np.ndarray:
        """Keep the source's own band, take the reconstructed band above its cutoff."""
        gen_spec = stft(generated, n_fft=_N_FFT, hop_size=_HOP, win_size=_N_FFT,
                        center=True, window=_hann())
        src_spec = stft(source, n_fft=_N_FFT, hop_size=_HOP, win_size=_N_FFT,
                        center=True, window=_hann())
        frames = min(gen_spec.shape[1], src_spec.shape[1])
        gen_spec, src_spec = gen_spec[:, :frames], src_spec[:, :frames]

        cutoff = _cutoff_bin(src_spec)
        merged = gen_spec.copy()
        merged[:cutoff] = src_spec[:cutoff]
        return istft(merged, n_fft=_N_FFT, hop_size=_HOP, win_size=_N_FFT,
                     center=True, length=source.size, window=_hann())

    def _upscale_array(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        self._ensure_models()
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        n_in = x.size
        if n_in == 0:
            return np.zeros(0, dtype=np.float32)
        target = int(round(_SR / sample_rate * n_in))

        peak = float(np.max(np.abs(x)))
        if not np.isfinite(peak) or peak <= 0:
            return np.zeros(target, dtype=np.float32)

        if sample_rate != _SR:
            from scipy.signal import resample_poly
            from math import gcd

            g = gcd(int(_SR), int(sample_rate))
            x = resample_poly(x, _SR // g, sample_rate // g).astype(np.float32)
        cond_wave = (x / max(float(np.max(np.abs(x))), 1e-12)).astype(np.float32)

        cond = log_mel(cond_wave).astype(np.float32)
        if cond.shape[1] == 0:      # shorter than one analysis window: nothing to extend
            return np.zeros(target, dtype=np.float32)
        mel = self._integrate(cond)
        wav = self._voc.run(
            None, {"mel": np.ascontiguousarray(mel.transpose(0, 2, 1), dtype=np.float32)}
        )[0].reshape(-1)

        if self._merge:
            usable = min(wav.size, cond_wave.size)
            wav = self._blend(wav[:usable], cond_wave[:usable])
            loudest = float(np.max(np.abs(wav)))
            if loudest > 0:
                wav = wav / loudest * 0.99

        out = np.zeros(target, dtype=np.float32)
        usable = min(target, wav.size)
        out[:usable] = wav[:usable]
        return out


register_engine(EngineEntry(
    alias="flowhigh",
    adapter_class=FLowHighAdapter,
    description=(
        "FLowHigh (Yun et al., ICASSP 2025): conditional flow-matching bandwidth "
        "extension to 48 kHz in 4 Euler steps, with a BigVGAN vocoder. Log-mel front-end, "
        "ODE loop and spectral merge in numpy. ONNX from TigreGotico/audiosronnx-flowhigh. "
        "(MIT)"
    ),
    input_sample_rate=0,
    output_sample_rate=_SR,
    license="MIT",
    extras="",
    kind="sr",
))
