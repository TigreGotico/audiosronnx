"""LavaSR adapter for audiosronnx (default engine).

LavaSR (Sharma, Interspeech 2026) is a lightweight Vocos-based bandwidth-extension /
speech-enhancement model. It restores low-quality / low-bandwidth speech to a clean
48 kHz signal, with an optional UL-UNAS denoiser stage and a Linkwitz-Riley spectral
merge that preserves the original low band.

Inference runs entirely through onnxruntime; every spectral operation (STFT, ISTFT,
mel filterbank, resampling, spectral merge) is pure numpy/scipy, so no ``torch.stft``
ever enters an ONNX graph. Two ONNX graphs are required (``backbone`` + ``spec_head``);
the ``denoiser`` graph is optional and only used with ``denoise=True``.

Input: any sample rate 8-48 kHz (internally processed at 16 kHz). Output: 48 kHz.

ONNX artifacts: ``TigreGotico/audiosronnx-lavasr`` (Apache-2.0).

References
----------
- https://github.com/ysharma3501/LavaSR
- https://github.com/Topping1/LavaSR-ONNX (pure-ONNX + numpy DSP approach)
"""
from __future__ import annotations

import math
import os
from typing import List, Optional

import numpy as np

from audiosronnx.base import EngineEntry, SRModel, register_engine
from audiosronnx.resolver import resolve

_HF_REPO = "TigreGotico/audiosronnx-lavasr"
_HF_REVISION = "b3df8a262cf44e59bf84a40b7084f4479ca566b4"
_BACKBONE = "backbone.onnx"
_SPEC_HEAD = "spec_head.onnx"
_DENOISER = "denoiser_core.onnx"

# Enhancer DSP constants (must match the exported graph's training front-end).
_ENH_SR = 44100
_OUT_SR = 48000
_ENH_NFFT = 2048
_ENH_HOP = 512
_ENH_MELS = 80
_DEN_SR = 16000
_DEN_NFFT = 512
_DEN_HOP = 256
_DEN_CHUNK = 63


class LavaSRAdapter(SRModel):
    """Bandwidth-extension adapter backed by LavaSR ONNX graphs.

    Parameters
    ----------
    denoise:
        Run the optional UL-UNAS denoiser before enhancement (default ``False``).
    cutoff_hz:
        Frequency below which the original signal is preserved in the spectral merge.
        ``None`` (default) picks ``min(sample_rate, 16000) / 2``.
    providers:
        onnxruntime execution providers (default CPU).
    cache_dir:
        Override the model download cache directory.
    """

    input_sample_rate = 0  # flexible 8-48 kHz
    output_sample_rate = _OUT_SR

    def __init__(
        self,
        denoise: bool = False,
        cutoff_hz: Optional[float] = None,
        *,
        providers: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
        revision: Optional[str] = None,
        **cfg,
    ):
        super().__init__(**cfg)
        self._denoise = bool(denoise)
        self._cutoff_hz = cutoff_hz
        self._providers = providers
        self._cache_dir = cache_dir
        self._revision = revision or _HF_REVISION
        self._backbone = None
        self._head = None
        self._denoiser = None
        self._mel_fb = _build_mel_filterbank(_ENH_SR, _ENH_NFFT, _ENH_MELS, 0.0, 8000.0)

    # ------------------------------------------------------------------ #
    def _ensure_models(self) -> None:
        if self._backbone is not None:
            return
        import onnxruntime as ort

        providers = self._providers or ["CPUExecutionProvider"]
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = os.cpu_count() or 4

        def _sess(fname: str):
            path = resolve(
                fname, hf_repo=_HF_REPO, revision=self._revision, cache_dir=self._cache_dir
            )
            return ort.InferenceSession(path, sess_options=opts, providers=providers)

        self._backbone = _sess(_BACKBONE)
        self._head = _sess(_SPEC_HEAD)
        if self._denoise:
            self._denoiser = _sess(_DENOISER)

    # ------------------------------------------------------------------ #
    def _upscale_array(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        self._ensure_models()
        cutoff = self._cutoff_hz
        if cutoff is None:
            cutoff = min(sample_rate, 16000) / 2.0

        wave16 = _resample(audio, sample_rate, _DEN_SR)
        if self._denoise and self._denoiser is not None:
            wave16 = self._run_denoiser(wave16)

        enh_in = _resample(wave16, _DEN_SR, _ENH_SR)
        enhanced = self._run_enhancer(enh_in)
        merged = _spectral_merge(enh_in, enhanced, _ENH_SR, cutoff, 1024)
        merged = _resample(merged, _ENH_SR, _OUT_SR)
        return np.clip(merged, -1.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------ #
    def _run_enhancer(self, waveform: np.ndarray) -> np.ndarray:
        spec = _stft(waveform, _ENH_SR, _ENH_NFFT, _ENH_HOP)
        mag = np.abs(spec).astype(np.float32)
        mel = np.matmul(self._mel_fb, mag)
        mel = np.log(np.maximum(mel, 1e-5)).astype(np.float32)[None, :, :]

        hidden = self._backbone.run(
            None, {self._backbone.get_inputs()[0].name: mel}
        )[0]
        head_out = self._head.run(
            None, {self._head.get_inputs()[0].name: hidden.astype(np.float32)}
        )
        real, imag = head_out[0][0], head_out[1][0]
        complex_spec = (real + 1j * imag).astype(np.complex64)
        return _istft(complex_spec, _ENH_SR, _ENH_NFFT, _ENH_HOP, waveform.shape[0])

    def _run_denoiser(self, waveform: np.ndarray) -> np.ndarray:
        spec = _stft(waveform, _DEN_SR, _DEN_NFFT, _DEN_HOP)
        ri = np.stack([spec.real, spec.imag], axis=0).transpose(0, 2, 1)
        den_in = ri[None, :, :, :].astype(np.float32)  # [1, 2, T, F]

        out = np.zeros_like(den_in, dtype=np.float32)
        weights = np.zeros((den_in.shape[2],), dtype=np.float32)
        window = np.hanning(_DEN_CHUNK).astype(np.float32)
        stride = max(1, _DEN_CHUNK // 2)
        frames = den_in.shape[2]
        name = self._denoiser.get_inputs()[0].name

        for start in range(0, frames, stride):
            end = min(start + _DEN_CHUNK, frames)
            valid = end - start
            chunk = den_in[:, :, start:end, :]
            if valid < _DEN_CHUNK:
                chunk = np.pad(chunk, ((0, 0), (0, 0), (0, _DEN_CHUNK - valid), (0, 0)))
            res = self._denoiser.run(None, {name: chunk})[0][:, :, :valid, :]
            w = window[:valid][None, None, :, None]
            out[:, :, start:end, :] += res * w
            weights[start:end] += window[:valid]
            if end >= frames:
                break

        weights = np.maximum(weights, 1e-6)
        out = out / weights[None, None, :, None]
        spec_out = (out[0, 0] + 1j * out[0, 1]).T.astype(np.complex64)
        return _istft(spec_out, _DEN_SR, _DEN_NFFT, _DEN_HOP, waveform.shape[0])


# --------------------------------------------------------------------------- #
# Host-side DSP (pure numpy / scipy)
# --------------------------------------------------------------------------- #
def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio.astype(np.float32, copy=False)
    from scipy import signal

    g = math.gcd(int(src_sr), int(dst_sr))
    return signal.resample_poly(audio, dst_sr // g, src_sr // g).astype(np.float32)


def _stft(wave: np.ndarray, sr: int, n_fft: int, hop: int) -> np.ndarray:
    from scipy import signal

    _, _, spec = signal.stft(
        wave, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop,
        nfft=n_fft, boundary="zeros", padded=True, return_onesided=True,
    )
    return spec.astype(np.complex64)


def _istft(spec, sr, n_fft, hop, target_len=None):
    from scipy import signal

    _, wav = signal.istft(
        spec, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop,
        nfft=n_fft, input_onesided=True, boundary=True,
    )
    wav = np.real(wav).astype(np.float32)
    if target_len is not None:
        if wav.shape[0] > target_len:
            wav = wav[:target_len]
        elif wav.shape[0] < target_len:
            wav = np.pad(wav, (0, target_len - wav.shape[0]))
    return wav


def _hz_to_mel(f):
    f_sp, min_log_hz = 200.0 / 3.0, 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    mel = np.empty_like(f, dtype=np.float64)
    lin = f < min_log_hz
    mel[lin] = f[lin] / f_sp
    mel[~lin] = min_log_mel + np.log(f[~lin] / min_log_hz) / logstep
    return mel


def _mel_to_hz(mel):
    f_sp, min_log_hz = 200.0 / 3.0, 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    hz = np.empty_like(mel, dtype=np.float64)
    lin = mel < min_log_mel
    hz[lin] = mel[lin] * f_sp
    hz[~lin] = min_log_hz * np.exp(logstep * (mel[~lin] - min_log_mel))
    return hz


def _build_mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    fft_freqs = np.linspace(0.0, sr / 2.0, (n_fft // 2) + 1)
    mel_edges = np.linspace(
        _hz_to_mel(np.array([fmin]))[0], _hz_to_mel(np.array([fmax]))[0], n_mels + 2
    )
    hz_edges = _mel_to_hz(mel_edges)
    fb = np.zeros((n_mels, fft_freqs.shape[0]), dtype=np.float32)
    for m in range(n_mels):
        left, center, right = hz_edges[m], hz_edges[m + 1], hz_edges[m + 2]
        if center <= left or right <= center:
            continue
        up = (fft_freqs - left) / (center - left)
        down = (right - fft_freqs) / (right - center)
        fb[m] = np.maximum(0.0, np.minimum(up, down)) * (2.0 / max(1e-8, right - left))
    return fb


def _spectral_merge(original, enhanced, sr, cutoff_hz, transition_bins):
    n = min(original.shape[0], enhanced.shape[0])
    if n <= 0:
        return enhanced
    original, enhanced = original[:n], enhanced[:n]
    spec_o, spec_e = np.fft.rfft(original), np.fft.rfft(enhanced)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    cutoff_bin = int(np.argmin(np.abs(freqs - cutoff_hz)))
    half = max(1, transition_bins // 2)
    start = max(0, cutoff_bin - half)
    end = min(spec_o.shape[0] - 1, cutoff_bin + half)
    mask = np.zeros(spec_o.shape[0], dtype=np.float32)
    if start > 0:
        mask[:start] = 1.0
    if end > start:
        t = np.linspace(1.0, 0.0, end - start + 1, dtype=np.float32)
        mask[start:end + 1] = 3.0 * (t**2) - 2.0 * (t**3)
    merged = spec_e + (spec_o - spec_e) * mask
    return np.fft.irfft(merged, n=n).astype(np.float32)


register_engine(
    EngineEntry(
        alias="lavasr",
        adapter_class=LavaSRAdapter,
        description=(
            "LavaSR: Vocos-based bandwidth extension with a Linkwitz-Riley spectral "
            "merge and optional UL-UNAS denoiser. Any input 8-48 kHz -> 48 kHz. "
            "~52 MB, ~50x realtime on CPU. ONNX from TigreGotico/audiosronnx-lavasr."
        ),
        input_sample_rate=0,
        output_sample_rate=_OUT_SR,
        license="Apache-2.0",
        extras="",
    )
)
