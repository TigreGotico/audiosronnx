"""Sidon speech-restoration adapter: ``sidon``.

Sidon (SARULab-Speech) restores degraded speech: it cleans **16 kHz** input and
resynthesises it at **48 kHz**. Two ONNX graphs do the neural work — an 8-layer
w2v-BERT 2.0 feature extractor (LoRA-adapted to denoise SSL representations) that maps
SeamlessM4T log-mel features ``[1, T, 160]`` to hidden states ``[1, T, 1024]`` at 50 Hz,
then a DAC vocoder decoder (960x upsample) that turns ``[1, 1024, T]`` back into a
``[1, 1, 960*T]`` 48 kHz waveform::

    feature_extractor : input_features[1,T,160] -> last_hidden_state[1,T,1024]
    decoder           : features[1,1024,T]      -> waveform[1,1,960*T]

The SeamlessM4T mel front-end (Kaldi-style log-mel + per-bin CMVN + stride-2 stacking)
runs in numpy here — ``seamless_fbank`` reproduces
``transformers.SeamlessM4TFeatureExtractor`` bit-for-bit using the bundled window and
mel-filterbank arrays — so nothing pulls torch or transformers at inference time.

Chunked inference follows upstream: 16 kHz audio is peak-normalised, high-passed at
50 Hz, tail-padded, split into 96 s windows, and each window is featurised (with a
±160-sample edge pad) and decoded. One feature frame is carried across the window seam
and the decoder tail (960 samples) is trimmed, exactly as the reference demo.

The feature extractor is shipped int8-quantized for CPU; the decoder stays fp32.

ONNX artifact: ``TigreGotico/audiosronnx-sidon`` (MIT).

Reference
---------
- https://github.com/sarulab-speech/Sidon  (MIT)
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from audiosronnx._kaiser import kaiser_resample
from audiosronnx.base import EngineEntry, SRModel, register_engine
from audiosronnx.resolver import resolve

_HF_REPO = "TigreGotico/audiosronnx-sidon"
_HF_REVISION: Optional[str] = "289a72a48d9e39c4efc3cf79fb3cd05178090292"
_FE = "feature_extractor.int8.onnx"
_DEC = "decoder.onnx"

_IN_SR = 16000
_OUT_SR = 48000
_CHUNK = _IN_SR * 96          # 96 s windows, as upstream
_EDGE_PAD = 160               # per-chunk edge pad fed to the mel front-end
_TAIL_PAD = 24000             # tail pad added before chunking (upstream)
_DEC_TRIM = 960               # decoder tail samples trimmed per chunk (one feature frame)

# SeamlessM4T fbank parameters (facebook/w2v-bert-2.0).
_FRAME = 400
_HOP = 160
_FFT = 512
_PREEMPH = 0.97
_MEL_FLOOR = 1.192092955078125e-07
_STRIDE = 2
_PAD_VALUE = 1.0              # SeamlessM4TFeatureExtractor.padding_value

_WINDOW: Optional[np.ndarray] = None
_MEL: Optional[np.ndarray] = None


def _load_asset(name: str) -> np.ndarray:
    from importlib.resources import files

    with files("audiosronnx").joinpath("data", name).open("rb") as fh:
        return np.load(fh)


def _assets() -> tuple[np.ndarray, np.ndarray]:
    global _WINDOW, _MEL
    if _WINDOW is None:
        _WINDOW = _load_asset("sidon_window.npy").astype(np.float64)      # (400,)
        _MEL = _load_asset("sidon_mel_filters.npy").astype(np.float64)    # (257, 80)
    return _WINDOW, _MEL


def seamless_fbank(waveform: np.ndarray) -> np.ndarray:
    """Numpy reimplementation of ``SeamlessM4TFeatureExtractor`` for a mono 16 kHz clip.

    Returns stacked log-mel features ``[T//2, 160]`` (float32), matching the HF extractor
    called with ``sampling_rate=16000`` (default ``do_normalize_per_mel_bins=True``,
    ``pad_to_multiple_of=2``).
    """
    window, mel = _assets()
    wav = np.squeeze(np.asarray(waveform)).astype(np.float64) * (2 ** 15)
    if wav.ndim != 1:
        raise ValueError(f"expected mono waveform, got shape {wav.shape}")

    num_frames = 1 + max(0, (wav.size - _FRAME) // _HOP)
    frames = np.lib.stride_tricks.sliding_window_view(wav, _FRAME)[::_HOP][:num_frames]
    frames = frames.copy()
    # remove DC offset, pre-emphasis, window (Kaldi order) — matches transformers.spectrogram
    frames -= frames.mean(axis=1, keepdims=True)
    frames[:, 1:] -= _PREEMPH * frames[:, :-1]
    frames[:, 0] *= 1 - _PREEMPH
    frames *= window

    power = np.abs(np.fft.rfft(frames, n=_FFT, axis=1)) ** 2      # [T, 257]
    feats = np.log(np.maximum(_MEL_FLOOR, power @ mel))          # [T, 80]

    # per-mel-bin CMVN over time (torch ddof=1)
    feats = (feats - feats.mean(0, keepdims=True)) / np.sqrt(
        feats.var(0, ddof=1, keepdims=True) + 1e-7)

    # pad to a multiple of stride, then stack `stride` frames -> 160 features
    if feats.shape[0] % _STRIDE:
        pad = _STRIDE - (feats.shape[0] % _STRIDE)
        feats = np.pad(feats, ((0, pad), (0, 0)), constant_values=_PAD_VALUE)
    t = feats.shape[0]
    return feats.reshape(t // _STRIDE, feats.shape[1] * _STRIDE).astype(np.float32)


class SidonAdapter(SRModel):
    """Sidon speech restoration (16 kHz -> 48 kHz), backed by two ONNX graphs.

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
        fe_path: Optional[str] = None,
        decoder_path: Optional[str] = None,
        chunk_samples: int = _CHUNK,
        **cfg,
    ):
        super().__init__(**cfg)
        self._providers = providers
        self._cache_dir = cache_dir
        self._revision = revision or _HF_REVISION
        self._fe_path = fe_path
        self._decoder_path = decoder_path
        # Window length (16 kHz samples) fed to the feature extractor per step. The w2v-BERT
        # attention is O(T^2), so the 96 s default is memory-heavy on CPU; a smaller value
        # trades a little seam overhead for a much lower peak footprint.
        self._chunk = int(chunk_samples)
        self._fe = None
        self._dec = None

    def _ensure_models(self) -> None:
        if self._fe is not None:
            return
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = os.cpu_count() or 4
        providers = self._providers or ["CPUExecutionProvider"]

        def _sess(explicit, name):
            path = explicit or resolve(
                name, hf_repo=_HF_REPO, revision=self._revision, cache_dir=self._cache_dir)
            return ort.InferenceSession(path, sess_options=opts, providers=providers)

        self._fe = _sess(self._fe_path, _FE)
        self._dec = _sess(self._decoder_path, _DEC)

    def _upscale_array(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        self._ensure_models()
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        n_in = x.size
        if n_in == 0:
            return np.zeros(0, dtype=np.float32)

        if sample_rate != _IN_SR:
            x = np.asarray(kaiser_resample(x, sample_rate, _IN_SR), dtype=np.float32)

        peak = float(np.max(np.abs(x)))
        if not np.isfinite(peak) or peak <= 0:
            return np.zeros(int(round(_OUT_SR / sample_rate * n_in)), dtype=np.float32)
        x = 0.9 * (x / peak)
        x = _highpass_50hz(x, _IN_SR)
        x = np.pad(x, (0, _TAIL_PAD))

        restored: List[np.ndarray] = []
        cache = None
        for start in range(0, x.size, self._chunk):
            chunk = x[start:start + self._chunk]
            if chunk.size == 0:
                continue
            feats = seamless_fbank(np.pad(chunk, (_EDGE_PAD, _EDGE_PAD)))[None]  # [1,T,160]
            hidden = self._fe.run(None, {"input_features": feats.astype(np.float32)})[0]
            if cache is not None:
                hidden = np.concatenate([cache, hidden], axis=1)
            wav = self._dec.run(
                None, {"features": np.ascontiguousarray(hidden.transpose(0, 2, 1))})[0]
            restored.append(wav.reshape(-1)[:-_DEC_TRIM])
            cache = hidden[:, -1:]

        out = np.concatenate(restored) if restored else np.zeros(0, dtype=np.float32)
        target = int(round(_OUT_SR / sample_rate * n_in))
        return out[:target].astype(np.float32)


def _highpass_50hz(x: np.ndarray, sr: int, cutoff: float = 50.0) -> np.ndarray:
    """Biquad high-pass matching ``torchaudio.functional.highpass_biquad`` (Q=0.707)."""
    from scipy.signal import lfilter

    w0 = 2 * np.pi * cutoff / sr
    q = 0.707106781186547
    alpha = np.sin(w0) / (2 * q)
    cos_w0 = np.cos(w0)
    b0 = (1 + cos_w0) / 2
    b1 = -(1 + cos_w0)
    b2 = (1 + cos_w0) / 2
    a0 = 1 + alpha
    a1 = -2 * cos_w0
    a2 = 1 - alpha
    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return lfilter(b, a, x).astype(np.float32)


register_engine(
    EngineEntry(
        alias="sidon",
        adapter_class=SidonAdapter,
        description=(
            "Sidon speech restoration: 8-layer w2v-BERT 2.0 feature predictor + DAC "
            "vocoder, 16 kHz -> 48 kHz. SeamlessM4T mel front-end in numpy, int8 "
            "feature extractor. ONNX from TigreGotico/audiosronnx-sidon. "
            "(SARULab-Speech, MIT)"
        ),
        input_sample_rate=_IN_SR,
        output_sample_rate=_OUT_SR,
        license="MIT",
        extras="",
        kind="sr",
    )
)
