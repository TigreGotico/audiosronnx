"""CallEnhancer call-centre speech-restoration adapter: ``callenhancer``.

CallEnhancer (Scicom-intl) restores narrowband, codec'd telephony / call-centre speech:
it cleans **16 kHz** input and resynthesises it at **48 kHz**. It is the heavier sibling
of :mod:`~audiosronnx.engines.sidon` — same two-graph shape, but the feature extractor is
the *full* 24-layer w2v-BERT 2.0 encoder (LoRA-adapted, r=64/alpha=16) rather than Sidon's
8 layers::

    feature_extractor : input_features[1,T,160] -> last_hidden_state[1,T,1024]   (50 Hz)
    decoder           : features[1,1024,T]      -> waveform[1,1,960*T]           (48 kHz)

The SeamlessM4T log-mel front-end is identical to Sidon's, so this adapter reuses
:func:`audiosronnx.engines.sidon.seamless_fbank` — a numpy reimplementation of
``transformers.SeamlessM4TFeatureExtractor`` — and nothing here pulls torch or transformers
at inference time.

Inference follows the upstream reference: 16 kHz audio is peak-normalised to 0.95, and by
default runs as a **single straight pass** — w2v-BERT uses rotary/relative position
embeddings and the DAC decoder is convolutional, so a full pass is length-invariant and
cleaner than windowing. For very long audio (self-attention is O(T^2)) a ``chunk_seconds``
window with a 2 s overlap and a linear crossfade is available as a memory fallback. Each
segment is edge-padded by 40 samples before the mel front-end, and the output is
peak-normalised to 0.97.

The feature extractor is shipped int8-quantized for CPU; the decoder stays fp32.

ONNX artifact: ``TigreGotico/audiosronnx-callenhancer`` (CC-BY-NC-4.0).

Reference
---------
- https://huggingface.co/Scicom-intl/CallEnhancer  (CC-BY-NC-4.0)
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

from audiosronnx._kaiser import kaiser_resample
from audiosronnx.base import EngineEntry, SRModel, register_engine
from audiosronnx.engines.sidon import seamless_fbank
from audiosronnx.resolver import resolve

_HF_REPO = "TigreGotico/audiosronnx-callenhancer"
_HF_REVISION: Optional[str] = "302682459d18c584710d162246e1b8265ab7f1cb"
_DEC = "decoder.onnx"
#: feature-extractor graph per precision. ``fp32`` is full-fidelity (weights ride in an
#: external ``.onnx.data`` sidecar that is fetched alongside); ``int8`` is ~4x smaller but,
#: on this 24-layer encoder, audibly lossy (~12 dB SNR vs fp32) — a size/speed trade, not
#: free. See the README precision table.
_FE_FILES = {"fp32": "feature_extractor.onnx", "int8": "feature_extractor.int8.onnx"}
_DEFAULT_PRECISION = "fp32"

_IN_SR = 16000
_OUT_SR = 48000
_RATIO = _OUT_SR // _IN_SR    # 3 — decoder upsamples 16 kHz input frames to 48 kHz
_EDGE_PAD = 40                # per-segment edge pad fed to the mel front-end (upstream)
#: minimum segment length (samples) fed to the front-end. The SeamlessM4T fbank frame is
#: 400 samples, and below ~3 mel frames the per-bin CMVN (ddof=1 over time) divides by a
#: near-zero variance and emits NaNs, so a very short clip is zero-padded up to this floor
#: before featurisation (its output is trimmed back to the length contract afterwards).
_MIN_SEG = 1200
_IN_PEAK = 0.95              # input peak-normalisation target (upstream)
_OUT_PEAK = 0.97            # output peak-normalisation target (upstream)
_OVERLAP = 2 * _IN_SR        # 2 s window overlap when chunking


class CallEnhancerAdapter(SRModel):
    """CallEnhancer call-centre restoration (16 kHz -> 48 kHz), two ONNX graphs.

    Parameters
    ----------
    providers:
        onnxruntime execution providers (default CPU).
    cache_dir:
        Override the model download cache directory.
    chunk_seconds:
        ``0`` (default) runs one straight pass over the whole clip — recommended, and
        length-invariant. A positive value windows the audio into ``chunk_seconds``-long
        segments with a 2 s crossfaded overlap, trading a little seam overhead for a much
        lower peak memory footprint on very long calls (w2v-BERT attention is O(T^2)).
    precision:
        Feature-extractor weight precision: ``"fp32"`` (default, full fidelity, ~2.3 GB)
        or ``"int8"`` (~580 MB, ~4x smaller and faster but audibly lossy on this model —
        ~12 dB SNR vs fp32). ``fe_path`` overrides this.
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
        chunk_seconds: float = 0.0,
        precision: str = _DEFAULT_PRECISION,
        **cfg,
    ):
        super().__init__(**cfg)
        if precision not in _FE_FILES:
            raise ValueError(
                f"precision must be one of {sorted(_FE_FILES)}, got {precision!r}")
        self._providers = providers
        self._cache_dir = cache_dir
        self._revision = revision if revision is not None else _HF_REVISION
        self._fe_path = fe_path
        self._decoder_path = decoder_path
        self._precision = precision
        self._chunk = max(0, int(round(float(chunk_seconds) * _IN_SR)))
        self._fe = None
        self._dec = None

    def _ensure_models(self) -> None:
        if self._fe is not None:
            return
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = os.cpu_count() or 4
        providers = self._providers or ["CPUExecutionProvider"]

        def _fetch(name):
            return resolve(
                name, hf_repo=_HF_REPO, revision=self._revision, cache_dir=self._cache_dir)

        def _sess(path):
            return ort.InferenceSession(path, sess_options=opts, providers=providers)

        fe_name = _FE_FILES[self._precision]
        fe_path = self._fe_path
        if fe_path is None:
            # fp32 weights live in an external ``.onnx.data`` sidecar; fetch it into the
            # same cache dir first so onnxruntime finds it beside the graph.
            if self._precision == "fp32":
                _fetch(fe_name + ".data")
            fe_path = _fetch(fe_name)
        self._fe = _sess(fe_path)
        self._dec = _sess(self._decoder_path or _fetch(_DEC))

    def _restore_segment(self, seg: np.ndarray) -> np.ndarray:
        """Run one 16 kHz segment through FE + decoder -> 48 kHz float32 waveform."""
        seg = np.pad(seg, (_EDGE_PAD, _EDGE_PAD))
        if seg.size < _MIN_SEG:  # too few mel frames -> unstable CMVN; pad up to a safe floor
            seg = np.pad(seg, (0, _MIN_SEG - seg.size))
        feats = seamless_fbank(seg)[None]  # [1,T,160]
        hidden = self._fe.run(None, {"input_features": feats.astype(np.float32)})[0]
        wav = self._dec.run(
            None, {"features": np.ascontiguousarray(hidden.transpose(0, 2, 1))})[0]
        return wav.reshape(-1)

    def _bounds(self, n: int) -> List[Tuple[int, int]]:
        """Segment boundaries over ``n`` input samples (single pass unless chunking)."""
        if self._chunk <= 0 or n <= self._chunk:
            return [(0, n)]
        hop = max(1, self._chunk - _OVERLAP)
        bounds = [(s, min(s + self._chunk, n)) for s in range(0, n, hop)]
        return [b for b in bounds if b[1] > b[0]]

    def _upscale_array(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        self._ensure_models()
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        n_in = x.size
        if n_in == 0:
            return np.zeros(0, dtype=np.float32)

        if sample_rate != _IN_SR:
            x = np.asarray(kaiser_resample(x, sample_rate, _IN_SR), dtype=np.float32)
        n = x.size

        peak = float(np.max(np.abs(x)))
        if not np.isfinite(peak) or peak <= 1e-6:
            return np.zeros(int(round(_OUT_SR / sample_rate * n_in)), dtype=np.float32)
        x = (x / peak) * _IN_PEAK

        bounds = self._bounds(n)
        out = np.zeros(n * _RATIO + _OUT_SR, dtype=np.float32)
        wsum = np.zeros_like(out)
        for s, e in bounds:
            y = self._restore_segment(x[s:e])
            o0 = s * _RATIO
            length = min(y.size, out.size - o0)
            ramp = np.ones(length, dtype=np.float32)
            if len(bounds) > 1:                       # taper edges for a clickless crossfade
                r = min(_OVERLAP * _RATIO, length // 2)
                if r > 0:
                    ramp[:r] = np.linspace(0.0, 1.0, r, dtype=np.float32)
                    ramp[-r:] = np.linspace(1.0, 0.0, r, dtype=np.float32)
            out[o0:o0 + length] += y[:length] * ramp
            wsum[o0:o0 + length] += ramp
        valid = wsum > 1e-6
        out[valid] /= wsum[valid]

        out = out[: n * _RATIO]
        opeak = float(np.max(np.abs(out)))
        if opeak > 1e-6:
            out = (out / opeak) * _OUT_PEAK
        target = int(round(_OUT_SR / sample_rate * n_in))
        return out[:target].astype(np.float32)


register_engine(
    EngineEntry(
        alias="callenhancer",
        adapter_class=CallEnhancerAdapter,
        description=(
            "CallEnhancer call-centre speech restoration: full 24-layer w2v-BERT 2.0 "
            "feature predictor (LoRA-merged) + DAC vocoder, 16 kHz -> 48 kHz. SeamlessM4T "
            "mel front-end in numpy; fp32 feature extractor by default (int8 available, "
            "smaller but lossy). ONNX from TigreGotico/audiosronnx-callenhancer. "
            "(Scicom-intl, CC-BY-NC-4.0)"
        ),
        input_sample_rate=_IN_SR,
        output_sample_rate=_OUT_SR,
        license="CC-BY-NC-4.0",
        extras="",
        kind="sr",
    )
)
