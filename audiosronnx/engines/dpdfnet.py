"""DPDFNet denoise adapter: ``dpdfnet``.

DPDFNet (Ceva) — *Dual-Path RNN-based DeepFilterNet* — is a streaming speech denoiser
built on DeepFilterNet2. Unlike the :mod:`~audiosronnx.engines.deepfilternet` adapter,
which needs ``libdf`` for its ERB/STFT front-end, DPDFNet folds the whole enhancement
stage into a single stateful ONNX graph that consumes **one STFT frame at a time**::

    spec[1, 1, F, 2] , state[S]  ->  spec_e[1, 1, F, 2] , state[S]

so the only work outside the graph is a Vorbis-windowed STFT/ISTFT, which runs in numpy
here. Nothing pulls torch or libdf.

The recurrent ``state`` vector is initialised from values the exporter embedded in the
ONNX metadata (``state_size``, ``erb_norm_init``, ``spec_norm_init``, …) rather than from
zeros — the ERB and spectral normalisation running-averages start at trained values, and
zeroing them audibly damages the first second of output.

Upstream ships variants at three sample rates and several capacities; the alias exposes
them through ``model=``, defaulting to the balanced 48 kHz model:

===================  ======  ==========================================
model                rate    notes
===================  ======  ==========================================
``dpdfnet2_48khz``   48 kHz  default — balanced, fullband
``dpdfnet8_48khz``   48 kHz  highest quality, ~1.5x the compute
``baseline``         16 kHz  fastest / lowest compute
``dpdfnet2``         16 kHz  balanced
``dpdfnet4``         16 kHz  higher quality
``dpdfnet8``         16 kHz  highest quality at 16 kHz
``dpdfnet2_8khz``    8 kHz   narrowband / telephony
``dpdfnet8_8khz``    8 kHz   narrowband, higher quality
===================  ======  ==========================================

ONNX artifact: ``TigreGotico/audiosronnx-dpdfnet`` (Apache-2.0).

Reference
---------
- https://github.com/ceva-ip/DPDFNet  (Apache-2.0)
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from audiosronnx._kaiser import kaiser_resample
from audiosronnx._stft import istft, stft, vorbis_window
from audiosronnx.base import Denoiser, EngineEntry, register_engine
from audiosronnx.resolver import resolve

_HF_REPO = "TigreGotico/audiosronnx-dpdfnet"
_HF_REVISION: Optional[str] = "e1b60686e2d6c7bdf2376bef189c670755d94a2c"

#: ``model`` alias -> (onnx filename, native sample rate)
_MODELS: Dict[str, Tuple[str, int]] = {
    "baseline": ("baseline.onnx", 16000),
    "dpdfnet2": ("dpdfnet2.onnx", 16000),
    "dpdfnet4": ("dpdfnet4.onnx", 16000),
    "dpdfnet8": ("dpdfnet8.onnx", 16000),
    "dpdfnet2_8khz": ("dpdfnet2_8khz.onnx", 8000),
    "dpdfnet8_8khz": ("dpdfnet8_8khz.onnx", 8000),
    "dpdfnet2_48khz": ("dpdfnet2_48khz_hr.onnx", 48000),
    "dpdfnet8_48khz": ("dpdfnet8_48khz_hr.onnx", 48000),
}
_DEFAULT_MODEL = "dpdfnet2_48khz"

#: the offline ISTFT path advances the output by 4 hops (== 2 windows); the noisy
#: reference is realigned by that much before attenuation-limit blending.
_ATTN_FRAME_OFFSET = 4


class DPDFNetAdapter(Denoiser):
    """DPDFNet streaming denoiser driven frame-by-frame through one ONNX graph.

    Parameters
    ----------
    model:
        Which published variant to run (see the table in the module docstring).
        Determines the native sample rate; input is resampled to it and the result
        is returned at that rate.
    attn_limit_db:
        Cap the maximum attenuation, in dB, by blending the noisy signal back in
        (``None`` = unlimited, the default). A finite value such as ``12`` keeps a
        floor of residual noise, which sounds more natural than aggressive gating.
    providers:
        onnxruntime execution providers (default CPU).
    cache_dir:
        Override the model download cache directory.
    """

    def __init__(
        self,
        *,
        model: str = _DEFAULT_MODEL,
        attn_limit_db: Optional[float] = None,
        providers: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
        revision: Optional[str] = None,
        onnx_path: Optional[str] = None,
        **cfg,
    ):
        super().__init__(**cfg)
        if model not in _MODELS:
            raise ValueError(
                f"unknown dpdfnet model {model!r}; choose one of {sorted(_MODELS)}")
        if attn_limit_db is not None:
            attn_limit_db = float(attn_limit_db)
            if not np.isfinite(attn_limit_db) and attn_limit_db != np.inf:
                raise ValueError("attn_limit_db must be non-negative, inf, or None")
            if attn_limit_db < 0:
                raise ValueError("attn_limit_db must be non-negative, inf, or None")
        self._model = model
        self._attn_limit_db = attn_limit_db
        self._providers = providers
        self._cache_dir = cache_dir
        self._revision = revision if revision is not None else _HF_REVISION
        self._onnx_path = onnx_path
        self.input_sample_rate = _MODELS[model][1]
        self._sess = None
        self._init_state: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    def _ensure_models(self) -> None:
        if self._sess is not None:
            return
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = os.cpu_count() or 4
        name = _MODELS[self._model][0]
        path = self._onnx_path or resolve(
            name, hf_repo=_HF_REPO, revision=self._revision, cache_dir=self._cache_dir)
        self._sess = ort.InferenceSession(
            path, sess_options=opts, providers=self._providers or ["CPUExecutionProvider"])
        self._init_state = _initial_state(self._sess)

    @property
    def _win_len(self) -> int:
        """Model window length, read back from the graph's frequency-bin count."""
        bins = self._sess.get_inputs()[0].shape[-2]
        return int((int(bins) - 1) * 2)

    def _denoise_array(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        self._ensure_models()
        sr = self.input_sample_rate
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sample_rate != sr:
            x = np.asarray(kaiser_resample(x, sample_rate, sr), dtype=np.float32)
        n_out = x.size
        if n_out == 0:
            return np.zeros(0, dtype=np.float32)

        win_len = self._win_len
        hop = win_len // 2
        window = vorbis_window(win_len)

        # upstream pads one window of silence so the tail flushes through the RNN
        padded = np.pad(x, (0, win_len))
        spec = stft(padded, n_fft=win_len, hop_size=hop, win_size=win_len,
                    center=True, window=window).T           # [T, F] complex
        noisy = np.stack([spec.real, spec.imag], axis=-1).astype(np.float32)  # [T,F,2]

        in_spec, in_state = (i.name for i in self._sess.get_inputs()[:2])
        out_spec, out_state = (o.name for o in self._sess.get_outputs()[:2])
        state = self._init_state.copy()
        frames = np.empty_like(noisy)
        for t in range(noisy.shape[0]):
            frame = np.ascontiguousarray(noisy[None, t: t + 1], dtype=np.float32)
            enh, state = self._sess.run([out_spec, out_state],
                                        {in_spec: frame, in_state: state})
            frames[t] = enh[0, 0]

        frames = _apply_attn_limit(noisy, frames, self._attn_limit_db)
        wav = istft((frames[..., 0] + 1j * frames[..., 1]).T, n_fft=win_len,
                    hop_size=hop, win_size=win_len, center=True, window=window)
        # the synthesis path runs two windows behind the input; drop that lead-in
        wav = np.concatenate([wav[win_len * 2:], np.zeros(win_len * 2, dtype=np.float32)])
        out = np.zeros(n_out, dtype=np.float32)
        out[:min(n_out, wav.size)] = wav[:n_out]
        return out


def _initial_state(session) -> np.ndarray:
    """Build the recurrent state vector from the values embedded in the ONNX metadata.

    The ERB and spectral normalisation running-averages must start at their trained
    values; starting from zeros audibly corrupts the first second of output.
    """
    meta = session.get_modelmeta().custom_metadata_map
    try:
        size = int(meta["state_size"])
        erb_n = int(meta["erb_norm_state_size"])
        spec_n = int(meta["spec_norm_state_size"])
        erb_init = np.fromstring(meta["erb_norm_init"], sep=",", dtype=np.float32)
        spec_init = np.fromstring(meta["spec_norm_init"], sep=",", dtype=np.float32)
    except KeyError as e:  # pragma: no cover - malformed/foreign graph
        raise ValueError(
            f"dpdfnet ONNX is missing state-init metadata key {e}; re-export the model"
        ) from e
    state = np.zeros(size, dtype=np.float32)
    state[:erb_n] = erb_init
    state[erb_n:erb_n + spec_n] = spec_init
    return state


def _apply_attn_limit(
    noisy: np.ndarray, enhanced: np.ndarray, attn_limit_db: Optional[float]
) -> np.ndarray:
    """Blend the noisy spectrum back in so attenuation never exceeds ``attn_limit_db``."""
    if attn_limit_db is None or not np.isfinite(attn_limit_db):
        return enhanced
    aligned = np.zeros_like(noisy)
    if noisy.shape[0] > _ATTN_FRAME_OFFSET:
        aligned[_ATTN_FRAME_OFFSET:] = noisy[:-_ATTN_FRAME_OFFSET]
    alpha = float(10.0 ** (-attn_limit_db / 20.0))
    return (alpha * aligned + (1.0 - alpha) * enhanced).astype(np.float32)


register_engine(EngineEntry(
    alias="dpdfnet",
    adapter_class=DPDFNetAdapter,
    description=(
        "DPDFNet (Ceva): dual-path-RNN DeepFilterNet2 streaming denoiser as a single "
        "stateful ONNX graph, 8/16/48 kHz variants. Vorbis STFT in numpy, no libdf. "
        "ONNX from TigreGotico/audiosronnx-dpdfnet. (Apache-2.0)"
    ),
    input_sample_rate=48000,
    output_sample_rate=48000,
    license="Apache-2.0",
    extras="",
    kind="denoise",
))
