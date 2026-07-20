"""MossFormer2 denoise adapter: ``mossformer2``.

MossFormer2 (Alibaba / ClearerVoice-Studio) is a 55 M-parameter hybrid transformer +
recurrent mask predictor and the strongest **fullband 48 kHz** denoiser here — it reports
**PESQ 3.16 / STOI 0.95 / SI-SDR 19.38** on VoiceBank+DEMAND, and unlike frcrn and gtcrn it
does not throw away everything above 8 kHz.

The exported graph predicts a spectral mask from log-mel features::

    fbank[1, T, 180]  ->  mask[1, T, 961]

The 180 channels are 60 Kaldi mel bins plus their first- and second-order deltas. That
front-end, the mask application and the STFT/ISTFT all live outside the graph, in numpy
(:mod:`audiosronnx._kaldi_fbank` and :mod:`audiosronnx._stft`), so nothing pulls torch or
torchaudio at inference time.

Two details are load-bearing and easy to get wrong:

* The waveform is scaled by ``32768`` before the filterbank. The model was trained on
  int16-scale features, and log-mel is not scale-invariant.
* Upstream's Kaldi front-end runs with ``dither=1.0``, which makes it stochastic. This
  adapter uses ``dither=0`` so the engine is deterministic; the downstream difference is
  ~1e-5, far below the model's own precision.

Long inputs are processed with a 4 s window and 3 s stride, discarding the transient edge
of each segment, following upstream's sliding-window decode.

ONNX artifact: ``TigreGotico/audiosronnx-mossformer2`` (Apache-2.0). The graph **must** be
exported with ``do_constant_folding=False`` — see the model card; folding silently bakes in
the traced sequence length.

Reference
---------
- https://github.com/modelscope/ClearerVoice-Studio  (Apache-2.0)
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from audiosronnx._kaiser import kaiser_resample
from audiosronnx._kaldi_fbank import deltas, fbank, hamming
from audiosronnx._stft import istft, stft
from audiosronnx.base import Denoiser, EngineEntry, register_engine
from audiosronnx.resolver import resolve

_HF_REPO = "TigreGotico/audiosronnx-mossformer2"
_HF_REVISION: Optional[str] = "0d91401f480ab971bb26daa108771c5fc9c8cfeb"
_ONNX = "mossformer2_48k.onnx"

_SR = 48000
_WIN = 1920                  # 40 ms analysis window
_HOP = 384                   # 8 ms hop
_NUM_MELS = 60
_MAX_WAV = 32768.0           # the model expects int16-scale input to the filterbank
_DECODE_WINDOW_S = 4.0
_SEGMENT_AFTER_S = 20.0


class MossFormer2Adapter(Denoiser):
    """MossFormer2 fullband 48 kHz denoiser: numpy Kaldi front-end + ONNX mask predictor.

    Parameters
    ----------
    dither:
        Kaldi filterbank dither. ``0`` (default) keeps the engine deterministic; upstream
        uses ``1.0``, which changes the output by ~1e-5.
    providers:
        onnxruntime execution providers (default CPU).
    cache_dir:
        Override the model download cache directory.
    """

    input_sample_rate = _SR

    def __init__(
        self,
        *,
        dither: float = 0.0,
        providers: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
        revision: Optional[str] = None,
        onnx_path: Optional[str] = None,
        **cfg,
    ):
        super().__init__(**cfg)
        self._dither = float(dither)
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

    def _features(self, scaled: np.ndarray) -> np.ndarray:
        """Kaldi log-mel plus first/second-order deltas -> ``[1, T, 180]``."""
        mels = fbank(scaled, _SR, _WIN, _HOP, _NUM_MELS, dither=self._dither)
        d1 = deltas(mels)
        d2 = deltas(d1)
        return np.concatenate([mels, d1, d2], axis=1)[None].astype(np.float32)

    def _enhance(self, scaled: np.ndarray) -> np.ndarray:
        """Run one segment (already ``_MAX_WAV``-scaled) and return it enhanced."""
        feats = self._features(scaled)
        if feats.shape[1] == 0:            # shorter than one analysis window
            return np.zeros(scaled.size, dtype=np.float32)
        mask = self._sess.run(None, {self._sess.get_inputs()[0].name: feats})[0]

        window = hamming(_WIN)
        spec = stft(scaled, n_fft=_WIN, hop_size=_HOP, win_size=_WIN,
                    center=False, window=window)                    # [F, T]
        mask = np.transpose(mask, (2, 1, 0))                        # [F, T, 1]
        frames = min(spec.shape[1], mask.shape[1])
        masked = spec[:, :frames] * mask[:, :frames, 0]
        return istft(masked, n_fft=_WIN, hop_size=_HOP, win_size=_WIN,
                     center=False, length=scaled.size, window=window)

    def _denoise_array(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        self._ensure_models()
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sample_rate != _SR:
            x = np.asarray(kaiser_resample(x, sample_rate, _SR), dtype=np.float32)
        n_out = x.size
        if n_out == 0:
            return np.zeros(0, dtype=np.float32)

        scaled = np.asarray(x, dtype=np.float64) * _MAX_WAV
        window = int(_SR * _DECODE_WINDOW_S)
        stride = int(window * 0.75)

        if scaled.size <= _SR * _SEGMENT_AFTER_S:
            out = self._enhance(scaled)
        else:
            # slide a window, dropping the transient edge of each segment
            give_up = (window - stride) // 2
            padded = np.pad(scaled, (0, max(0, window - scaled.size % stride)))
            out = np.zeros(padded.size, dtype=np.float32)
            idx = 0
            while idx + window <= padded.size:
                seg = self._enhance(padded[idx:idx + window])
                if idx == 0:
                    out[idx:idx + window - give_up] = seg[:window - give_up]
                else:
                    out[idx + give_up:idx + window - give_up] = seg[give_up:window - give_up]
                idx += stride

        result = np.zeros(n_out, dtype=np.float32)
        usable = min(n_out, out.size)
        result[:usable] = out[:usable] / _MAX_WAV
        return result


register_engine(EngineEntry(
    alias="mossformer2",
    adapter_class=MossFormer2Adapter,
    description=(
        "MossFormer2 (ClearerVoice-Studio): 55M-param hybrid transformer/recurrent mask "
        "predictor, the strongest fullband 48 kHz denoiser here (PESQ 3.16 on "
        "VoiceBank+DEMAND). Kaldi log-mel front-end and mask application in numpy. "
        "ONNX from TigreGotico/audiosronnx-mossformer2. (Alibaba, Apache-2.0)"
    ),
    input_sample_rate=_SR,
    output_sample_rate=_SR,
    license="Apache-2.0",
    extras="",
    kind="denoise",
))
