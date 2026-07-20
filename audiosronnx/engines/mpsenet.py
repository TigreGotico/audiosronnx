"""MP-SENet denoise adapter: ``mpsenet``.

MP-SENet (Lu et al.) denoises by predicting the **magnitude and phase spectra in parallel**
rather than a magnitude mask with the noisy phase reused — the same amplitude-plus-phase
idea as the :mod:`~audiosronnx.engines.apbwe` bandwidth extender, from the same authors. At
**2.26 M parameters / 9.7 MB** it is far smaller than frcrn or mossformer2 while still
being a full spectral model.

The graph is a clean spectral transform, with the STFT kept outside it as usual::

    amp[1, 201, T], pha[1, 201, T]  ->  amp_g[1, 201, T], pha_g[1, 201, T]

Magnitudes are **power-compressed** by ``compress_factor`` (0.3) before the model and
decompressed after — the model is trained in that domain, so skipping it produces silence.
Phase is carried as an angle, so it is reconstructed with ``amp * (cos φ + i sin φ)`` rather
than by reusing the noisy complex spectrum.

Upstream also RMS-normalises the whole utterance (``sqrt(n / Σx²)``) before the STFT and
undoes it afterwards; that is reproduced here.

Upstream publishes two checkpoints and they are **not** interchangeable. On speech
corrupted with broadband Gaussian noise (19 / 11 / 5 dB input SNR) the DNS Challenge
checkpoint recovers **+4.8 / +8.8 / +11.7 dB**, while the VoiceBank+DEMAND one manages only
+1.5 / +2.0 / +4.1 dB — it generalises poorly beyond the noise types it trained on. ``dns``
is therefore the default; ``vb`` is what the published VoiceBank+DEMAND PESQ figures were
measured with.

ONNX artifact: ``TigreGotico/audiosronnx-mpsenet`` (MIT).

Export note
-----------
The upstream model uses stock ``torch.nn.MultiheadAttention``, which specialises the
sequence length under both the TorchScript and dynamo exporters — the resulting graph fails
outright at any other length. The published graph was exported after swapping in a
shape-dynamic attention that reuses the same weights and reproduces the original
bit-identically (max abs err 0.0).

Reference
---------
- https://github.com/yxlu-0102/MP-SENet  (MIT)
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from audiosronnx._kaiser import kaiser_resample
from audiosronnx._stft import istft, stft
from audiosronnx.base import Denoiser, EngineEntry, register_engine
from audiosronnx.resolver import resolve

_HF_REPO = "TigreGotico/audiosronnx-mpsenet"
_HF_REVISION: Optional[str] = "0c310ef8265f41e131f3d8b45d3fc37e567e1616"
#: ``model`` alias -> onnx filename. Upstream publishes two checkpoints trained on
#: different corpora; they differ markedly on broadband noise (see the module docstring).
_MODELS = {"dns": "mpsenet_dns.onnx", "vb": "mpsenet.onnx"}
_DEFAULT_MODEL = "dns"

_SR = 16000
_N_FFT = 400
_HOP = 100
_COMPRESS = 0.3              # magnitude power-compression the model was trained with
#: upstream's numerical guards, kept so the compressed spectra match bit-for-bit
_MAG_EPS = 1e-9
_PHA_IM_EPS = 1e-10
_PHA_RE_EPS = 1e-5


class MPSENetAdapter(Denoiser):
    """MP-SENet 16 kHz denoiser — parallel magnitude and phase prediction.

    Parameters
    ----------
    model:
        Which upstream checkpoint to run. ``"dns"`` (default) is trained on the DNS
        Challenge data and handles broadband noise far better; ``"vb"`` is the
        VoiceBank+DEMAND checkpoint the published PESQ figures come from.
    providers:
        onnxruntime execution providers (default CPU).
    cache_dir:
        Override the model download cache directory.
    """

    input_sample_rate = _SR

    def __init__(
        self,
        *,
        model: str = _DEFAULT_MODEL,
        providers: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
        revision: Optional[str] = None,
        onnx_path: Optional[str] = None,
        **cfg,
    ):
        super().__init__(**cfg)
        if model not in _MODELS:
            raise ValueError(f"unknown mpsenet model {model!r}; choose one of {sorted(_MODELS)}")
        self._model = model
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
            _MODELS[self._model], hf_repo=_HF_REPO, revision=self._revision,
            cache_dir=self._cache_dir)
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
        norm = np.sqrt(x64.size / energy)          # upstream's utterance RMS normalisation
        x64 = x64 * norm

        spec = stft(x64, n_fft=_N_FFT, hop_size=_HOP, win_size=_N_FFT, center=True)
        mag = np.sqrt(spec.real ** 2 + spec.imag ** 2 + _MAG_EPS) ** _COMPRESS
        pha = np.arctan2(spec.imag + _PHA_IM_EPS, spec.real + _PHA_RE_EPS)

        amp_g, pha_g = self._sess.run(None, {
            "amp": mag[None].astype(np.float32),
            "pha": pha[None].astype(np.float32),
        })
        amp = np.asarray(amp_g[0], dtype=np.float64) ** (1.0 / _COMPRESS)
        ang = np.asarray(pha_g[0], dtype=np.float64)
        enhanced = amp * np.cos(ang) + 1j * amp * np.sin(ang)

        wav = istft(enhanced, n_fft=_N_FFT, hop_size=_HOP, win_size=_N_FFT,
                    center=True, length=n_out)
        return np.ascontiguousarray(wav / norm, dtype=np.float32)


register_engine(EngineEntry(
    alias="mpsenet",
    adapter_class=MPSENetAdapter,
    description=(
        "MP-SENet (Lu et al.): 16 kHz denoiser predicting magnitude and phase spectra in "
        "parallel rather than masking with the noisy phase. 2.26M params / 9.7 MB — the "
        "smallest full spectral model here. ONNX from TigreGotico/audiosronnx-mpsenet. "
        "(MIT)"
    ),
    input_sample_rate=_SR,
    output_sample_rate=_SR,
    license="MIT",
    extras="",
    kind="denoise",
))
