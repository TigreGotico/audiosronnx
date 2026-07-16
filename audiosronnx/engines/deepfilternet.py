"""DeepFilterNet3 denoise adapter: ``deepfilternet``.

DeepFilterNet3 (Schröter et al.) is a two-stage speech denoiser: a coarse ERB-band gain
mask, then *deep filtering* — a short complex FIR filter applied per low-frequency bin
across neighbouring frames — which recovers detail a real-valued mask cannot.

It runs as the three ONNX graphs DeepFilterNet's own runtime uses::

    enc      : (feat_erb[1,1,T,32], feat_spec[1,2,T,96]) -> e0..e3, emb, c0, lsnr
    erb_dec  : (emb, e3, e2, e1, e0)                     -> m[1,1,T,32]      ERB mask
    df_dec   : (emb, c0)                                 -> coefs[1,T,96,10] DF coefficients

The stages that cannot live in the graph run outside it: the STFT/ERB analysis and the
synthesis come from ``libdf`` (the ``DeepFilterLib`` wheel — Rust, no torch), while the mask
application and the deep filter are numpy. Inference itself is onnxruntime only, so nothing
here pulls torch.

ONNX artifact: ``TigreGotico/audiosronnx-deepfilternet`` (MIT).

Reference
---------
- https://github.com/Rikorose/DeepFilterNet
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from audiosronnx._kaiser import kaiser_resample
from audiosronnx.base import Denoiser, EngineEntry, register_engine
from audiosronnx.resolver import resolve

_HF_REPO = "TigreGotico/audiosronnx-deepfilternet"
_HF_REVISION: Optional[str] = None
_ENC, _ERB_DEC, _DF_DEC = "enc.onnx", "erb_dec.onnx", "df_dec.onnx"

_SR = 48000
_FFT = 960
_HOP = 480
_NB_ERB = 32          # ERB bands the mask is predicted over
_NB_DF = 96           # low-frequency bins the deep filter refines
_DF_ORDER = 5         # deep-filter taps (complex)
_DF_LOOKAHEAD = 2     # frames the deep filter looks ahead
_CONV_LOOKAHEAD = 2   # frames the encoder looks ahead
_MIN_NB_ERB_FREQS = 2
_NORM_TAU = 1.0       # exponential-norm time constant -> alpha below
_ALPHA = math.exp(-(_HOP / _SR) / _NORM_TAU)


class DeepFilterNetAdapter(Denoiser):
    """DeepFilterNet3 denoiser backed by the enc/erb_dec/df_dec ONNX graphs.

    Parameters
    ----------
    providers:
        onnxruntime execution providers (default CPU).
    cache_dir:
        Override the model download cache directory.
    mask_only:
        Skip the deep-filtering stage and return the ERB-masked signal. Cheaper, and a
        useful ablation, but leaves detail the deep filter would recover.
    """

    input_sample_rate = _SR

    def __init__(
        self,
        *,
        providers: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
        revision: Optional[str] = None,
        mask_only: bool = False,
        enc_path: Optional[str] = None,
        erb_dec_path: Optional[str] = None,
        df_dec_path: Optional[str] = None,
        **cfg,
    ):
        super().__init__(**cfg)
        try:
            from libdf import DF
        except ImportError as e:  # pragma: no cover - dependency hint
            raise ImportError(
                "the deepfilternet engine needs the DeepFilterLib wheel (libdf) for its "
                "STFT/ERB frontend: pip install 'audiosronnx[deepfilternet]'"
            ) from e
        import onnxruntime

        self._mask_only = bool(mask_only)
        rev = revision or _HF_REVISION
        providers = providers or ["CPUExecutionProvider"]

        def _sess(explicit, name):
            path = explicit or resolve(name, hf_repo=_HF_REPO, revision=rev, cache_dir=cache_dir)
            return onnxruntime.InferenceSession(path, providers=providers)

        self._enc = _sess(enc_path, _ENC)
        self._erb_dec = _sess(erb_dec_path, _ERB_DEC)
        self._df_dec = _sess(df_dec_path, _DF_DEC)

        self._df = DF(sr=_SR, fft_size=_FFT, hop_size=_HOP, nb_bands=_NB_ERB,
                      min_nb_erb_freqs=_MIN_NB_ERB_FREQS)
        self._erb_fb = self._df.erb_widths()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _pad_feat(x: np.ndarray, lookahead: int = _CONV_LOOKAHEAD) -> np.ndarray:
        """Apply the encoder's lookahead shift to a ``[B, C, T, F]`` feature block.

        DeepFilterNet pads features with ``ConstantPad2d((0, 0, -lookahead, lookahead))``
        before the encoder — a negative pad, i.e. drop ``lookahead`` frames from the front
        and append that many zero frames. It lives outside the exported graph, so without it
        every mask and coefficient lands ``lookahead`` frames off.
        """
        if lookahead <= 0:
            return x
        return np.pad(x[:, :, lookahead:, :], ((0, 0), (0, 0), (0, lookahead), (0, 0)))

    def _deep_filter(self, spec: np.ndarray, coefs: np.ndarray, frames: int) -> np.ndarray:
        """Apply the deep filter to the first ``_NB_DF`` bins of ``spec`` ``[1, T, F]``."""
        # df_dec emits [B, T, F, O, 2] (freq-major, then tap, then re/im) -> [B, O, T, F]
        c = coefs.reshape(1, frames, _NB_DF, _DF_ORDER, 2)
        c = (c[..., 0] + 1j * c[..., 1]).transpose(0, 3, 1, 2)
        low = spec[0, :, :_NB_DF]
        # unfold time into _DF_ORDER taps: pad (order-1-lookahead) before, lookahead after
        padded = np.pad(low, ((_DF_ORDER - 1 - _DF_LOOKAHEAD, _DF_LOOKAHEAD), (0, 0)))
        taps = np.stack([padded[i:i + frames] for i in range(_DF_ORDER)], axis=-1)
        filtered = np.einsum("tfn,ntf->tf", taps, c[0])
        out = spec.copy()
        out[0, :, :_NB_DF] = filtered
        return out

    def _denoise_array(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        from libdf import erb, erb_inv, erb_norm, unit_norm

        if sample_rate != _SR:
            audio = np.asarray(kaiser_resample(audio, sample_rate, _SR), dtype=np.float32)
        x = np.ascontiguousarray(audio, dtype=np.float32).reshape(1, -1)

        spec = self._df.analysis(x)                       # [1, T, F] complex
        frames = spec.shape[1]

        feat_erb = np.asarray(erb_norm(erb(spec, self._erb_fb), _ALPHA))[:, None, :, :]
        low = np.asarray(unit_norm(spec[..., :_NB_DF], _ALPHA))
        feat_spec = np.stack([low.real, low.imag], axis=1)
        feat_erb = self._pad_feat(feat_erb.astype(np.float32))
        feat_spec = self._pad_feat(feat_spec.astype(np.float32))

        e0, e1, e2, e3, emb, c0, _lsnr = self._enc.run(
            None, {"feat_erb": feat_erb, "feat_spec": feat_spec})
        mask = self._erb_dec.run(
            None, {"emb": emb, "e3": e3, "e2": e2, "e1": e1, "e0": e0})[0]

        gains = np.asarray(erb_inv(mask[:, 0, :, :].astype(np.float32), self._erb_fb))
        spec = spec * gains
        if not self._mask_only:
            coefs = self._df_dec.run(None, {"emb": emb, "c0": c0})[0]
            spec = self._deep_filter(spec, coefs, frames)

        return np.asarray(self._df.synthesis(spec), dtype=np.float32).reshape(-1)


register_engine(EngineEntry(
    alias="deepfilternet",
    adapter_class=DeepFilterNetAdapter,
    description="DeepFilterNet3 speech denoiser (ERB mask + deep filtering), 48 kHz",
    input_sample_rate=_SR,
    output_sample_rate=_SR,
    license="mit",
    extras="deepfilternet",
    kind="denoise",
))
