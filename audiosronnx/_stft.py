"""torch-faithful STFT / ISTFT in numpy (for spectral-domain engines like AP-BWE).

Reproduces ``torch.stft`` / ``torch.istft`` semantics — a ``win_length`` window centred
inside an ``n_fft`` frame, centre reflect padding, and the ISTFT window-envelope
normalisation — so a spectral model exported to ONNX keeps the exact front-end/back-end
it was trained with. Pure numpy; validated against torch to < 1e-4.
"""
from __future__ import annotations

import numpy as np


def _padded_hann(win_length: int, n_fft: int) -> np.ndarray:
    # torch.hann_window(win_length) is periodic by default: 0.5*(1 - cos(2*pi*n/win))
    n = np.arange(win_length, dtype=np.float64)
    win = 0.5 * (1.0 - np.cos(2.0 * np.pi * n / win_length))
    if win_length == n_fft:
        return win.astype(np.float64)
    pad_left = (n_fft - win_length) // 2
    out = np.zeros(n_fft, dtype=np.float64)
    out[pad_left:pad_left + win_length] = win
    return out


def stft(
    audio: np.ndarray, n_fft: int, hop_size: int, win_size: int, center: bool = True
) -> np.ndarray:
    """Return the complex STFT ``[n_fft//2+1, frames]`` matching ``torch.stft``."""
    x = np.asarray(audio, dtype=np.float64)
    win = _padded_hann(win_size, n_fft)
    if center:
        x = np.pad(x, (n_fft // 2, n_fft // 2), mode="reflect")
    n_frames = 1 + (len(x) - n_fft) // hop_size
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n_frames, n_fft),
        strides=(x.strides[0] * hop_size, x.strides[0]),
    )
    spec = np.fft.rfft(frames * win, n=n_fft, axis=1)
    return spec.T.astype(np.complex64)


def istft(
    spec: np.ndarray, n_fft: int, hop_size: int, win_size: int,
    center: bool = True, length: int | None = None,
) -> np.ndarray:
    """Inverse of :func:`stft`, matching ``torch.istft`` (window-envelope normalised)."""
    spec = np.asarray(spec, dtype=np.complex128)
    win = _padded_hann(win_size, n_fft)
    n_freq, n_frames = spec.shape
    frames = np.fft.irfft(spec.T, n=n_fft, axis=1)  # (n_frames, n_fft)
    frames = frames * win

    out_len = n_fft + hop_size * (n_frames - 1)
    ola = np.zeros(out_len, dtype=np.float64)
    win_sq = np.zeros(out_len, dtype=np.float64)
    w2 = win ** 2
    for i in range(n_frames):
        s = i * hop_size
        ola[s:s + n_fft] += frames[i]
        win_sq[s:s + n_fft] += w2
    win_sq = np.where(win_sq > 1e-11, win_sq, 1.0)
    ola = ola / win_sq

    if center:
        ola = ola[n_fft // 2: len(ola) - n_fft // 2]
    if length is not None:
        if len(ola) > length:
            ola = ola[:length]
        elif len(ola) < length:
            ola = np.pad(ola, (0, length - len(ola)))
    return ola.astype(np.float32)
