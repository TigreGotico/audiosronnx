"""Audio I/O and resampling helpers (numpy core, soundfile for file I/O).

Everything in :mod:`audiosronnx` works internally with mono ``float32`` PCM in the
range ``[-1, 1]``. These helpers convert the shapes users pass in (a WAV path or a
numpy array of various dtypes) into that canonical form, and resample between rates.
"""
from __future__ import annotations

import math
import wave
from typing import Tuple, Union

import numpy as np

AudioLike = Union[str, bytes, bytearray, memoryview, np.ndarray]


def to_float32_mono(audio: Union[bytes, bytearray, memoryview, np.ndarray]) -> np.ndarray:
    """Convert audio of various dtypes/containers to mono float32 in [-1, 1].

    Accepts raw little-endian ``int16`` PCM bytes, or a numpy array of dtype
    ``int16`` / ``int32`` / ``uint8`` / ``float32`` / ``float64``. Multi-channel
    arrays (shape ``(n, channels)``) are downmixed by averaging channels.
    """
    if isinstance(audio, (bytes, bytearray, memoryview)):
        return np.frombuffer(bytes(audio), dtype="<i2").astype(np.float32) / 32768.0

    if not isinstance(audio, np.ndarray):
        raise TypeError(f"audio must be bytes or np.ndarray, got {type(audio).__name__}")

    arr = audio
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    elif arr.ndim > 2:
        raise ValueError(f"audio array must be 1-D or 2-D, got {arr.ndim}-D")

    if arr.dtype == np.float32:
        out = arr
    elif arr.dtype == np.float64:
        out = arr.astype(np.float32)
    elif arr.dtype == np.int16:
        out = arr.astype(np.float32) / 32768.0
    elif arr.dtype == np.int32:
        out = arr.astype(np.float32) / 2147483648.0
    elif arr.dtype == np.uint8:
        out = (arr.astype(np.float32) - 128.0) / 128.0
    else:
        raise TypeError(f"unsupported audio dtype: {arr.dtype}")

    return np.ascontiguousarray(out, dtype=np.float32)


def resample(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample mono float32 audio from ``src_sr`` to ``dst_sr``.

    Uses ``scipy.signal.resample_poly`` when SciPy is installed (the default for a
    full install), otherwise falls back to linear interpolation via
    :func:`numpy.interp`.
    """
    if src_sr == dst_sr or x.size == 0:
        return x.astype(np.float32, copy=False)
    try:
        from scipy.signal import resample_poly  # type: ignore

        g = math.gcd(int(src_sr), int(dst_sr))
        up, down = int(dst_sr) // g, int(src_sr) // g
        return resample_poly(x, up, down).astype(np.float32)
    except Exception:
        n_out = int(round(x.shape[0] * dst_sr / src_sr))
        if n_out <= 0:
            return np.zeros(0, dtype=np.float32)
        src_idx = np.linspace(0.0, x.shape[0] - 1, num=n_out, dtype=np.float64)
        return np.interp(src_idx, np.arange(x.shape[0]), x).astype(np.float32)


def read_wav(path: str) -> Tuple[np.ndarray, int]:
    """Read audio into mono float32. Returns ``(audio, sample_rate)``.

    Uses :mod:`soundfile` when available (handles WAV/FLAC/OGG and any subtype),
    otherwise falls back to the standard-library :mod:`wave` module for PCM WAV.
    """
    try:
        import soundfile as sf  # type: ignore

        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return np.ascontiguousarray(audio, dtype=np.float32), int(sr)
    except ImportError:
        pass

    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    if sampwidth == 2:
        arr = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sampwidth == 4:
        arr = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sampwidth == 1:
        arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported WAV sample width: {sampwidth} bytes")

    if n_channels > 1:
        arr = arr.reshape(-1, n_channels).mean(axis=1)
    return np.ascontiguousarray(arr, dtype=np.float32), sr


def write_wav(path: str, audio: np.ndarray, sample_rate: int) -> None:
    """Write mono float32 ``audio`` as 16-bit PCM WAV to ``path``."""
    audio = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    try:
        import soundfile as sf  # type: ignore

        sf.write(str(path), audio, int(sample_rate), subtype="PCM_16")
        return
    except ImportError:
        pass

    pcm = (audio * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())
