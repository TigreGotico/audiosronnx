"""Base adapter contract and engine registry for audiosronnx.

Adding a new engine
-------------------
1. Subclass :class:`SRModel` and implement :meth:`_upscale_array`.
2. Create an :class:`EngineEntry` describing the engine.
3. Call :func:`register_engine` (usually at import time in the adapter module).
4. Auto-import the adapter module from ``audiosronnx/engines/__init__.py``.

The base class implements everything users actually call — array/path dispatch in
:meth:`upscale`, single-file :meth:`upscale_file`, and batch :meth:`upscale_dir` —
so an adapter only has to turn a mono float32 array at a known input rate into a
48 kHz mono float32 array.
"""
from __future__ import annotations

import abc
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import numpy as np

from .audio import AudioLike, read_wav, to_float32_mono, write_wav

#: Every super-resolution engine here produces 48 kHz output.
OUTPUT_SAMPLE_RATE = 48000

_AUDIO_EXTS = (".wav", ".flac", ".ogg", ".mp3", ".opus", ".m4a")


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
@dataclass
class EngineEntry:
    """Registry entry describing one super-resolution engine adapter."""

    alias: str
    adapter_class: "Type"
    description: str = ""
    #: native input sample rate the model expects (Hz). ``0`` means flexible.
    input_sample_rate: int = 16000
    output_sample_rate: int = OUTPUT_SAMPLE_RATE
    #: SPDX-ish license string, surfaced to users.
    license: str = ""
    #: pip extras key (empty when the base install already covers the engine).
    extras: str = ""
    #: task family — ``"sr"`` (bandwidth-extension/super-resolution → 48 kHz),
    #: ``"denoise"`` (noise removal, sample-rate preserving), or ``"enhance"``
    #: (holistic restoration). ``load_sr`` / ``load_denoise`` filter on this.
    kind: str = "sr"


ENGINE_REGISTRY: Dict[str, EngineEntry] = {}


def register_engine(entry: EngineEntry) -> EngineEntry:
    """Register an engine entry; returns *entry* for decorator-style use."""
    ENGINE_REGISTRY[entry.alias] = entry
    return entry


def get_engine(alias: str) -> EngineEntry:
    """Retrieve a registry entry; raises ``KeyError`` with a helpful message."""
    if alias not in ENGINE_REGISTRY:
        known = ", ".join(sorted(ENGINE_REGISTRY))
        raise KeyError(
            f"Unknown engine {alias!r}. Known engines: {known or '(none registered)'}"
        )
    return ENGINE_REGISTRY[alias]


def available_models() -> List[str]:
    """Return all registered engine aliases."""
    return sorted(ENGINE_REGISTRY)


# --------------------------------------------------------------------------- #
# Adapter base class
# --------------------------------------------------------------------------- #
class SRModel(abc.ABC):
    """Abstract base for per-engine audio super-resolution adapters.

    Subclasses implement :meth:`_upscale_array` — the pure numeric core that maps a
    mono float32 array (already resampled to the engine's input rate) to a 48 kHz
    mono float32 array.
    """

    #: Output sample rate in Hz (48 kHz for every engine).
    output_sample_rate: int = OUTPUT_SAMPLE_RATE
    #: Native input sample rate the model was trained for. ``0`` means flexible
    #: (the adapter handles arbitrary input rates itself).
    input_sample_rate: int = 16000

    def __init__(self, **cfg):
        self._cfg = cfg

    # ------------------------------------------------------------------ #
    # backend hook
    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    def _upscale_array(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Upscale mono float32 ``audio`` at ``sample_rate`` to a 48 kHz array."""

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def upscale(
        self, audio: AudioLike, sample_rate: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """Upscale audio to 48 kHz. The core low-level primitive.

        Args:
            audio: a path to an audio file, raw ``int16`` PCM bytes, or a numpy array.
            sample_rate: sample rate of ``audio``. Required for arrays/bytes; ignored
                (read from the file) when ``audio`` is a path.

        Returns:
            ``(out, 48000)`` where ``out`` is a mono float32 numpy array in [-1, 1].
        """
        if isinstance(audio, str):
            x, sr = read_wav(audio)
        else:
            if sample_rate is None:
                raise ValueError("sample_rate is required when audio is not a file path")
            x, sr = to_float32_mono(audio), int(sample_rate)

        if x.size == 0:
            return np.zeros(0, dtype=np.float32), self.output_sample_rate
        out = self._upscale_array(np.ascontiguousarray(x, dtype=np.float32), sr)
        return np.ascontiguousarray(out, dtype=np.float32), self.output_sample_rate

    def upscale_file(self, in_path: str, out_path: str) -> str:
        """Upscale one audio file and write a 48 kHz 16-bit WAV. Returns ``out_path``."""
        out, sr = self.upscale(in_path)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        write_wav(out_path, out, sr)
        return out_path

    def upscale_dir(
        self, in_dir: str, out_dir: str, *, pattern: Optional[Tuple[str, ...]] = None
    ) -> List[str]:
        """Upscale every audio file in ``in_dir`` into ``out_dir`` (as ``.wav``).

        Returns the list of written output paths.
        """
        exts = pattern or _AUDIO_EXTS
        os.makedirs(out_dir, exist_ok=True)
        written: List[str] = []
        for name in sorted(os.listdir(in_dir)):
            src = os.path.join(in_dir, name)
            if not os.path.isfile(src) or os.path.splitext(name)[1].lower() not in exts:
                continue
            dst = os.path.join(out_dir, os.path.splitext(name)[0] + ".wav")
            self.upscale_file(src, dst)
            written.append(dst)
        return written

    @property
    def sample_rate(self) -> int:
        """Output sample rate in Hz."""
        return self.output_sample_rate


class Denoiser(abc.ABC):
    """Abstract base for per-engine denoise / enhance adapters.

    Unlike :class:`SRModel`, a denoiser preserves the sample rate: it cleans the
    signal rather than extending its band. Subclasses implement
    :meth:`_denoise_array` — the pure numeric core that maps a mono float32 array
    (already resampled to the model's native rate) to a cleaned array at that same
    rate. The public :meth:`denoise` returns ``(out, rate)`` where ``rate`` is the
    model's native rate; callers resample back to their target if they need to.
    """

    #: Native sample rate the model runs at (Hz); input is resampled to this.
    input_sample_rate: int = 48000

    def __init__(self, **cfg):
        self._cfg = cfg

    @abc.abstractmethod
    def _denoise_array(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Denoise mono float32 ``audio`` at ``sample_rate`` -> array at the model rate."""

    def denoise(
        self, audio: AudioLike, sample_rate: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """Denoise audio. Returns ``(out, rate)`` at the model's native rate."""
        if isinstance(audio, str):
            x, sr = read_wav(audio)
        else:
            if sample_rate is None:
                raise ValueError("sample_rate is required when audio is not a file path")
            x, sr = to_float32_mono(audio), int(sample_rate)
        if x.size == 0:
            return np.zeros(0, dtype=np.float32), self.input_sample_rate
        out = self._denoise_array(np.ascontiguousarray(x, dtype=np.float32), sr)
        return np.ascontiguousarray(out, dtype=np.float32), self.input_sample_rate

    def denoise_file(self, in_path: str, out_path: str) -> str:
        """Denoise one audio file and write a 16-bit WAV. Returns ``out_path``."""
        out, sr = self.denoise(in_path)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        write_wav(out_path, out, sr)
        return out_path

    def denoise_dir(
        self, in_dir: str, out_dir: str, *, pattern: Optional[Tuple[str, ...]] = None
    ) -> List[str]:
        """Denoise every audio file in ``in_dir`` into ``out_dir`` (as ``.wav``)."""
        exts = pattern or _AUDIO_EXTS
        os.makedirs(out_dir, exist_ok=True)
        written: List[str] = []
        for name in sorted(os.listdir(in_dir)):
            src = os.path.join(in_dir, name)
            if not os.path.isfile(src) or os.path.splitext(name)[1].lower() not in exts:
                continue
            dst = os.path.join(out_dir, os.path.splitext(name)[0] + ".wav")
            self.denoise_file(src, dst)
            written.append(dst)
        return written

    @property
    def sample_rate(self) -> int:
        """Native sample rate in Hz."""
        return self.input_sample_rate
