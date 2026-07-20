"""Public entry point: :func:`load_sr`."""
from __future__ import annotations

from typing import List, Optional

from .base import (
    Denoiser,
    SRModel,
    available_models,
    get_engine,
    ENGINE_REGISTRY,
)

# Importing the engines package registers every built-in adapter.
from . import engines  # noqa: F401

__all__ = ["load_sr", "load_denoise", "available_models", "available_denoisers"]

DEFAULT_ENGINE = "lavasr"
# dpdfnet needs no optional dependencies, so a bare load_denoise() works on a base
# install; deepfilternet would raise ImportError without the `deepfilternet` extra.
DEFAULT_DENOISER = "dpdfnet"


def load_sr(
    engine: str = DEFAULT_ENGINE,
    *,
    providers: Optional[List[str]] = None,
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
    **engine_kwargs,
) -> SRModel:
    """Load an audio super-resolution engine behind the unified :class:`SRModel` API.

    Args:
        engine: registry alias (``"lavasr"`` (default) or ``"novasr"``).
        providers: onnxruntime execution providers (default CPU).
        cache_dir: override the model download cache directory.
        revision: pin a HuggingFace revision for the model weights.
        **engine_kwargs: forwarded to the engine adapter (e.g. ``denoise=True``,
            ``cutoff_hz=6000`` for LavaSR).

    Returns:
        A ready-to-use :class:`~audiosronnx.base.SRModel`.

    Example:
        >>> from audiosronnx import load_sr
        >>> sr = load_sr("lavasr")
        >>> out, rate = sr.upscale("telephone_8k.wav")   # -> (float32 array, 48000)
    """
    entry = get_engine(engine)
    return entry.adapter_class(
        providers=providers, cache_dir=cache_dir, revision=revision, **engine_kwargs
    )


def available_denoisers() -> List[str]:
    """Return the aliases of every registered denoise / enhance engine."""
    return sorted(a for a, e in ENGINE_REGISTRY.items() if e.kind in ("denoise", "enhance"))


def load_denoise(
    engine: str = DEFAULT_DENOISER,
    *,
    providers: Optional[List[str]] = None,
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
    **engine_kwargs,
) -> Denoiser:
    """Load a denoise / enhance engine behind the unified :class:`Denoiser` API.

    Args:
        engine: registry alias (``"deepfilternet"`` (default), ``"frcrn"``, …).
        providers: onnxruntime execution providers (default CPU).
        cache_dir: override the model download cache directory.
        revision: pin a HuggingFace revision for the model weights.
        **engine_kwargs: forwarded to the engine adapter.

    Returns:
        A ready-to-use :class:`~audiosronnx.base.Denoiser`.

    Example:
        >>> from audiosronnx import load_denoise
        >>> dn = load_denoise("deepfilternet")
        >>> clean, rate = dn.denoise("noisy.wav")
    """
    entry = get_engine(engine)
    if entry.kind not in ("denoise", "enhance"):
        raise ValueError(
            f"{engine!r} is a {entry.kind!r} engine; use load_sr() for it."
        )
    return entry.adapter_class(
        providers=providers, cache_dir=cache_dir, revision=revision, **engine_kwargs
    )
