"""Public entry point: :func:`load_sr`."""
from __future__ import annotations

from typing import List, Optional

from .base import SRModel, available_models, get_engine

# Importing the engines package registers every built-in adapter.
from . import engines  # noqa: F401

__all__ = ["load_sr", "available_models"]

DEFAULT_ENGINE = "lavasr"


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
