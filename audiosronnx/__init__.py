"""audiosronnx — pure-ONNX multi-engine audio super-resolution / bandwidth extension.

Upscale low-sample-rate or low-bandwidth speech (e.g. 8 kHz telephony) to a clean
48 kHz signal. Inference runs entirely on onnxruntime — no torch at runtime.

Quick start::

    from audiosronnx import load_sr

    sr = load_sr("lavasr")                       # default engine
    out, rate = sr.upscale("telephone_8k.wav")   # -> (float32 mono array, 48000)
    sr.upscale_file("in.wav", "out_48k.wav")

Engines are downloaded on first use from Hugging Face Hub and cached under an XDG
data directory. ``load_sr("novasr")`` selects the tiny fast engine instead.
"""
from .api import load_sr, load_denoise, available_denoisers
from .base import (
    ENGINE_REGISTRY,
    OUTPUT_SAMPLE_RATE,
    Denoiser,
    EngineEntry,
    SRModel,
    available_models,
    get_engine,
    register_engine,
)
from .version import __version__

__all__ = [
    "load_sr",
    "load_denoise",
    "available_models",
    "available_denoisers",
    "SRModel",
    "Denoiser",
    "EngineEntry",
    "ENGINE_REGISTRY",
    "register_engine",
    "get_engine",
    "OUTPUT_SAMPLE_RATE",
    "__version__",
]
