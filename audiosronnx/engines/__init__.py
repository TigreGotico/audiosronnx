"""Engine adapters for audiosronnx.

Importing this package auto-imports every built-in adapter so it self-registers in
:data:`audiosronnx.base.ENGINE_REGISTRY`.
"""

from audiosronnx.base import (
    ENGINE_REGISTRY,
    EngineEntry,
    SRModel,
    available_models,
    get_engine,
    register_engine,
)

# Auto-import built-in engine adapters so they self-register.
import audiosronnx.engines.lavasr  # noqa: F401,E402
import audiosronnx.engines.novasr  # noqa: F401,E402
import audiosronnx.engines.hifiganbwe  # noqa: F401,E402
import audiosronnx.engines.apbwe  # noqa: F401,E402
import audiosronnx.engines.sidon  # noqa: F401,E402
import audiosronnx.engines.flowhigh  # noqa: F401,E402
import audiosronnx.engines.callenhancer  # noqa: F401,E402
import audiosronnx.engines.deepfilternet  # noqa: F401,E402  (denoise)
import audiosronnx.engines.dpdfnet  # noqa: F401,E402  (denoise)
import audiosronnx.engines.gtcrn  # noqa: F401,E402  (denoise)
import audiosronnx.engines.frcrn  # noqa: F401,E402  (denoise)
import audiosronnx.engines.mossformer2  # noqa: F401,E402  (denoise)
import audiosronnx.engines.mpsenet  # noqa: F401,E402  (denoise)
import audiosronnx.engines.cmgan  # noqa: F401,E402  (denoise)
import audiosronnx.engines.metadenoiser  # noqa: F401,E402  (denoise)
import audiosronnx.engines.mossformergan  # noqa: F401,E402  (denoise)

__all__ = [
    "SRModel",
    "EngineEntry",
    "ENGINE_REGISTRY",
    "register_engine",
    "get_engine",
    "available_models",
]
