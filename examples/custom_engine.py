#!/usr/bin/env python3
"""Register a third-party engine so it works like a built-in.

Runs a deliberately trivial "engine" (linear interpolation) to keep the example about the
registry rather than about a model.

    python examples/custom_engine.py input.wav output_48k.wav
"""
import argparse

import numpy as np

from audiosronnx import EngineEntry, SRModel, available_models, load_sr, register_engine


class LinearUpsampler(SRModel):
    """Resample to 48 kHz by linear interpolation. A baseline, not an enhancer."""

    input_sample_rate = 0        # 0 means "accepts any input rate"
    output_sample_rate = 48000

    def _upscale_array(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        target = int(round(audio.size * self.output_sample_rate / sample_rate))
        src = np.arange(audio.size, dtype=np.float64)
        dst = np.linspace(0, audio.size - 1, target)
        return np.interp(dst, src, audio).astype(np.float32)


register_engine(EngineEntry(
    alias="linear",
    adapter_class=LinearUpsampler,
    description="Linear interpolation to 48 kHz — a baseline, no bandwidth extension",
    input_sample_rate=0,
    output_sample_rate=48000,
    license="Apache-2.0",
    kind="sr",
))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    args = ap.parse_args()

    assert "linear" in available_models()
    print(f"registered engines: {', '.join(available_models())}")

    # It now loads through the same entry point as any built-in engine.
    sr = load_sr("linear")
    sr.upscale_file(args.input, args.output)
    print(f"wrote {args.output} at {sr.sample_rate} Hz")
    print("A real engine would run an ONNX session here — see docs/custom-engines.md")


if __name__ == "__main__":
    main()
