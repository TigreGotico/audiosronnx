#!/usr/bin/env python3
"""Bandwidth-extend audio to 48 kHz.

    python examples/upscale.py input.wav output_48k.wav [--engine lavasr]
"""
import argparse

import soundfile as sf

from audiosronnx import available_models, load_sr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--engine", default="lavasr",
                    help=f"one of: {', '.join(available_models())}")
    args = ap.parse_args()

    sr = load_sr(args.engine)

    # The array form is the primitive: audio in, (float32 mono, 48000) out.
    audio, in_rate = sf.read(args.input, dtype="float32")
    wide, out_rate = sr.upscale(audio, in_rate)

    print(f"{args.input}: {in_rate} Hz, {len(audio) / in_rate:.2f} s")
    print(f"  -> {out_rate} Hz, {wide.size} samples via {args.engine}")

    # The file form does the same thing and writes the result.
    sr.upscale_file(args.input, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
