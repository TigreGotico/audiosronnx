#!/usr/bin/env python3
"""Remove background noise, keeping the band intact.

    python examples/denoise.py noisy.wav clean.wav [--engine dpdfnet] [--model ...]
"""
import argparse

import soundfile as sf

from audiosronnx import available_denoisers, load_denoise


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--engine", default="dpdfnet",
                    help=f"one of: {', '.join(available_denoisers())}")
    ap.add_argument("--model", default=None,
                    help="engine-specific checkpoint, e.g. dpdfnet8 or vb")
    args = ap.parse_args()

    kwargs = {"model": args.model} if args.model else {}
    dn = load_denoise(args.engine, **kwargs)

    audio, in_rate = sf.read(args.input, dtype="float32")
    clean, out_rate = dn.denoise(audio, in_rate)

    # A denoiser returns audio at its OWN native rate, which may differ from the input.
    print(f"{args.input}: {in_rate} Hz -> {out_rate} Hz via {args.engine}")
    if out_rate != in_rate:
        print(f"  note: {args.engine} runs at {out_rate} Hz, so the input was resampled")

    sf.write(args.output, clean, out_rate)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
