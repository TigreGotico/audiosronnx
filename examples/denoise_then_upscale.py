#!/usr/bin/env python3
"""Clean a noisy narrowband recording, then extend it to 48 kHz.

Order matters. A bandwidth extender invents high-frequency content from whatever it is
given, so running it on noisy input reconstructs a high band from the noise.

    python examples/denoise_then_upscale.py noisy_8k.wav restored_48k.wav
"""
import argparse

import soundfile as sf

from audiosronnx import load_denoise, load_sr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--denoiser", default="dpdfnet")
    ap.add_argument("--upscaler", default="lavasr")
    args = ap.parse_args()

    audio, in_rate = sf.read(args.input, dtype="float32")
    print(f"in:       {in_rate} Hz, {len(audio) / in_rate:.2f} s")

    clean, clean_rate = load_denoise(args.denoiser).denoise(audio, in_rate)
    print(f"denoised: {clean_rate} Hz via {args.denoiser}")

    wide, out_rate = load_sr(args.upscaler).upscale(clean, clean_rate)
    print(f"upscaled: {out_rate} Hz via {args.upscaler}")

    sf.write(args.output, wide, out_rate)
    print(f"wrote {args.output}")

    # A restoration engine (sidon, callenhancer) denoises as part of resynthesis, so it
    # replaces both stages rather than following them.


if __name__ == "__main__":
    main()
