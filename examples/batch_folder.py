#!/usr/bin/env python3
"""Process every audio file in a directory.

    python examples/batch_folder.py raw/ clean/ --job denoise
    python examples/batch_folder.py raw/ wide/  --job upscale --engine novasr

Reads .wav/.flac/.ogg/.mp3/.opus/.m4a and writes .wav.
"""
import argparse
import time
from pathlib import Path

from audiosronnx import load_denoise, load_sr

AUDIO_EXTS = (".wav", ".flac", ".ogg", ".mp3", ".opus", ".m4a")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("in_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--job", choices=("denoise", "upscale"), default="denoise")
    ap.add_argument("--engine", default=None, help="defaults to the job's default engine")
    args = ap.parse_args()

    files = sorted(p for p in Path(args.in_dir).iterdir()
                   if p.suffix.lower() in AUDIO_EXTS)
    if not files:
        raise SystemExit(f"no audio found in {args.in_dir}")

    if args.job == "denoise":
        engine = load_denoise(args.engine or "dpdfnet")
        run, label = engine.denoise_file, "denoised"
    else:
        engine = load_sr(args.engine or "lavasr")
        run, label = engine.upscale_file, "upscaled"

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for i, src in enumerate(files, 1):
        dst = Path(args.out_dir) / f"{src.stem}.wav"
        run(str(src), str(dst))
        print(f"[{i}/{len(files)}] {src.name} -> {dst.name}")

    elapsed = time.perf_counter() - started
    print(f"{label} {len(files)} file(s) in {elapsed:.1f} s "
          f"({elapsed / len(files):.2f} s/file)")

    # The library also exposes denoise_dir()/upscale_dir() for the same job without the
    # per-file loop, when progress reporting is not needed.


if __name__ == "__main__":
    main()
