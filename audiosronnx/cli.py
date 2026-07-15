"""``audiosronnx`` command-line interface.

Subcommands:
  * ``list``                     — list registered engines
  * ``probe <wav>``              — print an audio file's format/duration
  * ``upscale <in> <out>``       — upscale one file to 48 kHz
  * ``upscale-dir <in> <out>``   — upscale every file in a directory
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .api import load_sr
from .audio import read_wav
from .base import available_models, get_engine


def _cmd_list(args) -> int:
    for name in available_models():
        e = get_engine(name)
        in_sr = "flexible" if e.input_sample_rate == 0 else f"{e.input_sample_rate} Hz"
        lic = f" [{e.license}]" if e.license else ""
        print(f"{name}{lic}  ({in_sr} -> {e.output_sample_rate} Hz)")
        if e.description:
            print(f"    {e.description}")
    return 0


def _cmd_probe(args) -> int:
    audio, sr = read_wav(args.wav)
    dur = len(audio) / sr if sr else 0.0
    peak = float(abs(audio).max()) if audio.size else 0.0
    print(f"file: {args.wav}")
    print(f"  sample_rate: {sr} Hz")
    print(f"  samples:     {len(audio)}")
    print(f"  duration:    {dur:.3f} s")
    print(f"  peak:        {peak:.4f}")
    return 0


def _cmd_upscale(args) -> int:
    sr = load_sr(args.engine, **_engine_kwargs(args))
    out_path = sr.upscale_file(args.input, args.output)
    print(f"wrote {out_path} ({sr.output_sample_rate} Hz)")
    return 0


def _cmd_upscale_dir(args) -> int:
    sr = load_sr(args.engine, **_engine_kwargs(args))
    written = sr.upscale_dir(args.in_dir, args.out_dir)
    for path in written:
        print(path)
    print(f"upscaled {len(written)} file(s) -> {args.out_dir}", file=sys.stderr)
    return 0


def _engine_kwargs(args) -> dict:
    kw = {}
    if getattr(args, "denoise", False):
        kw["denoise"] = True
    return kw


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="audiosronnx", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list available engines").set_defaults(func=_cmd_list)

    pp = sub.add_parser("probe", help="print an audio file's format and duration")
    pp.add_argument("wav")
    pp.set_defaults(func=_cmd_probe)

    pu = sub.add_parser("upscale", help="upscale one audio file to 48 kHz")
    pu.add_argument("input")
    pu.add_argument("output")
    pu.add_argument("--engine", default="lavasr")
    pu.add_argument("--denoise", action="store_true", help="LavaSR: run the denoiser")
    pu.set_defaults(func=_cmd_upscale)

    pd = sub.add_parser("upscale-dir", help="upscale every file in a directory")
    pd.add_argument("in_dir")
    pd.add_argument("out_dir")
    pd.add_argument("--engine", default="lavasr")
    pd.add_argument("--denoise", action="store_true", help="LavaSR: run the denoiser")
    pd.set_defaults(func=_cmd_upscale_dir)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    # line-buffered stdout so piped/streamed output appears promptly
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
