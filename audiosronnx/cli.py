"""``audiosronnx`` command-line interface.

Subcommands:
  * ``list``                     — list registered engines
  * ``probe <wav>``              — print an audio file's format/duration
  * ``upscale <in> <out>``       — upscale one file to 48 kHz
  * ``upscale-dir <in> <out>``   — upscale every file in a directory
  * ``denoise <in> <out>``       — remove background noise from one file
  * ``denoise-dir <in> <out>``   — denoise every file in a directory
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .api import (
    DEFAULT_DENOISER,
    DEFAULT_ENGINE,
    available_denoisers,
    load_denoise,
    load_sr,
)
from .audio import read_wav
from .base import ENGINE_REGISTRY, available_models, get_engine


def _describe(name: str) -> None:
    entry = get_engine(name)
    in_sr = "flexible" if entry.input_sample_rate == 0 else f"{entry.input_sample_rate} Hz"
    lic = f" [{entry.license}]" if entry.license else ""
    extra = f" (needs the '{entry.extras}' extra)" if entry.extras else ""
    print(f"{name}{lic}  ({in_sr} -> {entry.output_sample_rate} Hz){extra}")
    if entry.description:
        print(f"    {entry.description}")


def _cmd_list(args) -> int:
    denoisers = set(available_denoisers())
    sr_engines = [n for n in available_models() if n not in denoisers]

    print("super-resolution / bandwidth extension  (load_sr, audiosronnx upscale)")
    for name in sr_engines:
        _describe(name)
    print()
    print("denoising  (load_denoise, audiosronnx denoise)")
    for name in sorted(denoisers):
        _describe(name)
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


def _cmd_denoise(args) -> int:
    dn = load_denoise(args.engine, **_denoise_kwargs(args))
    out_path = dn.denoise_file(args.input, args.output)
    print(f"wrote {out_path} ({dn.sample_rate} Hz)")
    return 0


def _cmd_denoise_dir(args) -> int:
    dn = load_denoise(args.engine, **_denoise_kwargs(args))
    written = dn.denoise_dir(args.in_dir, args.out_dir)
    for path in written:
        print(path)
    print(f"denoised {len(written)} file(s) -> {args.out_dir}", file=sys.stderr)
    return 0


def _engine_kwargs(args) -> dict:
    kw = {}
    if getattr(args, "denoise", False):
        kw["denoise"] = True
    if getattr(args, "precision", None):
        kw["precision"] = args.precision
    return kw


def _denoise_kwargs(args) -> dict:
    """Only forward ``--model``; engines that do not take one would reject it."""
    kw = {}
    if getattr(args, "model", None):
        kw["model"] = args.model
    return kw


def _add_denoise_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--engine", default=DEFAULT_DENOISER,
                        help=f"denoise engine (default: {DEFAULT_DENOISER})")
    parser.add_argument("--model", default=None,
                        help="engine-specific checkpoint, e.g. dpdfnet8 or vb")


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
    pu.add_argument("--engine", default=DEFAULT_ENGINE)
    pu.add_argument("--denoise", action="store_true", help="LavaSR: run the denoiser")
    pu.add_argument("--precision", choices=("fp32", "int8"), default=None,
                    help="CallEnhancer: feature-extractor precision")
    pu.set_defaults(func=_cmd_upscale)

    pd = sub.add_parser("upscale-dir", help="upscale every file in a directory")
    pd.add_argument("in_dir")
    pd.add_argument("out_dir")
    pd.add_argument("--engine", default=DEFAULT_ENGINE)
    pd.add_argument("--denoise", action="store_true", help="LavaSR: run the denoiser")
    pd.add_argument("--precision", choices=("fp32", "int8"), default=None,
                    help="CallEnhancer: feature-extractor precision")
    pd.set_defaults(func=_cmd_upscale_dir)

    dn = sub.add_parser("denoise", help="remove background noise from one file")
    dn.add_argument("input")
    dn.add_argument("output")
    _add_denoise_args(dn)
    dn.set_defaults(func=_cmd_denoise)

    dd = sub.add_parser("denoise-dir", help="denoise every file in a directory")
    dd.add_argument("in_dir")
    dd.add_argument("out_dir")
    _add_denoise_args(dd)
    dd.set_defaults(func=_cmd_denoise_dir)
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
