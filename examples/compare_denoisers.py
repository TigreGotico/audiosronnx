#!/usr/bin/env python3
"""Score every denoise engine on your own audio.

Takes a CLEAN reference, adds broadband noise at several levels, and reports the SNR each
engine recovers. Published benchmarks rank engines on their own test sets; this ranks them
on yours, which is the number that matters.

    python examples/compare_denoisers.py clean_speech.wav
    python examples/compare_denoisers.py clean.wav --engines dpdfnet,gtcrn
"""
import argparse

import numpy as np
import soundfile as sf

from audiosronnx import available_denoisers, load_denoise
from audiosronnx._kaiser import kaiser_resample


def snr_db(estimate: np.ndarray, reference: np.ndarray) -> float:
    n = min(estimate.size, reference.size)
    err = ((estimate[:n] - reference[:n]) ** 2).sum()
    return 10 * np.log10((reference[:n] ** 2).sum() / (err + 1e-20))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clean", help="a CLEAN reference recording")
    ap.add_argument("--engines", default=None, help="comma-separated subset")
    ap.add_argument("--levels", default="0.02,0.05,0.1",
                    help="noise standard deviations to test")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    engines = args.engines.split(",") if args.engines else available_denoisers()
    levels = [float(x) for x in args.levels.split(",")]
    reference, ref_rate = sf.read(args.clean, dtype="float32")
    if reference.ndim > 1:
        reference = reference.mean(axis=1)

    print(f"reference: {args.clean} ({ref_rate} Hz, {reference.size / ref_rate:.2f} s)\n")
    header = "engine".ljust(15) + "".join(f"{f'sd={l}':>12}" for l in levels)
    print(header)
    print("-" * len(header))

    for name in engines:
        try:
            engine = load_denoise(name)
            rate = engine.sample_rate
            # score against the reference at the engine's own rate
            ref = (reference if rate == ref_rate
                   else np.asarray(kaiser_resample(reference, ref_rate, rate),
                                   dtype=np.float32))
            cells = []
            for level in levels:
                rng = np.random.RandomState(args.seed)   # same noise for every engine
                noisy = (ref + level * rng.randn(ref.size)).astype(np.float32)
                out, _ = engine.denoise(noisy, rate)
                cells.append(f"{snr_db(out, ref) - snr_db(noisy, ref):+11.1f}")
            print(name.ljust(15) + "".join(cells))
        except Exception as exc:                          # missing extra, no network, ...
            print(name.ljust(15) + f"  unavailable: {str(exc)[:40]}")

    print("\nGain in dB SNR. Broadband Gaussian noise is a hostile synthetic case —")
    print("test with your real noise before trusting the ordering.")


if __name__ == "__main__":
    main()
