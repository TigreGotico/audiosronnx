"""Benchmark CLI: ``python -m benchmarks.run --kind {denoise,sr,enhance}``.

Runs one track's board and writes a summary plus a per-file JSONL under
``benchmarks/results/``. The results directory is git-ignored; only summaries
explicitly promoted with ``--promote`` land in ``results/published/``, which is
the *only* thing :mod:`benchmarks.render` reads. Promotion is deliberate: a docs
table changes because someone chose to publish a run, never as a side effect.

Examples
--------
Smoke a couple of small denoisers on four files::

    python -m benchmarks.run --kind denoise --engines gtcrn,dpdfnet --limit 4

Full denoise board, then publish it and regenerate the docs::

    python -m benchmarks.run --kind denoise --limit 824 --promote
    python -m benchmarks.render
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from . import harness

RESULTS_DIR = Path(__file__).resolve().parent / "results"
PUBLISHED_DIR = RESULTS_DIR / "published"


def _progress(msg: str) -> None:
    print(f"  … {msg}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True, choices=["denoise", "sr", "enhance"])
    ap.add_argument("--engines", default=None,
                    help="comma-separated subset (default: every engine of this kind)")
    ap.add_argument("--limit", type=int, default=24, help="number of files to score")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--input-rate", type=int, default=8000, choices=[8000, 16000],
                    help="narrowband input rate for the sr track")
    ap.add_argument("--skip-nc", action="store_true",
                    help="exclude non-commercially-licensed engines")
    ap.add_argument("--out", default=None, help="results directory (default benchmarks/results)")
    ap.add_argument("--promote", action="store_true",
                    help="copy the summary into results/published/ (what render.py reads)")
    args = ap.parse_args(argv)

    engines = args.engines.split(",") if args.engines else None
    out_dir = Path(args.out) if args.out else RESULTS_DIR

    summary, results = harness.run_board(
        args.kind, engines=engines, limit=args.limit, seed=args.seed,
        input_rate=args.input_rate, skip_nc=args.skip_nc, progress=_progress,
    )
    summary_path, jsonl_path = harness.write_results(summary, results, out_dir)
    print(f"summary → {summary_path}")
    print(f"per-file → {jsonl_path}")

    # human-readable recap
    prim = summary["primary"]
    print(f"\n{args.kind} board (primary {prim}):")
    rows = sorted(summary["engines"],
                  key=lambda e: e["metrics"].get(prim, float("-inf")), reverse=True)
    for e in rows:
        val = e["metrics"].get(prim)
        cell = f"{val:.3f}" if isinstance(val, (int, float)) else e["status"]
        rtf = e.get("rtf_median")
        rtf_s = f"RTF {rtf:.3f}" if rtf else ""
        print(f"  {e['alias']:<16} {prim}={cell:<8} {rtf_s}")

    if args.promote:
        PUBLISHED_DIR.mkdir(parents=True, exist_ok=True)
        dest = PUBLISHED_DIR / f"{args.kind}.summary.json"
        shutil.copyfile(summary_path, dest)
        print(f"\npromoted → {dest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
