"""The scoring loop: enumerate engines, run them on a track, score with metrics.

The harness never hard-codes an engine list. It enumerates ``ENGINE_REGISTRY``
filtered by ``kind`` so a newly-registered engine joins the board automatically,
and an engine whose optional dependency or weights are missing is recorded as a
``not installed`` / ``unavailable`` row rather than crashing the run — a board
that aborts on the first missing extra is useless. Engines still under review and
absent from the registry (e.g. unmerged branches) are simply never enumerated.

Metrics come from ``speechonnxmetrics``. No-reference MOS predictors (DNSMOS,
SIGMOS) score the output alone; intrusive metrics score it against the clean
reference at one common rate per track, so every engine is compared on the same
footing. Speaker similarity, when the ``speaker`` extra is installed, guards
against an engine that scores well by resynthesising a different-sounding voice.
"""
from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

import audiosronnx.resolver as _resolver
from audiosronnx._kaiser import kaiser_resample
from audiosronnx.base import ENGINE_REGISTRY

from . import datasets as _datasets

# --------------------------------------------------------------------------- #
# Per-track metric plan
# --------------------------------------------------------------------------- #
@dataclass
class TrackPlan:
    kind: str
    primary: str  # flattened metric key, e.g. "dnsmos.ovrl"
    noref_metrics: list[str]  # scored on the output alone
    intrusive_metrics: list[str]  # scored output-vs-reference
    intrusive_rate: int  # common rate for intrusive scoring
    output_sr_for_noref: Optional[int] = None  # None → use the engine's own output rate
    speaker_guard: bool = True


TRACK_PLANS: dict[str, TrackPlan] = {
    "denoise": TrackPlan(
        kind="denoise",
        primary="dnsmos.ovrl",
        noref_metrics=["dnsmos", "dnsmos_p808"],
        intrusive_metrics=["si_sdr", "stoi", "estoi"],
        intrusive_rate=16000,
    ),
    "sr": TrackPlan(
        kind="sr",
        primary="sigmos.ovrl",
        noref_metrics=["sigmos"],
        intrusive_metrics=["lsd", "mel_l1"],
        intrusive_rate=48000,
        output_sr_for_noref=48000,
        speaker_guard=False,
    ),
    "enhance": TrackPlan(
        kind="enhance",
        primary="dnsmos.ovrl",
        noref_metrics=["dnsmos", "sigmos"],
        intrusive_metrics=["stoi", "mcd", "log_f0_rmse"],
        intrusive_rate=16000,
    ),
}


# --------------------------------------------------------------------------- #
# Engine size probe: capture the ONNX paths resolve() hands back
# --------------------------------------------------------------------------- #
class _SizeProbe:
    """Wrap ``resolver.resolve`` to record every ONNX file an engine loads.

    Model size is then the on-disk sum of those files (weights plus any
    ``.onnx.data`` sidecar) — read straight from the resolver cache, so it is the
    real footprint, not a hand-maintained number.
    """

    def __init__(self):
        self._orig = _resolver.resolve
        self.paths: set[str] = set()

    def __enter__(self):
        probe = self

        def wrapped(*a, **k):
            p = probe._orig(*a, **k)
            if p:
                probe.paths.add(p)
            return p

        _resolver.resolve = wrapped
        # engines import resolve by name, so patch their module bindings too
        import audiosronnx.engines as eng_pkg

        self._patched = []
        for mod in list(vars(eng_pkg).values()):
            if hasattr(mod, "resolve") and getattr(mod, "resolve") is self._orig:
                mod.resolve = wrapped
                self._patched.append(mod)
        return self

    def __exit__(self, *exc):
        _resolver.resolve = self._orig
        for mod in getattr(self, "_patched", []):
            mod.resolve = self._orig

    def total_bytes(self) -> Optional[int]:
        total = 0
        found = False
        for p in self.paths:
            for cand in (p, p + ".data"):
                if os.path.isfile(cand):
                    total += os.path.getsize(cand)
                    found = True
        return total if found else None


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _score_sample(plan: TrackPlan, out: np.ndarray, out_sr: int,
                  ref: Optional[np.ndarray], ref_sr: Optional[int]) -> dict:
    import speechonnxmetrics as sm

    row: dict[str, float] = {}
    noref_sr = plan.output_sr_for_noref or out_sr
    noref_in = out if noref_sr == out_sr else kaiser_resample(out, out_sr, noref_sr)
    try:
        row.update(sm.score(noref_in, plan.noref_metrics, sr=noref_sr))
    except Exception as exc:  # noqa: BLE001
        row["_noref_error"] = str(exc)

    if ref is not None and plan.intrusive_metrics:
        r = plan.intrusive_rate
        out_r = out if out_sr == r else kaiser_resample(out, out_sr, r)
        ref_r = ref if ref_sr == r else kaiser_resample(ref, ref_sr, r)
        n = min(out_r.shape[0], ref_r.shape[0])
        try:
            row.update(sm.score(out_r[:n], plan.intrusive_metrics, ref=ref_r[:n], sr=r))
        except Exception as exc:  # noqa: BLE001
            row["_intrusive_error"] = str(exc)

    if plan.speaker_guard and ref is not None:
        try:
            from speechonnxmetrics.speaker import speaker_similarity

            row["speaker_similarity"] = float(
                speaker_similarity(out, out_sr, ref=ref, ref_sr=ref_sr)
            )
        except Exception:  # noqa: BLE001 - optional extra / weights absent
            pass
    return row


# --------------------------------------------------------------------------- #
# Engine loading
# --------------------------------------------------------------------------- #
def _load_engine(alias: str, kind: str, providers, cache_dir):
    from audiosronnx import load_denoise, load_sr

    if kind == "sr":
        return load_sr(alias, providers=providers, cache_dir=cache_dir)
    return load_denoise(alias, providers=providers, cache_dir=cache_dir)


def _run_engine(engine, kind: str, audio: np.ndarray, sr: int):
    if kind == "sr":
        return engine.upscale(audio, sr)
    return engine.denoise(audio, sr)


# --------------------------------------------------------------------------- #
# Board run
# --------------------------------------------------------------------------- #
@dataclass
class EngineResult:
    alias: str
    kind: str
    license: str
    status: str = "ok"
    note: str = ""
    n_files: int = 0
    model_bytes: Optional[int] = None
    rtf_median: Optional[float] = None
    load_seconds: Optional[float] = None
    metrics: dict[str, float] = field(default_factory=dict)
    per_file: list[dict] = field(default_factory=list)


def engines_for_kind(kind: str, only: Optional[list[str]]) -> list[str]:
    """Registry aliases for a track. ``sr`` includes ``enhance`` engines too
    (they resynthesise and can be run on the extension board, ranked separately);
    ``enhance`` selects the holistic-restoration engines specifically."""
    if kind == "enhance":
        aliases = [a for a, e in ENGINE_REGISTRY.items() if e.kind == "enhance"]
    elif kind == "sr":
        aliases = [a for a, e in ENGINE_REGISTRY.items() if e.kind == "sr"]
    else:  # denoise — denoise + enhance engines both clean noise
        aliases = [a for a, e in ENGINE_REGISTRY.items() if e.kind in ("denoise", "enhance")]
    if only:
        aliases = [a for a in aliases if a in only]
    return sorted(aliases)


def benchmark_engine(alias: str, plan: TrackPlan, samples: list, *,
                     providers=None, cache_dir=None,
                     progress: Optional[Callable[[str], None]] = None) -> EngineResult:
    entry = ENGINE_REGISTRY[alias]
    res = EngineResult(alias=alias, kind=entry.kind, license=entry.license)

    probe = _SizeProbe()
    try:
        with probe:
            t0 = time.perf_counter()
            engine = _load_engine(alias, plan.kind, providers, cache_dir)
            # first run forces the (lazy) download + session build → load time
            first = samples[0]
            out, out_sr = _run_engine(engine, plan.kind, first.degraded, first.degraded_sr)
            res.load_seconds = round(time.perf_counter() - t0, 3)
    except ImportError as exc:
        res.status = "not installed"
        res.note = str(exc)[:120]
        return res
    except Exception as exc:  # noqa: BLE001 - missing weights / no network / OOM
        res.status = "unavailable"
        res.note = f"{type(exc).__name__}: {exc}"[:160]
        return res

    res.model_bytes = probe.total_bytes()
    rtfs: list[float] = []
    accum: dict[str, list[float]] = {}

    def _consume(sample, out, out_sr, proc_t):
        dur = sample.degraded.shape[0] / max(sample.degraded_sr, 1)
        if dur > 0:
            rtfs.append(proc_t / dur)
        scores = _score_sample(plan, out, out_sr, sample.ref, sample.ref_sr)
        rec = {"id": sample.id, **{k: v for k, v in scores.items()}}
        res.per_file.append(rec)
        for k, v in scores.items():
            if isinstance(v, (int, float)) and not k.startswith("_"):
                accum.setdefault(k, []).append(float(v))

    # first sample already processed above (its time includes cold load, so it is
    # excluded from the RTF median but its scores still count)
    _score_first = _score_sample(plan, out, out_sr, first.ref, first.ref_sr)
    res.per_file.append({"id": first.id, **_score_first})
    for k, v in _score_first.items():
        if isinstance(v, (int, float)) and not k.startswith("_"):
            accum.setdefault(k, []).append(float(v))

    for sample in samples[1:]:
        try:
            t = time.perf_counter()
            out, out_sr = _run_engine(engine, plan.kind, sample.degraded, sample.degraded_sr)
            proc_t = time.perf_counter() - t
        except Exception as exc:  # noqa: BLE001
            res.per_file.append({"id": sample.id, "_error": str(exc)[:120]})
            continue
        _consume(sample, out, out_sr, proc_t)
        if progress:
            progress(f"{alias}: {len(res.per_file)}/{len(samples)}")

    res.n_files = sum(1 for r in res.per_file if "_error" not in r)
    res.rtf_median = round(statistics.median(rtfs), 4) if rtfs else None
    res.metrics = {k: round(statistics.fmean(v), 4) for k, v in accum.items() if v}
    return res


def run_board(kind: str, *, engines: Optional[list[str]], limit: int, seed: int,
              input_rate: int, skip_nc: bool, providers=None, cache_dir=None,
              progress: Optional[Callable[[str], None]] = None) -> dict:
    """Run every engine of ``kind`` over a freshly-materialised sample list."""
    plan = TRACK_PLANS[kind]
    aliases = engines_for_kind(kind, engines)
    if skip_nc:
        aliases = [a for a in aliases if "NC" not in ENGINE_REGISTRY[a].license.upper()]

    if progress:
        progress(f"loading {limit} {kind} samples…")
    samples = list(_datasets.iter_samples(kind, limit, seed, input_rate))
    if not samples:
        raise RuntimeError(f"no samples materialised for kind={kind!r}")

    results = []
    for alias in aliases:
        if progress:
            progress(f"engine {alias}")
        results.append(benchmark_engine(alias, plan, samples, providers=providers,
                                        cache_dir=cache_dir, progress=progress))

    summary = build_summary(kind, plan, results, limit=limit, seed=seed, input_rate=input_rate)
    return summary, results


def build_summary(kind: str, plan: TrackPlan, results: list[EngineResult], *,
                  limit: int, seed: int, input_rate: int) -> dict:
    import datetime as _dt

    return {
        "kind": kind,
        "primary": plan.primary,
        "provenance": {
            **_datasets.DATASET_PROVENANCE[kind],
            "seed": seed,
            "limit": limit,
            "input_rate": input_rate if kind == "sr" else None,
            "date": _dt.date.today().isoformat(),
            "metrics_lib": "speechonnxmetrics",
        },
        "engines": [
            {
                "alias": r.alias,
                "kind": r.kind,
                "license": r.license,
                "status": r.status,
                "note": r.note,
                "n_files": r.n_files,
                "model_bytes": r.model_bytes,
                "rtf_median": r.rtf_median,
                "load_seconds": r.load_seconds,
                "metrics": r.metrics,
            }
            for r in results
        ],
    }


def write_results(summary: dict, results: list[EngineResult], out_dir: Path) -> tuple[Path, Path]:
    """Write the summary JSON and a per-file JSONL under ``out_dir``.

    The summary is the small, promotable digest; the JSONL keeps every per-file
    score for auditing and is git-ignored.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    kind = summary["kind"]
    summary_path = out_dir / f"{kind}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    jsonl_path = out_dir / f"{kind}.perfile.jsonl"
    with jsonl_path.open("w") as fh:
        for r in results:
            for rec in r.per_file:
                fh.write(json.dumps({"engine": r.alias, **rec}) + "\n")
    return summary_path, jsonl_path
