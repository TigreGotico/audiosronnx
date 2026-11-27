"""Regression floor for the benchmark board — opt-in, not part of the default suite.

This test downloads engine weights and streams gold audio from HuggingFace, so it
is gated behind the ``benchmark`` marker and skipped by the default run
(``addopts = -m 'not benchmark'`` in pyproject). Run it deliberately::

    pytest -m benchmark test/test_benchmark_floor.py

It re-runs each track's default engine at the seed and limit recorded in
``benchmarks/floors.json`` and asserts the primary metric has not fallen below
``floor - epsilon``. A red result means a real regression: a model swap, a DSP
change, or a metrics-library change moved the number. Investigate before touching
the floor, and only ever raise it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FLOORS_PATH = Path(__file__).resolve().parent.parent / "benchmarks" / "floors.json"

pytestmark = pytest.mark.benchmark


def _floors() -> dict:
    return json.loads(FLOORS_PATH.read_text())


@pytest.mark.parametrize("track", ["denoise", "sr", "enhance"])
def test_default_engine_meets_floor(track: str) -> None:
    pytest.importorskip("speechonnxmetrics")
    pytest.importorskip("datasets")
    from benchmarks import harness

    cfg = _floors()
    spec = cfg["tracks"][track]
    eps = cfg["epsilon"]

    summary, _results = harness.run_board(
        track,
        engines=[spec["engine"]],
        limit=cfg["limit"],
        seed=cfg["seed"],
        input_rate=cfg["input_rate"],
        skip_nc=False,
    )
    row = next(e for e in summary["engines"] if e["alias"] == spec["engine"])
    assert row["status"] == "ok", f"{spec['engine']} unavailable: {row['note']}"
    value = row["metrics"].get(spec["primary"])
    assert value is not None, f"{spec['primary']} not scored for {spec['engine']}"
    assert value >= spec["floor"] - eps, (
        f"{track} default {spec['engine']} {spec['primary']}={value:.3f} fell below "
        f"floor {spec['floor']} - {eps}"
    )
