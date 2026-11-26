"""Render promoted summaries into the docs, between HTML comment markers.

``render.py`` reads only ``benchmarks/results/published/*.summary.json`` — the
files a human deliberately promoted — and rewrites the block between
``<!-- benchmark:{id}:start -->`` and ``<!-- benchmark:{id}:end -->`` in each
target document. Everything outside the markers, including the hand-written
decision guides, is left untouched.

It is **idempotent**: running it twice produces no diff. Tables are sorted
deterministically, floats are formatted to a fixed precision, and the block is
byte-for-byte reproducible from the summary JSON.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
PUBLISHED_DIR = BENCH_DIR / "results" / "published"
REPO = BENCH_DIR.parent

# Which published summary feeds which marker in which file.
TARGETS = [
    ("denoise", REPO / "docs" / "denoising.md", "benchmark:denoise"),
    ("enhance", REPO / "docs" / "denoising.md", "benchmark:enhance"),
    ("sr", REPO / "docs" / "engines.md", "benchmark:sr"),
]

# Secondary metric columns shown per track (in order), beyond the primary.
SECONDARY_COLUMNS = {
    "denoise": [("dnsmos.sig", "SIG"), ("si_sdr", "SI-SDR"), ("stoi", "STOI")],
    "enhance": [("sigmos.ovrl", "SIGMOS"), ("stoi", "STOI"), ("mcd", "MCD↓")],
    "sr": [("lsd", "LSD↓"), ("mel_l1", "mel-L1↓")],
}
PRIMARY_LABEL = {
    "denoise": "DNSMOS", "enhance": "DNSMOS", "sr": "SIGMOS",
}


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, (int, float)):
        return f"{v:.3f}"
    return str(v)


def _size(bytes_) -> str:
    if not bytes_:
        return "—"
    mb = bytes_ / 1e6
    return f"{mb:.1f} MB" if mb >= 1 else f"{bytes_ / 1e3:.0f} KB"


def _rtf(v) -> str:
    return f"{v:.3f}×" if isinstance(v, (int, float)) else "—"


def render_table(summary: dict) -> str:
    kind = summary["kind"]
    primary = summary["primary"]
    prov = summary["provenance"]
    secondaries = SECONDARY_COLUMNS.get(kind, [])

    scored = [e for e in summary["engines"] if e.get("metrics", {}).get(primary) is not None]
    missing = [e for e in summary["engines"] if e.get("metrics", {}).get(primary) is None]
    scored.sort(key=lambda e: e["metrics"][primary], reverse=True)

    header = ["Engine", PRIMARY_LABEL.get(kind, "Primary")]
    header += [label for _, label in secondaries]
    header += ["Spk-sim", "RTF", "Size", "License"]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]

    for i, e in enumerate(scored):
        m = e["metrics"]
        name = f"**{e['alias']}**" + (" — SOTA (measured)" if i == 0 else "")
        cells = [name, _fmt(m.get(primary))]
        cells += [_fmt(m.get(key)) for key, _ in secondaries]
        cells += [_fmt(m.get("speaker_similarity")), _rtf(e.get("rtf_median")),
                  _size(e.get("model_bytes")), e.get("license") or "—"]
        lines.append("| " + " | ".join(cells) + " |")

    for e in missing:
        note = e.get("status", "unavailable")
        lines.append(f"| {e['alias']} | _{note}_ |" + " |" * (len(header) - 2))

    prov_bits = [f"dataset `{prov.get('dataset')}`"]
    if prov.get("revision"):
        prov_bits.append(f"rev `{prov['revision'][:10]}`")
    if prov.get("input_rate"):
        prov_bits.append(f"{prov['input_rate']} Hz input")
    prov_bits += [f"seed {prov.get('seed')}", f"{prov.get('limit')} files",
                  prov.get("date", "")]
    provenance = ", ".join(b for b in prov_bits if b)

    body = [
        "",
        f"_**Provisional — full run pending.** Smoke subset only._ Scored with "
        f"`speechonnxmetrics`; {provenance}.",
        "",
        "\n".join(lines),
        "",
        "Higher is better except columns marked ↓. Non-reference MOS (DNSMOS, "
        "SIGMOS) scores the output alone; intrusive columns compare against the "
        "clean reference. `Spk-sim` is speaker-embedding cosine — a low value flags "
        "an engine that resynthesised a different-sounding voice.",
        "",
    ]
    return "\n".join(body)


def _replace_block(text: str, marker: str, content: str) -> str:
    start = f"<!-- {marker}:start -->"
    end = f"<!-- {marker}:end -->"
    block = f"{start}\n{content}\n{end}"
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if pattern.search(text):
        return pattern.sub(lambda _: block, text)
    raise KeyError(f"marker {marker!r} not found in target document")


def render_all() -> list[Path]:
    """Render every promoted summary into its target document. Returns changed files."""
    changed: list[Path] = []
    for kind, path, marker in TARGETS:
        summary_file = PUBLISHED_DIR / f"{kind}.summary.json"
        if not summary_file.is_file():
            continue
        if not path.is_file():
            continue
        summary = json.loads(summary_file.read_text())
        content = render_table(summary)
        text = path.read_text()
        new = _replace_block(text, marker, content)
        if new != text:
            path.write_text(new)
            changed.append(path)
    return changed


def main() -> int:
    changed = render_all()
    if changed:
        for p in changed:
            print(f"updated {p}")
    else:
        print("no changes (docs already match published summaries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
