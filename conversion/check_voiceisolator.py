#!/usr/bin/env python3
"""Check an exported 555 M voice isolator against upstream's PyTorch streaming step.

    python conversion/check_voiceisolator.py out/release meeting.wav
    python conversion/check_voiceisolator.py out/agent meeting.wav --sub agent

Runs the same audio through the audiosronnx runtime on the exported graphs and through
upstream ``FrameFormer.stream_step`` in float32 on the CPU, in steps of the export's chunk,
and prints the largest sample difference. Use at least 14 s of 16 kHz speech, so the 10 s
attention rings wrap. Cells: an enrollment clip (the recording's last 3 s) and the null
vector; for a model with a presence head, the served gate as well.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from export_voiceisolator import load_upstream  # noqa: E402

from audiosronnx import load_denoise  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("wav", help="16 kHz mono speech, 14 s or longer")
    ap.add_argument("--sub", default="")
    ap.add_argument("--seconds", type=float, default=14.0)
    ap.add_argument("--repo", default="Scicom-intl/voice-isolator")
    ap.add_argument("--revision", default="d41d6d8166a6566c9c2df7ca17e9ca38abcd35d6")
    args = ap.parse_args()

    m, up_cfg, _ = load_upstream(args.repo, args.sub, args.revision)
    m.float().eval()
    from vi.model import parse_gate

    C = json.loads((Path(args.model_dir) / "config.json").read_text())["chunk_frames"]
    x, sr = sf.read(args.wav, dtype="float32")
    assert sr == 16000 and x.ndim == 1, "16 kHz mono"
    xs = x[: int(args.seconds * sr) // (C * 160) * (C * 160)]
    clip = x[-3 * sr:]
    gates = ["off"] + (["auto"] if m.gate_head is not None else [])
    for enrolled in (True, False):
        for gate in gates:
            m.gate = parse_gate(up_cfg["serve"]["gate"]) if gate == "auto" else None
            st = m.stream_init(1, "cpu", torch.float32, max_chunk=C)
            with torch.no_grad():
                emb = m.embed(torch.from_numpy(clip)[None], torch.tensor([clip.size])) if enrolled else m.null[None]
                ref = np.concatenate([m.stream_step(torch.from_numpy(xs[i:i + C * 160])[None], st, emb)[0].numpy()
                                      for i in range(0, xs.size, C * 160)])
            kw = dict(enrollment=clip) if enrolled else dict(live_enroll=False)
            ours = load_denoise("voiceisolator", model_dir=args.model_dir, gate=gate, **kw).stream().push(xs)
            err = np.abs(ours - ref)
            print(f"{'clip' if enrolled else 'null'} gate={gate}: {xs.size / sr:.1f} s, "
                  f"max abs err {err.max():.2e} (after 10.5 s: {err[int(10.5 * sr):].max():.2e}), "
                  f"silent samples ours {np.mean(ours == 0):.3f} upstream {np.mean(ref == 0):.3f}")


if __name__ == "__main__":
    main()
