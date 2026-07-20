"""Export the Facebook/Meta Research denoiser (dns48 / dns64) to ONNX.

A causal Demucs working on the waveform: convolutional encoder/decoder around an LSTM
bottleneck, with no spectral front-end. Amplitude normalisation and the internal length
padding live inside ``forward``, so the exported graph is self-contained.

**Fixed window.** ``Demucs.valid_length`` computes its padding from the input length with
Python arithmetic, which the tracer bakes in. A dynamic-length export is correct only at
the length it was traced at — measured max abs err 3.6e-07 at the traced length and ~1.0
elsewhere, with the output shape frozen. These graphs are therefore exported at a fixed
10 s window and the adapter slides that window with a crossfaded overlap.

The weights are **CC-BY-NC-4.0**.

Usage::

    python export_metadenoiser.py --output-dir ./out/metadenoiser
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

WINDOW = 160_000        # 10 s at 16 kHz


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("./out/metadenoiser"))
    ap.add_argument("--window", type=int, default=WINDOW)
    a = ap.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)

    from denoiser.pretrained import dns48, dns64

    for name, loader in (("dns64", dns64), ("dns48", dns48)):
        model = loader().eval()
        params = sum(p.numel() for p in model.parameters()) / 1e6
        dummy = torch.randn(1, 1, a.window)
        path = a.output_dir / f"{name}.onnx"
        torch.onnx.export(
            model, (dummy,), str(path),
            input_names=["noisy"], output_names=["enhanced"],
            dynamic_axes={"noisy": {0: "batch"}, "enhanced": {0: "batch"}},
            opset_version=17, do_constant_folding=False, dynamo=False)
        print(f"{name}: {params:.1f}M params, {path.stat().st_size / 1e6:.1f} MB")

        import onnxruntime as ort

        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        for trial in range(3):
            x = np.random.randn(1, 1, a.window).astype(np.float32)
            with torch.no_grad():
                ref = model(torch.from_numpy(x)).numpy()
            err = float(np.abs(ref - sess.run(None, {"noisy": x})[0]).max())
            print(f"  parity trial {trial}: max abs err={err:.2e} "
                  f"{'PASS' if err < 1e-3 else 'FAIL'}")


if __name__ == "__main__":
    main()
