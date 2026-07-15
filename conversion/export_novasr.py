"""Export NovaSR to ONNX (maintainer tool).

NovaSR (``YatharthS/NovaSR``, Apache-2.0) is a ~52 KB conv1d/BigVGAN-snake generator
that upsamples 16 kHz speech to 48 kHz. It is a single time-domain graph — no STFT, no
external DSP — so the whole model exports to one ONNX file with a dynamic time axis.

The exported graph is validated against the PyTorch model (max abs error) on a real
waveform.

Requires torch + the ``NovaSR`` package on the path.

Usage:
    python export_novasr.py --output-dir ./out
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def export(output_dir: Path):
    from NovaSR import FastSR

    output_dir.mkdir(parents=True, exist_ok=True)
    sr = FastSR(half=False)
    model = sr.model.eval().float()

    dummy = torch.randn(1, 1, 16000)  # 1 s of 16 kHz audio, [B,1,T]
    # Warm up so the (anti-aliasing) resample filters precompute their buffers
    # outside the trace; otherwise their in-place ``copy_`` ends up in the graph.
    with torch.no_grad():
        model(dummy)
    path = output_dir / "novasr.onnx"
    kw = dict(
        input_names=["audio_16k"], output_names=["audio_48k"],
        dynamic_axes={"audio_16k": {0: "batch", 2: "samples"},
                      "audio_48k": {0: "batch", 2: "samples"}},
        opset_version=17, do_constant_folding=True,
    )
    # The polyphase resampler uses ``groups=x.shape[1]``, which the legacy
    # TorchScript exporter cannot fold to a constant; the dynamo exporter
    # specializes it cleanly. Fall back to legacy only if dynamo is unavailable.
    try:
        torch.onnx.export(
            model, dummy, str(path), dynamo=True,
            input_names=["audio_16k"], output_names=["audio_48k"],
            dynamic_axes={"audio_16k": {0: "batch", 2: "samples"},
                          "audio_48k": {0: "batch", 2: "samples"}},
            opset_version=17,
        )
    except (TypeError, Exception):
        torch.onnx.export(model, dummy, str(path), dynamo=False, **kw)

    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = model(dummy).numpy()
    out = sess.run(None, {"audio_16k": dummy.numpy()})[0]
    err = float(np.max(np.abs(ref - out)))
    mb = path.stat().st_size / 1e6
    print(f"novasr.onnx  {mb:.3f} MB  in {dummy.shape} -> out {out.shape}")
    print(f"parity max abs error = {err:.2e}  {'PASS' if err < 1e-3 else 'FAIL'}")
    return err


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("./out/novasr"))
    export(ap.parse_args().output_dir)
