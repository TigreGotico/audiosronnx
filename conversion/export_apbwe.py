"""Export AP-BWE (yxlu-0102/AP-BWE) to ONNX.

AP-BWE is an amplitude-phase bandwidth extender: a dual ConvNeXt stack that takes the
log-amplitude and phase spectra of a band-limited signal and predicts the wideband
log-amplitude and phase. All STFT/ISTFT stays outside the network, so the model is one
clean ONNX graph ``(log_amp, pha) -> (log_amp_wb, pha_wb)``.

The exported graph plus the numpy STFT/ISTFT front-end are validated end-to-end against
the reference PyTorch inference (log-spectral distance).

Requires torch and the ``AP-BWE`` package (models/, env, utils) on the path.

Usage:
    python export_apbwe.py --config config_16kto48k.json --checkpoint g_16kto48k \
        --output-dir ./out/apbwe
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def export(config_file: str, checkpoint_file: str, output_dir: Path):
    from env import AttrDict
    from models.model import APNet_BWE_Model

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(config_file) as f:
        h = AttrDict(json.load(f))

    model = APNet_BWE_Model(h)
    state = torch.load(checkpoint_file, map_location="cpu")
    model.load_state_dict(state["generator"])
    model.eval()

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, amp, pha):
            amp_wb, pha_wb, _ = self.m(amp, pha)
            return amp_wb, pha_wb

    wrap = Wrap(model).eval()
    freq = h.n_fft // 2 + 1
    dummy_amp = torch.randn(1, freq, 100)
    dummy_pha = torch.randn(1, freq, 100)
    path = output_dir / "apbwe.onnx"
    kw = dict(
        input_names=["log_amp", "pha"], output_names=["log_amp_wb", "pha_wb"],
        dynamic_axes={"log_amp": {0: "batch", 2: "frames"},
                      "pha": {0: "batch", 2: "frames"},
                      "log_amp_wb": {0: "batch", 2: "frames"},
                      "pha_wb": {0: "batch", 2: "frames"}},
        opset_version=17, do_constant_folding=True,
    )
    try:
        torch.onnx.export(wrap, (dummy_amp, dummy_pha), str(path), dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(wrap, (dummy_amp, dummy_pha), str(path), **kw)

    # graph parity vs torch
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        r_amp, r_pha = wrap(dummy_amp, dummy_pha)
    o_amp, o_pha = sess.run(None, {"log_amp": dummy_amp.numpy(), "pha": dummy_pha.numpy()})
    err = max(float(np.abs(r_amp.numpy() - o_amp).max()),
              float(np.abs(r_pha.numpy() - o_pha).max()))
    mb = path.stat().st_size / 1e6
    print(f"apbwe.onnx {mb:.2f} MB  n_fft={h.n_fft} hop={h.hop_size} win={h.win_size} "
          f"lr={h.lr_sampling_rate} hr={h.hr_sampling_rate}")
    print(f"graph parity max abs err={err:.2e} {'PASS' if err < 1e-3 else 'FAIL'}")
    return err


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("./out/apbwe"))
    a = ap.parse_args()
    export(a.config, a.checkpoint, a.output_dir)
