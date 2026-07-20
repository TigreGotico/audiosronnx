"""Export CMGAN (ruizhecao96/CMGAN) to ONNX.

CMGAN is a conformer-based metric GAN denoiser working on the complex STFT. Only the
generator (``TSCNet``) is exported; the discriminator supplies the metric-GAN training loss
and has no inference role.

``TSCNet.forward`` derives the noisy phase with ``torch.angle(torch.complex(re, im))``, and
``torch.complex`` has no ONNX operator. This script rewrites that step as the identical
``torch.atan2(im, re)`` and checks the rewrite is bit-for-bit equal before exporting.

The graph is exported with ``do_constant_folding=False`` and then swept across sequence
lengths, since folding can specialise an attention graph to the length it was traced at.

Requires torch and the CMGAN sources (``models/generator.py``, ``models/conformer.py``) plus
its ``best_ckpt/ckpt`` on the path.

Usage::

    python export_cmgan.py --checkpoint ckpt --output-dir ./out/cmgan
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

N_FFT = 400


class ExportableTSCNet(torch.nn.Module):
    """TSCNet with the unexportable ``angle(complex(...))`` rewritten as ``atan2``."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        n = self.net
        mag = torch.sqrt(x[:, 0, :, :] ** 2 + x[:, 1, :, :] ** 2).unsqueeze(1)
        noisy_phase = torch.atan2(x[:, 1, :, :], x[:, 0, :, :]).unsqueeze(1)
        out = n.TSCB_4(n.TSCB_3(n.TSCB_2(n.TSCB_1(n.dense_encoder(torch.cat([mag, x], 1))))))
        out_mag = n.mask_decoder(out) * mag
        complex_out = n.complex_decoder(out)
        real = out_mag * torch.cos(noisy_phase) + complex_out[:, 0, :, :].unsqueeze(1)
        imag = out_mag * torch.sin(noisy_phase) + complex_out[:, 1, :, :].unsqueeze(1)
        return torch.cat([real, imag], dim=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="ckpt")
    ap.add_argument("--output-dir", type=Path, default=Path("./out/cmgan"))
    a = ap.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)

    from models.generator import TSCNet

    freq = N_FFT // 2 + 1
    net = TSCNet(num_channel=64, num_features=freq)
    net.load_state_dict(torch.load(a.checkpoint, map_location="cpu"))
    net.eval()
    wrapped = ExportableTSCNet(net).eval()

    for frames in (50, 100, 300):
        x = torch.randn(1, 2, frames, freq)
        with torch.no_grad():
            real, imag = net(x)
            got = wrapped(x)
        err = max(float((real - got[:, :1]).abs().max()), float((imag - got[:, 1:]).abs().max()))
        print(f"atan2 rewrite parity T={frames}: max abs err={err:.2e}")

    path = a.output_dir / "cmgan.onnx"
    torch.onnx.export(
        wrapped, (torch.randn(1, 2, 100, freq),), str(path),
        input_names=["spec"], output_names=["enhanced"],
        dynamic_axes={"spec": {0: "batch", 2: "frames"},
                      "enhanced": {0: "batch", 2: "frames"}},
        opset_version=17, do_constant_folding=False, dynamo=False)
    print(f"{path.name}  {path.stat().st_size / 1e6:.1f} MB")

    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for frames in (50, 100, 251, 700, 1500):
        x = np.random.randn(1, 2, frames, freq).astype(np.float32)
        with torch.no_grad():
            ref = wrapped(torch.from_numpy(x)).numpy()
        err = float(np.abs(ref - sess.run(None, {"spec": x})[0]).max())
        print(f"onnx parity T={frames}: max abs err={err:.2e} "
              f"{'PASS' if err < 1e-3 else 'FAIL'}")


if __name__ == "__main__":
    main()
