"""Export MossFormerGAN_SE_16K (ClearerVoice-Studio) to ONNX.

MossFormer attention with a metric-GAN objective — the highest published PESQ of any model
surveyed for this library. Only the generator is exported; the discriminator supplies the
training loss.

Three obstacles, handled here:

* ``torch.complex`` has no ONNX operator. ``angle(complex(re, im))`` is ``atan2(im, re)``;
  patch ``models/mossformer_gan_se/generator.py`` accordingly.
* ``torch.eye(dtype=bool)`` exports to ``EyeLike(bool)``, for which onnxruntime has no
  kernel. An arange equality builds the same identity mask; patch
  ``models/mossformer_gan_se/mossformer.py``.
* **Fixed window.** MossFormer's group attention reshapes the sequence into fixed groups and
  that reshape captures the traced length, so a dynamic-length graph runs only at its trace
  size — even with constant folding disabled. Export at a fixed frame count and let the
  adapter slide that window. Tracing at 1601 frames (upstream's 10 s decode window)
  exhausts memory; 401 frames (~2.5 s) exports comfortably.

Usage::

    python export_mossformergan.py --frames 401 --output-dir ./out/mossformergan
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

FREQ_BINS = 201


def _patch_sources() -> None:
    """Rewrite the two unexportable operators in the installed clearvoice sources."""
    import clearvoice
    import os

    root = os.path.dirname(clearvoice.__file__)

    gen = os.path.join(root, "models/mossformer_gan_se/generator.py")
    text = open(gen).read()
    old = "noisy_phase = torch.angle(torch.complex(x[:, 0, :, :], x[:, 1, :, :])).unsqueeze(1)"
    if old in text:
        new = "noisy_phase = torch.atan2(x[:, 1, :, :], x[:, 0, :, :]).unsqueeze(1)"
        open(gen, "w").write(text.replace(old, new))
        print("patched generator.py: angle(complex(...)) -> atan2")

    mf = os.path.join(root, "models/mossformer_gan_se/mossformer.py")
    text = open(mf).read()
    old = "        mask_c = torch.eye(quad_q_c.shape[-2], dtype = torch.bool, device = device)"
    if old in text:
        new = ("        _n_c = quad_q_c.shape[-2]\n"
               "        _idx_c = torch.arange(_n_c, device = device)\n"
               "        mask_c = _idx_c[:, None] == _idx_c[None, :]")
        open(mf, "w").write(text.replace(old, new))
        print("patched mossformer.py: eye(bool) -> arange equality")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=401)
    ap.add_argument("--output-dir", type=Path, default=Path("./out/mossformergan"))
    a = ap.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)

    _patch_sources()
    original = torch.load
    torch.load = lambda *args, **kw: original(*args, **{**kw, "map_location": "cpu"})
    torch.nn.Module.cuda = lambda self, *args, **kw: self
    torch.Tensor.cuda = lambda self, *args, **kw: self

    from clearvoice import ClearVoice

    cv = ClearVoice(task="speech_enhancement", model_names=["MossFormerGAN_SE_16K"])
    net = cv.models[0].model
    while hasattr(net, "model") and isinstance(net.model, torch.nn.Module):
        net = net.model
    net.eval()

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, spec):
            real, imag = self.m(spec)
            return torch.cat([real, imag], dim=1)

    wrapped = Wrap(net).eval()
    dummy = torch.randn(1, 2, a.frames, FREQ_BINS)
    path = a.output_dir / "mossformergan.onnx"
    torch.onnx.export(
        wrapped, (dummy,), str(path),
        input_names=["spec"], output_names=["enhanced"],
        dynamic_axes={"spec": {0: "batch"}, "enhanced": {0: "batch"}},   # batch only
        opset_version=17, do_constant_folding=False, dynamo=False)
    print(f"{path.name}  {path.stat().st_size / 1e6:.1f} MB  (fixed {a.frames} frames)")

    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for trial in range(3):
        x = np.random.randn(1, 2, a.frames, FREQ_BINS).astype(np.float32)
        with torch.no_grad():
            ref = wrapped(torch.from_numpy(x)).numpy()
        err = float(np.abs(ref - sess.run(None, {"spec": x})[0]).max())
        print(f"  parity trial {trial}: max abs err={err:.2e} "
              f"{'PASS' if err < 1e-3 else 'FAIL'}")


if __name__ == "__main__":
    main()
