"""Export LavaSR to ONNX (maintainer tool).

Splits the upstream PyTorch LavaSR checkpoint (``YatharthS/LavaSR``, Apache-2.0) into
three ONNX graphs with weights embedded (no ``.data`` sidecars):

  * ``backbone.onnx``      — VocosBackbone (mel -> hidden)
  * ``spec_head.onnx``     — ISTFTHead projection -> (real, imag) spectrogram
  * ``denoiser_core.onnx`` — UL-UNAS denoiser core (spec RI -> enhanced spec RI)

All STFT / ISTFT / mel stay in numpy at runtime, so no ``torch.stft`` enters any
graph. Each graph is validated against its PyTorch submodule (max abs error).

Requires torch + vocos + the ``LavaSR`` package on the path.

Usage:
    python export_lavasr.py --output-dir ./out
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class BackboneExport(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, mel):  # mel [B,80,T] -> hidden [B,T,512]
        return self.backbone(mel)


class SpecHeadExport(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.out = head.out  # Linear(512, 2050)

    def forward(self, hidden):  # [B,T,512] -> real/imag [B,1025,T]
        x = self.out(hidden).transpose(1, 2)
        mag, phase = x.chunk(2, dim=1)
        mag = torch.clamp(torch.exp(mag), max=1e2)
        return mag * torch.cos(phase), mag * torch.sin(phase)


class DenoiserCoreExport(nn.Module):
    def __init__(self, ulunas):
        super().__init__()
        self.erb = ulunas.erb
        self.encoder = ulunas.encoder
        self.dpgrnn = ulunas.dpgrnn
        self.decoder = ulunas.decoder

    def forward(self, spec):  # [B,2,T,F] -> [B,2,T,F]
        feat = torch.log10(torch.norm(spec, dim=1, keepdim=True).clamp(1e-12))
        feat = self.erb.bm(feat)
        feat, en_outs = self.encoder(feat)
        feat = self.dpgrnn(feat)
        m_feat = self.decoder(feat, en_outs)
        return spec * self.erb.bs(m_feat)


def _export(module, dummy, path, in_names, out_names, dyn):
    module.eval()
    kw = dict(input_names=in_names, output_names=out_names, dynamic_axes=dyn,
              opset_version=17, do_constant_folding=True)
    try:
        torch.onnx.export(module, dummy, str(path), dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(module, dummy, str(path), **kw)


def _parity(path, module, dummy, in_name):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = module(dummy)
    outs = sess.run(None, {in_name: dummy.numpy()})
    refs = ref if isinstance(ref, tuple) else (ref,)
    return max(float(np.max(np.abs(r.numpy() - o))) for r, o in zip(refs, outs))


def export_all(output_dir: Path, model_path: str | None = None):
    from LavaSR.model import LavaEnhance2

    if model_path is None:
        from huggingface_hub import snapshot_download

        model_path = snapshot_download("YatharthS/LavaSR")

    output_dir.mkdir(parents=True, exist_ok=True)
    model = LavaEnhance2(model_path=model_path, device="cpu")
    vocos = model.bwe_model.bwe_model
    ulunas = model.denoiser_model.model

    report = {}

    bb = BackboneExport(vocos.backbone).eval()
    bb_dummy = torch.randn(1, 80, 200)
    _export(bb, bb_dummy, output_dir / "backbone.onnx", ["mel"], ["hidden"],
            {"mel": {0: "batch", 2: "frames"}, "hidden": {0: "batch", 1: "frames"}})
    report["backbone"] = _parity(output_dir / "backbone.onnx", bb, bb_dummy, "mel")

    head = SpecHeadExport(vocos.head).eval()
    with torch.no_grad():
        hidden = bb(bb_dummy)
    _export(head, hidden, output_dir / "spec_head.onnx", ["hidden"], ["real", "imag"],
            {"hidden": {0: "batch", 1: "frames"}, "real": {0: "batch", 2: "frames"},
             "imag": {0: "batch", 2: "frames"}})
    report["spec_head"] = _parity(output_dir / "spec_head.onnx", head, hidden, "hidden")

    dn = DenoiserCoreExport(ulunas).eval()
    dn_dummy = torch.randn(1, 2, 63, 257)
    _export(dn, dn_dummy, output_dir / "denoiser_core.onnx", ["spec_ri"],
            ["enhanced_ri"], {})
    report["denoiser_core"] = _parity(output_dir / "denoiser_core.onnx", dn, dn_dummy,
                                      "spec_ri")

    print("\nParity (max abs error, ONNX vs PyTorch submodule):")
    for name, err in report.items():
        path = output_dir / f"{name}.onnx"
        mb = path.stat().st_size / 1e6
        print(f"  {name:14s} {mb:6.2f} MB  err={err:.2e}  {'PASS' if err < 1e-3 else 'FAIL'}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("./out/lavasr"))
    ap.add_argument("--model-path", default=None, help="local staged model dir")
    args = ap.parse_args()
    export_all(args.output_dir, args.model_path)
