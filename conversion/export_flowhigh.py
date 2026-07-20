"""Export FLowHigh (jjunak-yun/FLowHigh_code) and its BigVGAN vocoder to ONNX.

FLowHigh upscales speech to 48 kHz with conditional flow matching. Two graphs carry the
neural work; the log-mel front-end, the Euler loop and the spectral merge stay in numpy::

    flow field : x[1,T,256], times[1], cond[1,T,256] -> velocity[1,T,256]
    vocoder    : mel[1,256,T]                        -> waveform[1,1,480*T]

Two tracing obstacles are rewritten first, each verified bit-for-bit equivalent:

* **BigVGAN's alias-free resamplers** build their depthwise kernel at call time with
  ``filter.expand(C, -1, -1)``, so the tracer sees a convolution of unknown kernel shape.
  The channel count is fixed per layer, so ``_bigvgan_static`` records it once and
  materialises each expanded kernel as a buffer.
* **Rotary positions** come from ``seq_len``, which under tracing arrives as a 0-dim tensor
  and skips ``RotaryEmbedding``'s ``arange`` branch. Patch ``modules.py`` to build
  positions with an explicit ``torch.arange(x.shape[-2])``.

Both graphs are exported with ``do_constant_folding=False`` and swept across sequence
lengths afterwards, since folding can specialise an attention graph to its traced length.

Requires torch plus the FLowHigh sources, its checkpoint, and the BigVGAN 48 kHz
checkpoint. Upstream calls ``.cuda()`` unconditionally and loads without ``map_location``,
so both are neutralised for a CPU export.

Usage::

    python export_flowhigh.py --checkpoint FLowHigh_indep_adaptive_400k.pt \
        --vocoder g_48_00850000 --vocoder-config bigvgan_48khz_256band_config.json \
        --output-dir ./out/flowhigh
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

N_MELS = 256


def _neutralise_cuda() -> None:
    original = torch.load
    torch.load = lambda *a, **k: original(*a, **{**k, "map_location": "cpu"})
    torch.nn.Module.cuda = lambda self, *a, **k: self
    torch.Tensor.cuda = lambda self, *a, **k: self


def _sweep(session, torch_fn, feeds_fn, lengths, label) -> None:
    for n in lengths:
        feeds = feeds_fn(n)
        with torch.no_grad():
            ref = torch_fn(feeds).numpy()
        got = session.run(None, {k: v for k, v in feeds.items()})[0]
        err = float(np.abs(ref - got).max())
        print(f"{label} T={n}: max abs err={err:.2e} {'PASS' if err < 1e-3 else 'FAIL'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--vocoder", required=True)
    ap.add_argument("--vocoder-config", required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("./out/flowhigh"))
    a = ap.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    _neutralise_cuda()

    from _bigvgan_static import make_static
    from bigvgan.env import AttrDict
    from bigvgan.models import BigVGAN
    from cfm_superresolution import ConditionalFlowMatcherWrapper, FLowHigh, MelVoco

    # ---- vocoder -------------------------------------------------------- #
    cfg = AttrDict(json.load(open(a.vocoder_config)))
    voc = BigVGAN(cfg)
    voc.load_state_dict(torch.load(a.vocoder)["generator"])
    voc.remove_weight_norm()
    voc.eval()
    reference = copy.deepcopy(voc).eval()
    replaced = make_static(voc, torch.randn(1, cfg.num_mels, 50))
    print(f"materialised {replaced} alias-free resampler kernels")
    for n in (25, 50, 137):
        mel = torch.randn(1, cfg.num_mels, n)
        with torch.no_grad():
            err = float((reference(mel) - voc(mel)).abs().max())
        print(f"static-kernel parity T={n}: max abs err={err:.2e}")

    voc_path = a.output_dir / "bigvgan.onnx"
    torch.onnx.export(
        voc, (torch.randn(1, cfg.num_mels, 50),), str(voc_path),
        input_names=["mel"], output_names=["wav"],
        dynamic_axes={"mel": {0: "batch", 2: "frames"}, "wav": {0: "batch", 2: "samples"}},
        opset_version=17, do_constant_folding=False, dynamo=False)
    print(f"{voc_path.name}  {voc_path.stat().st_size / 1e6:.1f} MB")

    # ---- flow field ----------------------------------------------------- #
    enc = MelVoco(n_mels=N_MELS, sampling_rate=48000, f_max=24000, n_fft=2048,
                  win_length=2048, hop_length=480, vocoder="bigvgan",
                  vocoder_config=a.vocoder_config, vocoder_path=a.vocoder)
    gen = FLowHigh(dim_in=N_MELS, audio_enc_dec=enc, depth=2, dim_head=64, heads=16,
                   architecture="transformer")
    wrapper = ConditionalFlowMatcherWrapper(
        flowhigh=gen, cfm_method="independent_cfm_adaptive",
        torchdiffeq_ode_method="euler", sigma=1e-4)
    wrapper.load_state_dict(torch.load(a.checkpoint)["model"])
    gen.eval()

    class FlowField(torch.nn.Module):
        """One ODE right-hand-side evaluation."""

        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x, times, cond):
            return self.net.forward(x, times=times, cond=cond, cond_drop_prob=0.0)

    field = FlowField(gen).eval()
    flow_path = a.output_dir / "flowfield.onnx"
    example = (torch.randn(1, 60, N_MELS), torch.tensor([0.25]), torch.randn(1, 60, N_MELS))
    torch.onnx.export(
        field, example, str(flow_path),
        input_names=["x", "times", "cond"], output_names=["velocity"],
        dynamic_axes={"x": {0: "batch", 1: "frames"}, "cond": {0: "batch", 1: "frames"},
                      "velocity": {0: "batch", 1: "frames"}},
        opset_version=17, do_constant_folding=False, dynamo=False)
    print(f"{flow_path.name}  {flow_path.stat().st_size / 1e6:.1f} MB")

    import onnxruntime as ort

    _sweep(ort.InferenceSession(str(voc_path), providers=["CPUExecutionProvider"]),
           lambda f: voc(torch.from_numpy(f["mel"])),
           lambda n: {"mel": np.random.randn(1, cfg.num_mels, n).astype(np.float32)},
           (25, 50, 137, 300), "vocoder")
    _sweep(ort.InferenceSession(str(flow_path), providers=["CPUExecutionProvider"]),
           lambda f: field(torch.from_numpy(f["x"]), torch.from_numpy(f["times"]),
                           torch.from_numpy(f["cond"])),
           lambda n: {"x": np.random.randn(1, n, N_MELS).astype(np.float32),
                      "times": np.array([0.7], dtype=np.float32),
                      "cond": np.random.randn(1, n, N_MELS).astype(np.float32)},
           (30, 60, 150, 400), "flow field")


if __name__ == "__main__":
    main()
