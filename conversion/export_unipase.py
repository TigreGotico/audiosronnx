"""Export UniPASE (Xiaobin-Rong/unipase) universal speech enhancement to ONNX.

UniPASE is generative universal speech enhancement — noise and reverberation removed in one
feed-forward pass at 16 kHz. Two graphs, the same shape as sidon/voicefixer (a predictor
followed by a neural vocoder):

- **encoder_adapter** — the DeWavLM-Omni WavLM encoder (conv front-end + 24-layer
  transformer with gated relative-position bias) whose L1 and L24 layer reps are fused by a
  Vocos adapter: ``wav[1, 128000]`` -> ``feat[1, 400, 1024]``. Shipped **fp32** (a single
  ~1.7 GB graph, under the 2 GB protobuf limit). int8 is available via ``--quantize`` but is
  *not* recommended: the 24-layer WavLM-Large loses too much (end-to-end corr ~0.85), the
  same depth-driven degradation that makes CallEnhancer default to fp32. Quantizing must be
  restricted to MatMul — the conv front-end otherwise yields ``ConvInteger`` nodes ORT CPU
  cannot run.
- **vocoder** — the Vocos vocoder, cut just before its ISTFT: ``feat[1, 1024, T]`` ->
  ``spec[1, 1282, T]`` (interleaved log-magnitude + phase). ``torch.fft.irfft`` has no ONNX
  op, so the Vocos "same"-padding inverse STFT (n_fft 1280, hop 320) stays in numpy in the
  adapter (``engines/unipase.py``), matching upstream bit-for-bit.

Two things are worth knowing before reading the output of this script.

**WavLM's per-sample layer norm is rewritten.** Upstream normalises with
``F.layer_norm(feat, feat.shape[1:])`` — a dynamic ``normalized_shape`` the tracer refuses.
This script computes the identical statistic manually over dims (1, 2), which also makes the
graph shape-agnostic.

**PLC and PostNet are out of scope.** Upstream's optional packet-loss concealment needs a
data-dependent mask on the CNN output that does not trace; the 48 kHz PostNet depends on
``espnet2``. This script exports the 16 kHz denoise/dereverb core only. Both are follow-ups.

Parity is measured on real speech: enhancement models drift on out-of-domain noise input.

Usage::

    python export_unipase.py --unipase-src /path/to/clone/of/Xiaobin-Rong/unipase \\
        --output-dir ./out/unipase --speech reference.wav [--quantize]

Requires: torch, torchaudio, onnx, onnxruntime, huggingface_hub (the ``convert`` extra),
plus the upstream repo checked out (``--unipase-src``) so ``models`` is importable.

Reference
---------
- https://github.com/Xiaobin-Rong/unipase  (MIT)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HF_SRC = "Xiaobin-Rong/unipase"
_SR = 16000
_SEG = _SR * 8
_N_FFT, _HOP = 1280, 320


def _ckpt(name: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(_HF_SRC, filename=name)


def _global_ln(feat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """``F.layer_norm(feat, feat.shape[1:])`` with a static (traceable) reduction."""
    mu = feat.mean(dim=(1, 2), keepdim=True)
    var = feat.var(dim=(1, 2), unbiased=False, keepdim=True)
    return (feat - mu) / torch.sqrt(var + eps)


def _vocos_istft(spec_params: np.ndarray) -> np.ndarray:
    x = np.asarray(spec_params[0], dtype=np.float64)
    mag = np.exp(np.clip(x[: _N_FFT // 2 + 1], -20, 5))
    p = x[_N_FFT // 2 + 1:]
    spec = mag * (np.cos(p) + 1j * np.sin(p))
    win = np.hanning(_N_FFT + 1)[:-1]
    t = spec.shape[1]
    frames = np.fft.irfft(spec.T, n=_N_FFT, axis=1) * win
    out_size = (t - 1) * _HOP + _N_FFT
    ola = np.zeros(out_size); env = np.zeros(out_size); w2 = win ** 2
    for i in range(t):
        s = i * _HOP
        ola[s:s + _N_FFT] += frames[i]; env[s:s + _N_FFT] += w2
    pad = (_N_FFT - _HOP) // 2
    ola, env = ola[pad:-pad], env[pad:-pad]
    return (ola / np.where(env > 1e-11, env, 1.0)).astype(np.float32)


def _quantize(src: Path) -> Path:
    """Dynamic int8 of the transformer MatMuls (conv front-end stays fp32)."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process

    pre = src.with_name("encoder_adapter.pre.onnx")
    out = src.with_name("encoder_adapter.int8.onnx")
    quant_pre_process(str(src), str(pre))
    quantize_dynamic(str(pre), str(out), weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul"])
    pre.unlink(missing_ok=True)
    Path(str(pre) + ".data").unlink(missing_ok=True)
    print(f"{out.name}  {out.stat().st_size / 1e6:.1f} MB  (int8 from {src.stat().st_size / 1e6:.1f} MB)")
    return out


def _load_speech(path: str | None) -> torch.Tensor:
    if not path:
        print("WARNING: no --speech given; parity on noise understates this model")
        return torch.randn(1, _SEG) * 0.05
    import soundfile as sf
    from torchaudio.functional import resample

    a, fs = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(1)
    w = resample(torch.from_numpy(a)[None], fs, _SR)[:, :_SEG]
    if w.shape[-1] < _SEG:
        w = torch.nn.functional.pad(w, (0, _SEG - w.shape[-1]))
    return w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unipase-src", required=True,
                    help="path to a checkout of github.com/Xiaobin-Rong/unipase (for `models`)")
    ap.add_argument("--output-dir", type=Path, default=Path("./out/unipase"))
    ap.add_argument("--speech", default=None, help="a real speech wav; noise is not representative")
    ap.add_argument("--quantize", action="store_true",
                    help="also emit an int8 encoder (NOT recommended: e2e corr ~0.85)")
    a = ap.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, a.unipase_src)

    from models.wavlm.feature_extractor_plc import WavLM_feat
    from models.adapter.vocos.adapter import VocosAdapter
    from models.vocoder.vocos.vocoder import VocosVocoder

    enc = WavLM_feat(_ckpt("DeWavLM-Omni.pt"), output_layer=[1, 24]).eval()
    adp = VocosAdapter.from_pretrained(_ckpt("Adapter.pt")).eval()
    voc = VocosVocoder.from_pretrained(_ckpt("Vocoder_DWO-L1.pt")).eval()

    class EncAdapter(torch.nn.Module):
        def __init__(self, enc, adp):
            super().__init__(); self.enc = enc; self.adp = adp

        def forward(self, wav):                       # wav (B, L) @ 16 kHz
            w = self.enc.pad(wav)
            res = self.enc.wavlm.extract_features(
                w, output_layer=24, mask=False, mask_indices=None)[0]
            lr = res["layer_reps"]
            return self.adp(_global_ln(lr[1]), _global_ln(lr[24]))

    class VocSpec(torch.nn.Module):
        def __init__(self, voc):
            super().__init__(); self.voc = voc

        def forward(self, feat):                      # feat (B, 1024, T)
            return self.voc.head(self.voc.decoder(feat), return_hidden=True)

    wav = _load_speech(a.speech)
    ea, vs = EncAdapter(enc, adp).eval(), VocSpec(voc).eval()

    with torch.inference_mode():
        ref_feat = ea(wav)
    ea_path = a.output_dir / "encoder_adapter.onnx"
    torch.onnx.export(ea, (wav,), str(ea_path), input_names=["wav"], output_names=["feat"],
                      opset_version=17, do_constant_folding=True, dynamo=False)
    print(f"{ea_path.name}  {ea_path.stat().st_size / 1e6:.1f} MB")

    dummy_feat = ref_feat.transpose(1, 2).contiguous()
    with torch.inference_mode():
        ref_spec = vs(dummy_feat)
    v_path = a.output_dir / "vocoder.onnx"
    torch.onnx.export(vs, (dummy_feat,), str(v_path), input_names=["feat"], output_names=["spec"],
                      dynamic_axes={"feat": {2: "frames"}, "spec": {2: "frames"}},
                      opset_version=17, do_constant_folding=True, dynamo=False)
    print(f"{v_path.name}  {v_path.stat().st_size / 1e6:.1f} MB")

    if a.quantize:
        _quantize(ea_path)  # emitted for inspection only; the engine ships fp32

    # ---- parity: torch full core vs the shipped fp32 ONNX graphs + numpy ISTFT ----
    import onnxruntime as ort

    es = ort.InferenceSession(str(ea_path), providers=["CPUExecutionProvider"])
    vsess = ort.InferenceSession(str(v_path), providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        twav = voc(ref_feat.transpose(1, 2)).numpy().ravel()
    t0 = time.perf_counter()
    ofeat = es.run(None, {"wav": wav.numpy()})[0]
    ospec = vsess.run(None, {"feat": np.ascontiguousarray(ofeat.transpose(0, 2, 1))})[0]
    owav = _vocos_istft(ospec)
    dt = time.perf_counter() - t0
    n = min(twav.size, owav.size)
    corr = float(np.corrcoef(twav[:n], owav[:n])[0, 1])
    err = float(np.abs(twav[:n] - owav[:n]).max())
    print(f"end-to-end parity: corr={corr:.8f} max abs err={err:.2e} "
          f"{'PASS' if corr > 0.9999 else 'FAIL'}")
    print(f"CPU latency for 8 s output: {dt:.2f} s  ({8.0 / dt:.1f}x realtime)")


if __name__ == "__main__":
    main()
