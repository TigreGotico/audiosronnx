"""Export Sidon (sarulab-speech/Sidon) speech restoration to ONNX.

Sidon restores degraded 16 kHz speech to clean 48 kHz. It is two neural stages:

- a **feature extractor** — an 8-layer w2v-BERT 2.0 encoder (``facebook/w2v-bert-2.0``,
  ``num_hidden_layers=8``) LoRA-adapted to denoise SSL representations, mapping the
  SeamlessM4T log-mel features ``[1, T, 160]`` to hidden states ``[1, T, 1024]`` at
  50 Hz.
- a **decoder** — a DAC vocoder (rates ``[8, 5, 4, 3, 2]`` -> 960x upsample) turning
  ``[1, 1024, T]`` hidden states into a ``[1, 1, 960*T]`` 48 kHz waveform.

Upstream publishes both stages as TorchScript (``sarulab-speech/sidon-v0.1``:
``feature_extractor_cpu.pt`` / ``decoder_cpu.pt``). This script traces those scripted
modules to ONNX (opset 17), then dynamic-quantizes the feature extractor to int8 so the
transformer runs at a sane size/speed on CPU; the tiny decoder stays fp32.

The SeamlessM4T mel front-end (``spectrogram`` + per-mel-bin CMVN + stride-2 stacking)
is *not* exported — it stays in numpy in the adapter (``engines/sidon.py``), matching the
repo's "no torch at runtime, STFT/mel in numpy" rule. This script also parity-checks that
numpy front-end against ``transformers.SeamlessM4TFeatureExtractor``.

ONNX artifact: ``TigreGotico/audiosronnx-sidon`` (MIT).

Usage::

    python export_sidon.py --output-dir ./out/sidon           # pulls .pt from HF
    python export_sidon.py --output-dir ./out/sidon --no-quantize

Requires: torch, transformers, onnx, onnxruntime, huggingface_hub (the ``convert`` extra).

Reference
---------
- https://github.com/sarulab-speech/Sidon  (MIT)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

_HF_SRC = "sarulab-speech/sidon-v0.1"
_FE_PT = "feature_extractor_cpu.pt"
_DEC_PT = "decoder_cpu.pt"


def _load_jit(filename: str) -> torch.jit.ScriptModule:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(_HF_SRC, filename=filename)
    return torch.jit.load(path, map_location="cpu").eval()


def _export_feature_extractor(output_dir: Path) -> tuple[Path, float]:
    # The scripted module returns a dict (prim::DictConstruct, unexportable). Wrap it in a
    # *scripted* module that unpacks last_hidden_state so ONNX sees one tensor output.
    inner = _load_jit(_FE_PT)

    class _Unpack(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_features):
            return self.m(input_features)["last_hidden_state"]

    fe = torch.jit.script(_Unpack(inner)).eval()
    dummy = torch.randn(1, 100, 160)
    path = output_dir / "feature_extractor.onnx"
    kw = dict(
        input_names=["input_features"], output_names=["last_hidden_state"],
        dynamic_axes={"input_features": {0: "batch", 1: "frames"},
                      "last_hidden_state": {0: "batch", 1: "frames"}},
        opset_version=17, do_constant_folding=True,
    )
    try:
        torch.onnx.export(fe, (dummy,), str(path), dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(fe, (dummy,), str(path), **kw)
    err = _parity(path, {"input_features": dummy}, lambda: fe(dummy))
    return path, err


def _export_decoder(output_dir: Path) -> tuple[Path, float]:
    dec = _load_jit(_DEC_PT)
    dummy = torch.randn(1, 1024, 100)
    path = output_dir / "decoder.onnx"
    kw = dict(
        input_names=["features"], output_names=["waveform"],
        dynamic_axes={"features": {0: "batch", 2: "frames"},
                      "waveform": {0: "batch", 2: "samples"}},
        opset_version=17, do_constant_folding=True,
    )
    try:
        torch.onnx.export(dec, (dummy,), str(path), dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(dec, (dummy,), str(path), **kw)
    err = _parity(path, {"features": dummy}, lambda: dec(dummy))
    return path, err


def _parity(path: Path, feeds: dict, torch_call) -> float:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = torch_call().numpy()
    got = sess.run(None, {k: v.numpy() for k, v in feeds.items()})[0]
    n = min(ref.shape[-1], got.shape[-1])
    err = float(np.abs(ref[..., :n] - got[..., :n]).max())
    print(f"{path.name}  {path.stat().st_size / 1e6:.1f} MB  parity max abs err={err:.2e} "
          f"{'PASS' if err < 1e-3 else 'FAIL'}")
    return err


def _quantize(path: Path) -> Path:
    """Dynamic int8 quantization of the (large) feature-extractor transformer."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process

    pre = path.with_name("feature_extractor.preproc.onnx")
    out = path.with_name("feature_extractor.int8.onnx")
    quant_pre_process(str(path), str(pre))
    quantize_dynamic(str(pre), str(out), weight_type=QuantType.QInt8)
    pre.unlink(missing_ok=True)
    print(f"{out.name}  {out.stat().st_size / 1e6:.1f} MB  (int8 from {path.stat().st_size / 1e6:.1f} MB)")
    return out


def _mel_parity() -> float:
    """Verify the adapter's numpy SeamlessM4T front-end against transformers."""
    import transformers

    from audiosronnx.engines.sidon import seamless_fbank

    pre = transformers.SeamlessM4TFeatureExtractor.from_pretrained(
        "facebook/w2v-bert-2.0", sampling_rate=16000)
    rng = np.random.RandomState(0)
    wav = (rng.randn(16000).astype(np.float32) * 0.1)
    ref = pre(np.pad(wav, (160, 160)), sampling_rate=16000,
              return_tensors="np")["input_features"][0]
    got = seamless_fbank(np.pad(wav, (160, 160)))
    n = min(ref.shape[0], got.shape[0])
    err = float(np.abs(ref[:n] - got[:n]).max())
    print(f"mel front-end parity max abs err={err:.2e} {'PASS' if err < 1e-4 else 'FAIL'}")
    return err


def _latency(fe_path: Path, dec_path: Path) -> None:
    import onnxruntime as ort

    fe = ort.InferenceSession(str(fe_path), providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(str(dec_path), providers=["CPUExecutionProvider"])
    feat_in = np.random.randn(1, 3000, 160).astype(np.float32)  # ~60 s at 50 Hz
    t0 = time.perf_counter()
    hidden = fe.run(None, {"input_features": feat_in})[0]
    dec.run(None, {"features": np.ascontiguousarray(hidden.transpose(0, 2, 1))})
    dt = time.perf_counter() - t0
    audio_s = 3000 / 50.0 * (48000 / 48000)
    print(f"CPU latency for ~{audio_s:.0f} s output: {dt:.2f} s  ({audio_s / dt:.1f}x realtime)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("./out/sidon"))
    ap.add_argument("--no-quantize", action="store_true")
    a = ap.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)

    fe_path, fe_err = _export_feature_extractor(a.output_dir)
    dec_path, dec_err = _export_decoder(a.output_dir)
    fe_run = fe_path
    if not a.no_quantize:
        fe_run = _quantize(fe_path)
    _mel_parity()
    _latency(fe_run, dec_path)
    print(f"graphs: fe_err={fe_err:.2e} dec_err={dec_err:.2e}")


if __name__ == "__main__":
    main()
