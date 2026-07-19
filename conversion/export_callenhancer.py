"""Export CallEnhancer (Scicom-intl/CallEnhancer) call-centre restoration to ONNX.

CallEnhancer restores narrowband, codec'd telephony / call-centre speech to clean
48 kHz. Like Sidon it is two neural stages, but a heavier feature extractor:

- a **feature extractor** — the *full* 24-layer w2v-BERT 2.0 encoder
  (``facebook/w2v-bert-2.0``) LoRA-adapted (r=64, alpha=16) on the ``output_dense``
  projections, mapping SeamlessM4T log-mel features ``[1, T, 160]`` to hidden states
  ``[1, T, 1024]`` at 50 Hz. The LoRA adapter is **merged into the base weights** here
  (``W_eff = W + (alpha/r)*B@A``, trained ``lora_only`` biases replace the base biases),
  so inference needs neither ``peft`` nor the adapter at runtime.
- a **decoder** — a DAC vocoder (188M params, rates ``[8, 5, 4, 3, 2]`` -> 960x
  upsample) turning ``[1, 1024, T]`` hidden states into a ``[1, 1, 960*T]`` 48 kHz
  waveform.

Upstream ships PyTorch checkpoints (``Scicom-intl/CallEnhancer``:
``fe_adapter_full.pt`` — the LoRA adapter + hyperparams; ``decoder_only.pt`` — the DAC
decoder state dict). This script rebuilds both eager modules, merges the LoRA, traces
them to ONNX (opset 17), then dynamic-quantizes the feature extractor to int8 so the
transformer runs at a sane size/speed on CPU; the decoder stays fp32.

The SeamlessM4T mel front-end is *not* exported — it stays in numpy in the adapter
(``engines/callenhancer.py`` reuses ``engines/sidon.seamless_fbank``), matching the
repo's "no torch at runtime, STFT/mel in numpy" rule. This script parity-checks that
numpy front-end against ``transformers.SeamlessM4TFeatureExtractor``.

ONNX artifact: ``TigreGotico/audiosronnx-callenhancer`` (CC-BY-NC-4.0).

Usage::

    python export_callenhancer.py --output-dir ./out/callenhancer            # pulls .pt from HF
    python export_callenhancer.py --output-dir ./out/callenhancer --no-quantize

Requires: torch, torchaudio, transformers>=4.56, descript-audio-codec, onnx,
onnxruntime, huggingface_hub (the ``convert`` extra).

Reference
---------
- https://huggingface.co/Scicom-intl/CallEnhancer  (CC-BY-NC-4.0)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

_HF_SRC = "Scicom-intl/CallEnhancer"
_FE_PT = "fe_adapter_full.pt"
_DEC_PT = "decoder_only.pt"
_SSL_MODEL = "facebook/w2v-bert-2.0"


def _hf(filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(_HF_SRC, filename=filename)


def _build_fe() -> torch.nn.Module:
    """Rebuild the 24-layer w2v-BERT base and merge the trained LoRA adapter into it."""
    from transformers import Wav2Vec2BertModel

    ck = torch.load(_hf(_FE_PT), map_location="cpu")
    ad = ck["adapter"]
    scaling = ck["lora_alpha"] / ck["r"]
    layers = ck.get("layers", 24)
    model = Wav2Vec2BertModel.from_pretrained(_SSL_MODEL, num_hidden_layers=layers, layerdrop=0.0)
    sd = model.state_dict()
    prefixes = sorted({k[: -len(".lora_A.default.weight")]
                       for k in ad if k.endswith(".lora_A.default.weight")})
    for p in prefixes:                       # p e.g. encoder.layers.0.ffn1.output_dense
        A = ad[p + ".lora_A.default.weight"].float()   # (r, in)
        B = ad[p + ".lora_B.default.weight"].float()   # (out, r)
        delta = scaling * (B @ A)                       # (out, in)
        wkey = p + ".weight"
        sd[wkey] = sd[wkey].float() + delta.to(sd[wkey].dtype)
        bkey = p + ".base_layer.bias"                   # trained (lora_only) bias
        if bkey in ad:
            sd[p + ".bias"] = ad[bkey].to(sd[p + ".bias"].dtype)
    model.load_state_dict(sd)
    model.eval()
    print(f"[fe] merged LoRA into {len(prefixes)} output_dense layers (scaling={scaling}, "
          f"layers={layers})")

    class _Unpack(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_features):
            return self.m(input_features=input_features).last_hidden_state

    return _Unpack(model).eval()


def _build_decoder() -> torch.nn.Module:
    import dac

    ck = torch.load(_hf(_DEC_PT), map_location="cpu")
    ch = ck.get("dec_channels", 3072)
    dec = dac.model.dac.Decoder(input_channel=1024, channels=ch, rates=[8, 5, 4, 3, 2])
    dec.load_state_dict(ck["decoder"])
    dec.eval()
    print(f"[dec] DAC decoder channels={ch} "
          f"({sum(p.numel() for p in dec.parameters()) / 1e6:.1f}M params)")
    return dec


def _export_feature_extractor(output_dir: Path) -> tuple[Path, float]:
    fe = _build_fe()
    dummy = torch.randn(1, 100, 160)
    path = output_dir / "feature_extractor.onnx"
    kw = dict(
        input_names=["input_features"], output_names=["last_hidden_state"],
        dynamic_axes={"input_features": {0: "batch", 1: "frames"},
                      "last_hidden_state": {0: "batch", 1: "frames"}},
        opset_version=17, do_constant_folding=True,
    )
    with torch.no_grad():
        try:
            torch.onnx.export(fe, (dummy,), str(path), dynamo=False, **kw)
        except TypeError:
            torch.onnx.export(fe, (dummy,), str(path), **kw)
    err = _parity(path, {"input_features": dummy}, lambda: fe(dummy))
    return path, err


def _export_decoder(output_dir: Path) -> tuple[Path, float]:
    dec = _build_decoder()
    dummy = torch.randn(1, 1024, 100)
    path = output_dir / "decoder.onnx"
    kw = dict(
        input_names=["features"], output_names=["waveform"],
        dynamic_axes={"features": {0: "batch", 2: "frames"},
                      "waveform": {0: "batch", 2: "samples"}},
        opset_version=17, do_constant_folding=True,
    )
    with torch.no_grad():
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
    print(f"{out.name}  {out.stat().st_size / 1e6:.1f} MB  "
          f"(int8 from {path.stat().st_size / 1e6:.1f} MB)")
    return out


def _mel_parity() -> float:
    """Verify the adapter's numpy SeamlessM4T front-end against transformers."""
    import transformers

    from audiosronnx.engines.sidon import seamless_fbank

    pre = transformers.SeamlessM4TFeatureExtractor.from_pretrained(
        _SSL_MODEL, sampling_rate=16000)
    rng = np.random.RandomState(0)
    wav = rng.randn(16000).astype(np.float32) * 0.1
    ref = pre(np.pad(wav, (40, 40)), sampling_rate=16000,
              return_tensors="np")["input_features"][0]
    got = seamless_fbank(np.pad(wav, (40, 40)))
    n = min(ref.shape[0], got.shape[0])
    err = float(np.abs(ref[:n] - got[:n]).max())
    print(f"mel front-end parity max abs err={err:.2e} {'PASS' if err < 1e-4 else 'FAIL'}")
    return err


def _latency(fe_path: Path, dec_path: Path) -> None:
    import onnxruntime as ort

    fe = ort.InferenceSession(str(fe_path), providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(str(dec_path), providers=["CPUExecutionProvider"])
    feat_in = np.random.randn(1, 500, 160).astype(np.float32)  # ~10 s at 50 Hz
    t0 = time.perf_counter()
    hidden = fe.run(None, {"input_features": feat_in})[0]
    dec.run(None, {"features": np.ascontiguousarray(hidden.transpose(0, 2, 1))})
    dt = time.perf_counter() - t0
    audio_s = 500 / 50.0
    print(f"CPU latency for ~{audio_s:.0f} s output: {dt:.2f} s  ({audio_s / dt:.2f}x realtime)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("./out/callenhancer"))
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
