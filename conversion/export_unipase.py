"""Export UniPASE (Xiaobin-Rong/unipase) universal speech enhancement to ONNX.

UniPASE is generative universal speech enhancement — noise and reverberation removed in one
feed-forward pass at 16 kHz. Two graphs, the same shape as sidon/voicefixer (a predictor
followed by a neural vocoder):

- **encoder_adapter** — the DeWavLM-Omni WavLM encoder (conv front-end + 24-layer
  transformer with gated relative-position bias) whose L1 and L24 layer reps are fused by a
  Vocos adapter: ``wav[1, 128000]`` -> ``feat[1, 400, 1024]``. Shipped **fp32** (a single
  ~1.7 GB graph, under the 2 GB protobuf limit) with an optional near-lossless **fp16**
  variant (``--fp16``, end-to-end corr 0.99993, ~45%% smaller download). int8 is **not**
  shipped: the per-layer ``--sweep`` shows the 24-layer WavLM-Large loses too much to
  dynamic quantization — whole-transformer int8 drops end-to-end corr to ~0.86 (the naive
  whole-graph value is ~0.85), the same depth-driven degradation that makes CallEnhancer
  default to fp32, and the widest int8 set holding >=0.999 barely shrinks the graph. fp16
  keeps the conv front-end and every LayerNorm in fp32, casting only the heavy weight
  MatMuls; boundary casts are inserted explicitly (the stock converters leave mixed-type
  edges ORT refuses to load).
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
        --output-dir ./out/unipase --speech reference.wav [--fp16] [--sweep]

Requires: torch, torchaudio, onnx, onnxruntime, huggingface_hub (the ``convert`` extra),
plus the upstream repo checked out (``--unipase-src``) so ``models`` is importable.

Reference
---------
- https://github.com/Xiaobin-Rong/unipase  (MIT)
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HF_SRC = "Xiaobin-Rong/unipase"
_SR = 16000
_SEG = _SR * 8
_N_FFT, _HOP = 1280, 320
_NORM_OPS = ("LayerNormalization", "InstanceNormalization")
#: transformer weight-MatMul name suffixes eligible for fp16 / int8. The conv front-end,
#: the gated relative-position math (``grep_linear``, the activation-only ``MatMul_3/4``),
#: every LayerNorm, the Vocos adapter and the vocoder are never touched.
_HEAVY_SUFFIXES = (
    "/fc1/MatMul", "/fc2/MatMul",
    "/self_attn/MatMul", "/self_attn/MatMul_1", "/self_attn/MatMul_2", "/self_attn/Gemm",
    "/pwconv1/MatMul", "/pwconv2/MatMul", "/adp/proj/MatMul", "/adp/head/MatMul",
    "/post_extract_proj/MatMul",
)


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


# --------------------------------------------------------------------------- #
# fp16 — deterministic boundary-cast conversion (needs no fp16 runtime).
# --------------------------------------------------------------------------- #
def _float_tensor_names(src: Path) -> set:
    """Names of all float32 tensors in ``src`` (disk-based shape inference, memory-safe)."""
    import onnx

    inferred = src.with_name(src.stem + ".shapes.onnx")
    onnx.shape_inference.infer_shapes_path(str(src), str(inferred))
    m = onnx.load(str(inferred), load_external_data=False)
    fl = {vi.name for vi in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output)
          if vi.type.tensor_type.elem_type == onnx.TensorProto.FLOAT}
    fl |= {i.name for i in m.graph.initializer if i.data_type == onnx.TensorProto.FLOAT}
    inferred.unlink(missing_ok=True)
    return fl


def build_fp16(src: Path, dst: Path, keep_fp32_node) -> Path:
    """Cast ``src`` to fp16, keeping ``keep_fp32_node(node)`` nodes and all norms in fp32.

    onnxruntime/onnxconverter-common's own casting is unreliable on this graph (the WavLM
    conv front-end's Transpose/Cast/GELU pattern and the pervasive fp32 LayerNorms leave
    mixed-type edges ORT refuses to load). This does the conversion explicitly instead: it
    up-casts only the initializers feeding fp16 nodes, then inserts a ``Cast`` on every float
    edge whose producer dtype differs from what the consumer runs in. Graph I/O stay fp32.
    Deterministic and needs no fp16 kernels; the model never leaves fp32 arithmetic on CPU.
    """
    import onnx
    from onnx import helper, numpy_helper as nh, TensorProto

    fp32, fp16 = TensorProto.FLOAT, TensorProto.FLOAT16
    float_names = _float_tensor_names(src)
    m = onnx.load(str(src))
    g = m.graph
    fp32_nodes = {n.name for n in g.node if n.op_type in _NORM_OPS or keep_fp32_node(n)}

    def is16(node):
        return node.name not in fp32_nodes

    producer, cast_to = {}, {}
    for n in g.node:
        for o in n.output:
            producer[o] = n
        if n.op_type == "Cast":
            cast_to[n.name] = next(a.i for a in n.attribute if a.name == "to")

    inputs = {i.name for i in g.input}
    consumed16, consumed32 = set(), set()
    for n in g.node:
        (consumed16 if is16(n) else consumed32).update(n.input)

    for tp in g.initializer:                                   # initializers used only by fp16 nodes
        if tp.data_type == fp32 and tp.name in consumed16 and tp.name not in consumed32:
            tp.CopyFrom(nh.from_array(nh.to_array(tp).astype(np.float16), tp.name))
    for n in g.node:                                           # Constant tensors owned by fp16 nodes
        if n.op_type == "Constant" and is16(n) and n.output[0] in float_names:
            for a in n.attribute:
                if a.name == "value" and a.t.data_type == fp32:
                    a.t.CopyFrom(nh.from_array(nh.to_array(a.t).astype(np.float16)))
    init = {i.name: i for i in g.initializer}

    def dtype(t):
        if t not in float_names:
            return None
        if t in init:
            return init[t].data_type
        if t in inputs:
            return fp32
        p = producer.get(t)
        if p is None:
            return fp32
        return cast_to.get(p.name, fp32) if p.op_type == "Cast" else (fp16 if is16(p) else fp32)

    cache, new_nodes, ctr = {}, [], [0]

    def cast(t, want):
        if (t, want) not in cache:
            ctr[0] += 1
            out = f"{t}__cast{'16' if want == fp16 else '32'}_{ctr[0]}"
            new_nodes.append(helper.make_node("Cast", [t], [out], to=want, name=f"bcast_{ctr[0]}"))
            cache[(t, want)] = out
        return cache[(t, want)]

    for n in g.node:
        if n.op_type == "Cast":
            continue
        want = fp16 if is16(n) else fp32
        for i, t in enumerate(n.input):
            if dtype(t) not in (None, want):
                n.input[i] = cast(t, want)
    for o in g.output:                                         # keep graph outputs fp32
        p = producer.get(o.name)
        if o.name in float_names and p is not None and p.op_type != "Cast" and is16(p):
            tmp = o.name + "__fp16"
            for nn in g.node:
                nn.output[:] = [tmp if x == o.name else x for x in nn.output]
            new_nodes.append(helper.make_node("Cast", [tmp], [o.name], to=fp32, name=f"ocast_{o.name}"))

    g.node.extend(new_nodes)
    _topo_sort(g)
    onnx.save(m, str(dst))
    print(f"{dst.name}  {dst.stat().st_size / 1e6:.1f} MB  (fp16 from {src.stat().st_size / 1e6:.1f} MB)")
    return dst


def _topo_sort(g) -> None:
    have = {i.name for i in g.initializer} | {i.name for i in g.input}
    pending, ordered = list(g.node), []
    while pending:
        rest, moved = [], False
        for n in pending:
            if all(x in have or x == "" for x in n.input):
                ordered.append(n); have.update(n.output); moved = True
            else:
                rest.append(n)
        pending = rest
        if not moved:
            ordered.extend(pending); break
    del g.node[:]
    g.node.extend(ordered)


def _keep_fp32_heavy_only(node) -> bool:
    """fp32 for everything except the heavy transformer weight MatMuls (the fp16 target)."""
    return not any(node.name.endswith(s) for s in _HEAVY_SUFFIXES)


# --------------------------------------------------------------------------- #
# int8 — selective dynamic quantization + per-layer sensitivity sweep.
# --------------------------------------------------------------------------- #
def _safe_int8_nodes(src: Path):
    """{layer -> [MatMul names]} for the FFN + q/k/v-projection weight MatMuls per layer.

    Excludes ``grep_linear`` (gated relative-position math), the activation-only attention
    MatMuls, the conv front-end and the adapter — the quantization-sensitive parts.
    """
    import onnx

    m = onnx.load(str(src))
    inits = {i.name for i in m.graph.initializer}
    safe_suf = ("/fc1/MatMul", "/fc2/MatMul",
                "/self_attn/MatMul", "/self_attn/MatMul_1", "/self_attn/MatMul_2")
    per = {}
    for n in m.graph.node:
        if n.op_type != "MatMul":
            continue
        mo = re.match(r"/encoder/layers\.(\d+)/", n.name)
        if mo and any(n.name.endswith(s) for s in safe_suf) and any(x in inits for x in n.input):
            per.setdefault(int(mo.group(1)), []).append(n.name)
    return {k: per[k] for k in sorted(per)}


def quantize_int8(src: Path, dst: Path, nodes) -> Path:
    """Weights-only int8 dynamic quant (QOperator) of exactly ``nodes`` (MatMul names)."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul"], nodes_to_quantize=list(nodes))
    return dst


def sensitivity_sweep(src: Path, score, layer_kinds=("/self_attn/", "/fc1/", "/fc2/")):
    """Per-layer int8 sensitivity sweep: quantize one layer's safe MatMuls at a time.

    ``score(quantized_path) -> float`` runs the caller's end-to-end pipeline and returns
    correlation vs the fp32 output on real speech. Returns ``{layer -> corr}``. This is the
    evidence behind unipase shipping fp32/fp16 only: on this 24-layer WavLM-Large even a
    single quantized layer falls short of the 0.999 bar, and quantizing all FFN + projection
    weights drops end-to-end corr to ~0.86 (near the naive whole-graph 0.85). The widest set
    that holds >=0.999 (q/k/v projections of ~6 middle layers) barely shrinks the graph, so no
    int8 encoder is published — see ``_FILES`` in ``engines/unipase.py``.
    """
    safe = _safe_int8_nodes(src)
    tmp = src.with_name(src.stem + ".sweep.int8.onnx")
    out = {}
    for layer, nodes in safe.items():
        quantize_int8(src, tmp, nodes)
        out[layer] = score(tmp)
        print(f"  layer {layer:2d}: corr {out[layer]:.6f}", flush=True)
    tmp.unlink(missing_ok=True)
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
    ap.add_argument("--fp16", action="store_true",
                    help="also emit fp16 encoder + vocoder (~45%% smaller download, corr 0.99993)")
    ap.add_argument("--sweep", action="store_true",
                    help="run the int8 per-layer sensitivity sweep (evidence for shipping no int8)")
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

    # ---- parity: torch full core vs the shipped fp32 ONNX graphs + numpy ISTFT ----
    import onnxruntime as ort

    def _onnx_e2e(enc_path: Path, voc_path: Path):
        es = ort.InferenceSession(str(enc_path), providers=["CPUExecutionProvider"])
        vsess = ort.InferenceSession(str(voc_path), providers=["CPUExecutionProvider"])
        t0 = time.perf_counter()
        f = es.run(None, {"wav": wav.numpy()})[0]
        s = vsess.run(None, {"feat": np.ascontiguousarray(f.transpose(0, 2, 1))})[0]
        return _vocos_istft(s), time.perf_counter() - t0

    with torch.inference_mode():
        twav = voc(ref_feat.transpose(1, 2)).numpy().ravel()
    owav, dt = _onnx_e2e(ea_path, v_path)
    n = min(twav.size, owav.size)
    corr = float(np.corrcoef(twav[:n], owav[:n])[0, 1])
    err = float(np.abs(twav[:n] - owav[:n]).max())
    print(f"end-to-end parity: corr={corr:.8f} max abs err={err:.2e} "
          f"{'PASS' if corr > 0.9999 else 'FAIL'}")
    print(f"CPU latency for 8 s output: {dt:.2f} s  ({8.0 / dt:.1f}x realtime)")

    def _corr_vs_fp32(enc_path, voc_path):
        o, _ = _onnx_e2e(enc_path, voc_path)
        k = min(o.size, owav.size)
        return float(np.corrcoef(o[:k], owav[:k])[0, 1])

    # ---- optional fp16 variant: fp32 front-end + norms, fp16 heavy weight MatMuls ----
    if a.fp16:
        ea16 = build_fp16(ea_path, a.output_dir / "encoder_adapter.fp16.onnx", _keep_fp32_heavy_only)
        v16 = build_fp16(v_path, a.output_dir / "vocoder.fp16.onnx", lambda n: False)  # norms fp32 only
        c = _corr_vs_fp32(ea16, v16)
        print(f"fp16 end-to-end parity vs fp32: corr={c:.6f} {'PASS' if c > 0.999 else 'FAIL'}")

    # ---- optional int8 sensitivity sweep (documents why no int8 encoder ships) ----
    if a.sweep:
        print("int8 per-layer sensitivity sweep (corr vs fp32; target >=0.999):")
        res = sensitivity_sweep(ea_path, lambda p: _corr_vs_fp32(p, v_path))
        best = max(res.values())
        print(f"best single-layer corr {best:.6f} — below 0.999; whole-transformer int8 ~0.86. "
              "No int8 encoder is published.")


if __name__ == "__main__":
    main()
