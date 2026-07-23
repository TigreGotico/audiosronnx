# Conversion

Weights are exported from the upstream PyTorch releases, validated, and published to the
Hugging Face Hub pinned by revision. The scripts live in [`../conversion/`](../conversion)
and need the `convert` extra:

```bash
pip install 'audiosronnx[convert]'
python conversion/export_sidon.py --output-dir ./out/sidon
```

They are maintainer tools. Nothing in `conversion/` is imported at runtime, and torch is
never a runtime dependency.

## Hosted weights

| Repository | Files |
|------------|-------|
| `TigreGotico/audiosronnx-lavasr` | `backbone.onnx`, `spec_head.onnx`, `denoiser_core.onnx` |
| `TigreGotico/audiosronnx-novasr` | `novasr.onnx` |
| `TigreGotico/audiosronnx-hifiganbwe` | `hifiganbwe_wavenet.onnx` |
| `TigreGotico/audiosronnx-apbwe` | `apbwe.onnx` |
| `TigreGotico/audiosronnx-sidon` | `feature_extractor.int8.onnx`, `decoder.onnx` |
| `TigreGotico/audiosronnx-unipase` | `encoder_adapter.onnx`, `vocoder.onnx` |
| `TigreGotico/audiosronnx-callenhancer` | `feature_extractor.onnx` (+ `.data`), `feature_extractor.int8.onnx`, `decoder.onnx` |
| `TigreGotico/audiosronnx-deepfilternet` | `enc.onnx`, `erb_dec.onnx`, `df_dec.onnx` |
| `TigreGotico/audiosronnx-dpdfnet` | 8 variants across 8 / 16 / 48 kHz |
| `TigreGotico/audiosronnx-gtcrn` | `gtcrn_simple.onnx` |
| `TigreGotico/audiosronnx-frcrn` | `frcrn.onnx` |
| `TigreGotico/audiosronnx-mossformer2` | `mossformer2_48k.onnx` |
| `TigreGotico/audiosronnx-mpsenet` | `mpsenet_dns.onnx`, `mpsenet.onnx` |

## Validation

A graph is not accepted on a successful export. Each is checked two ways:

1. **Per-graph parity** against the PyTorch module on identical input.
2. **End-to-end parity** against the *upstream inference pipeline*, including whatever
   normalisation, padding and windowing that pipeline applies outside the model.

The second matters more than the first. Reproducing a model but not its surrounding
pipeline yields audio that sounds plausible and is quietly wrong.

| Engine | End-to-end agreement |
|--------|---------------------|
| dpdfnet | max abs err 1.7e-8 |
| mossformer2 | correlation 0.99999999 |
| frcrn | correlation 0.99999994 |
| mpsenet | correlation 0.99998752 |
| hifiganbwe | correlation 1.0000 |
| apbwe | correlation 0.9998 |
| unipase | correlation 0.99999988 |

## Export hazards

Three failure modes that a naive parity check passes.

### Constant folding bakes in the traced length

`do_constant_folding=True` can specialise a graph to the sequence length it was traced at.
MossFormer2 exported this way reads a clean 1e-4 **at the traced length** and is badly
wrong elsewhere — max abs err 1.4 at 1.2× that length, 4.4 at 2×.

Export attention models with `do_constant_folding=False`, and sweep several lengths against
the PyTorch module afterwards. Fully-convolutional graphs are immune; `frcrn` was verified
length-independent with folding enabled.

### `nn.MultiheadAttention` specialises the sequence length

Stock `torch.nn.MultiheadAttention` bakes its reshapes to the traced length under both the
TorchScript and dynamo exporters. The resulting graph raises a `Reshape` shape mismatch at
any other length — loud rather than silent, but still fatal.

The fix is a drop-in attention module that reuses the same `in_proj_weight` / `out_proj`
weights and writes every reshape with `-1`. MP-SENet's published graph was exported that
way; the replacement reproduces stock attention bit-identically.

### The pipeline outside the graph

Upstream code routinely normalises, pads or scales around the model, and the model is often
sensitive to it:

- **FRCRN** applies a two-stage RMS normalisation to −25 dBFS and an idiosyncratic padding
  rule. Omitting them drops end-to-end correlation to **0.27** while still producing
  speech-like output.
- **MossFormer2** scales the waveform by 32768 before its filterbank; log-mel is not
  scale-invariant.
- **DPDFNet** seeds its recurrent state from values embedded in the ONNX metadata. Starting
  from zeros audibly corrupts the first second.

The reliable way to find these is to instrument the upstream pipeline — wrap the model's
`forward` to capture exactly what tensor it receives — rather than reading the code and
inferring.

## Checkpoints are not interchangeable

Where a project publishes several checkpoints, they can differ more than the architecture
does. MP-SENet's DNS Challenge checkpoint recovers **+4.8 / +8.9 / +11.7 dB** on broadband
noise where its VoiceBank+DEMAND checkpoint manages **+1.5 / +2.0 / +4.1 dB**. Both are
published, and `dns` is the default.

Measure each checkpoint rather than taking the one the paper's headline number was reported
on.
