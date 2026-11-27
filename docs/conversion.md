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
| `AudioSRonnx/lavasr` | `backbone.onnx`, `spec_head.onnx`, `denoiser_core.onnx` |
| `AudioSRonnx/novasr` | `novasr.onnx` |
| `AudioSRonnx/hifiganbwe` | `hifiganbwe_wavenet.onnx` |
| `AudioSRonnx/apbwe` | `apbwe.onnx` |
| `AudioSRonnx/sidon` | `feature_extractor.int8.onnx`, `decoder.onnx` |
| `AudioSRonnx/unipase` | `encoder_adapter.onnx`, `vocoder.onnx`, `encoder_adapter.fp16.onnx`, `vocoder.fp16.onnx` |
| `AudioSRonnx/callenhancer` | `feature_extractor.onnx` (+ `.data`), `feature_extractor.int8.onnx`, `decoder.onnx` |
| `AudioSRonnx/deepfilternet` | `enc.onnx`, `erb_dec.onnx`, `df_dec.onnx` |
| `AudioSRonnx/dpdfnet` | 8 variants across 8 / 16 / 48 kHz |
| `AudioSRonnx/gtcrn` | `gtcrn_simple.onnx` |
| `AudioSRonnx/frcrn` | `frcrn.onnx` |
| `AudioSRonnx/mossformer2` | `mossformer2_48k.onnx` |
| `AudioSRonnx/mpsenet` | `mpsenet_dns.onnx`, `mpsenet.onnx` |
| `AudioSRonnx/voiceisolator` | `step.onnx`, `enroll.onnx`, `config.json` per model: CPU `/`, `int8/`, `agent/`, `agent/int8/`; 555 M `large/`, `large/agent/` (+ `step.onnx.data`) |

The repositories are gathered in the
[audiosronnx collection](https://huggingface.co/collections/TigreGotico/audiosronnx-pure-onnx-speech-enhancement-6af87eb6110e592e16d35fe9).

The voice isolator's CPU folders are the upstream graphs unchanged. The `large/` folders
come from:

```bash
python conversion/export_voiceisolator.py --out ./out/large
python conversion/export_voiceisolator.py --sub agent --out ./out/large/agent
```

Each export writes `step.onnx` with its 2.2 GB of weights in `step.onnx.data`, `enroll.onnx`
and `config.json`. `load_denoise("voiceisolator", model_dir=...)` loads a local export.
`conversion/check_voiceisolator.py <dir> <speech.wav>` compares an export with upstream's
PyTorch streaming step on 14 s of audio, past the point where its 10 s rings wrap.

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
| voiceisolator (CPU release, both models, fp32 and int8) | max abs err 0.0 vs upstream's ONNX runtime |
| voiceisolator (555 M, release and agent, gated and ungated) | max abs err 1.2e-7 or better vs upstream `stream_step`, 14 s |
| unipase | correlation 0.99999988 (fp16 variant: 0.99993) |

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
