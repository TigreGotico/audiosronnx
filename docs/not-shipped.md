# Evaluated but not shipped

An engine ships when it meets four conditions: it exports to a **single static ONNX graph**,
it runs on CPU through onnxruntime, it carries a **permissive license**, and its output is
validated against the original implementation.

The models below fail at least one. Each entry names which.

## Rejected: license

| Model | License | Notes |
|-------|---------|-------|
| [mdctGAN](https://github.com/neoncloud/mdctGAN) | **NOASSERTION** | No resolvable license. Its MDCT front-end also relies on `torch.fft`, which exports unreliably. |

A restrictive license is **not** grounds for exclusion here — it is a labelling decision.
`callenhancer` and `metadenoiser` both ship under CC-BY-NC-4.0, opt-in, with the license
stated at every point of use and reported by `audiosronnx list`. The restriction covers the
weights rather than audio processed with them; whether that is acceptable is the caller's
call to make, not the library's.

What remains disqualifying is weights published with **no declared terms at all**, since
that leaves nothing to redistribute under.

## Rejected: iterative samplers and external vocoders

These share a shape that does not reduce to a static graph — a diffusion or flow sampler
running tens of network evaluations per utterance, and/or a separate neural vocoder.

| Model | License | Blocker |
|-------|---------|---------|
| [AudioSR](https://github.com/haoheliu/versatile_audio_super_resolution) | MIT | ~6 GB latent diffusion (VAE + LDM + vocoder). Iterative sampler, ~0.6× realtime **on GPU**. |
| [resemble-enhance](https://github.com/resemble-ai/resemble-enhance) | MIT | Four networks: UNet denoiser, IRMAE autoencoder, a CFM ODE sampler (32–64 evaluations per utterance), and a UnivNet vocoder. The vocoder is LVCNet, whose *location-variable* convolution kernels are predicted per position and applied via `einsum` over `unfold`s — this does not fold into a static graph. |
| [SGMSE / SGMSE+](https://github.com/sp-uhh/sgmse) | MIT | Score-based diffusion in the complex STFT domain, requiring iterative reverse-diffusion steps. Its own 2025 streaming follow-up reaches real time only on a consumer GPU. |
| [NU-Wave2](https://github.com/maum-ai/nuwave2) | BSD-3-Clause | Diffusion with an iterative sampler. The license is permissive; the sampler is the blocker. |
| [VoiceFixer / NVSR](https://github.com/haoheliu/voicefixer) | MIT | Two-stage ResUNet mel predictor plus a TFGAN neural vocoder. Restoration rather than bandwidth extension, and a multi-graph export. |

The `sidon` and `callenhancer` engines show the bar this rules out: both use an external
vocoder, but a **DAC decoder that is a plain convolutional stack**, exportable as one
graph. A vocoder is not disqualifying; a vocoder with dynamic, position-dependent kernels
is.

## Rejected: not an exportable graph

| Model | License | Blocker |
|-------|---------|---------|
| [RNNoise](https://github.com/xiph/rnnoise) | BSD-3-Clause | Ships as a hand-written C inference engine, not a trained graph in an exportable framework. Reproducing it would mean retraining an equivalent network. Useful as a reference architecture. |

## Rejected: superseded

| Model | License | Notes |
|-------|---------|-------|
| [NSNet2](https://github.com/microsoft/DNS-Challenge) | CC-BY-4.0 (weights) | Already published as ONNX, so it would be trivial to add — but it is a 2020-era gain-mask RNN comfortably beaten by every shipped denoiser. Adding it would widen the menu without improving any outcome. |

## Under consideration

Not rejected — evaluated as viable and not yet integrated.

| Model | License | State |
|-------|---------|-------|
| [Fast-ULCNet](https://github.com/narrietal/Fast-ULCNet) | MIT | Low-complexity CNN + FastGRNN, in `gtcrn`'s size class. Not yet attempted. |
| [VoiceFixer / NVSR](https://github.com/haoheliu/voicefixer) | MIT | ResUNet mel predictor plus a TFGAN vocoder — the same two-stage shape as `sidon`, so the old "multi-component" objection does not apply. Not yet attempted. |
| [NU-Wave2](https://github.com/maum-ai/nuwave2) | BSD-3-Clause | Diffusion, but few-step; the sampler loop would run in numpy outside the graph as `flowhigh`'s does. Not yet attempted. |

## A note on "multi-component" as a reason

An external vocoder or a multi-graph pipeline is **not** grounds for rejection here.
`deepfilternet` ships as three graphs, and `sidon` and `callenhancer` are both a feature
predictor followed by a neural vocoder. What actually disqualifies a design is *dynamic*
structure — a sampler loop whose length is data-dependent, or convolution kernels predicted
per position — not the number of graphs.

An iterative sampler is likewise closer to a cost problem than an export one: the loop can
run in numpy outside the graph, exactly as the STFT does. The objection to AudioSR and
SGMSE is that tens of network evaluations per utterance is impractical on CPU, not that the
graph cannot be produced.

## Attempted and blocked

Both of these were exported and measured rather than reasoned about.

### LiSenNet — the published ONNX port does not reconstruct

56 K parameters and under 300 KB, which would make it the smallest engine here, and the
third-party ONNX port is MIT. The graph runs and its output looks plausible (it attenuates,
median gain 0.39), but the full pipeline **destroys the signal**: −10.8 dB SNR at 11 dB
input.

That is not a porting error on this side. An adapter written to the port's own reference
implementation reproduces that reference **exactly**, and the reference produces the same
−10.8 dB. So the exported graph does not match the front-end its reference documents, and
there is no upstream-validated export to check against. Revisit if an official export
appears, or by exporting from the original PyTorch weights.

### MossFormerGAN_SE_16K — length-specialised reshape

Apache-2.0, 3.13 M parameters, and the strongest reported PESQ (3.47) of any candidate
surveyed. Two obstacles were cleared: `torch.complex` (rewritten to the identical `atan2`)
and `torch.eye(dtype=bool)`, which exports to `EyeLike(bool)` and has no onnxruntime
implementation (rebuilt as an arange equality).

What remains is structural: MossFormer's group attention reshapes the sequence into fixed
groups, and that reshape captures the traced length — the graph runs at its trace size and
fails elsewhere, even with constant folding disabled. Two ways forward: patch the chunking
to stay dynamic, or export at a fixed length and window the input, which upstream's own
decode already does with a 10 s window. Worth finishing given the quality on offer.

## Recurring export blockers

Patterns worth checking before investing in a candidate:

- **Iterative samplers** — diffusion and flow-matching models evaluate a network many
  times per utterance. There is no static graph to export.
- **Location-variable convolutions** — kernels predicted per position and applied through
  `unfold`/`einsum` (LVCNet, UnivNet) do not fold into fixed operators.
- **`torch.stft` inside the model** — exports unreliably. A *Fourier-kernel `Conv1d`* is a
  different matter and exports cleanly; `frcrn` ships on exactly that basis.
- **Unlicensed weights** — a permissive code license does not cover a checkpoint published
  without terms.
- **Runtime-constructed tensors** — `filter.expand(C, -1, -1)`, `torch.eye`, and similar
  build a tensor from an input's shape at call time. The tracer then sees an operand of
  unknown shape. Where the value is fixed per layer, materialising it as a buffer fixes the
  export and is bit-for-bit equivalent.
- **Unsupported ops with exact equivalents** — `torch.complex` has no ONNX operator, but
  `angle(complex(re, im))` is `atan2(im, re)`. `EyeLike(bool)` has no onnxruntime kernel,
  but an arange equality builds the same mask. These look like blockers and are not.
- **Sequence-length specialisation** — separate from constant folding. Group/chunked
  attention can capture the traced length in a reshape. Always sweep several lengths
  against the PyTorch module rather than checking one.
