# Evaluated but not shipped

An engine ships when it meets four conditions: it exports to a **single static ONNX graph**,
it runs on CPU through onnxruntime, it carries a **permissive license**, and its output is
validated against the original implementation.

The models below fail at least one. Each entry names which.

## Rejected: license

| Model | License | Notes |
|-------|---------|-------|
| [facebookresearch/denoiser](https://github.com/facebookresearch/denoiser) (dns48/dns64) | **CC-BY-NC-4.0** | Architecturally an excellent fit — a causal Demucs-style waveform model, real-time on CPU by its authors' own measurements. Rejected purely on the non-commercial term. |
| [mdctGAN](https://github.com/neoncloud/mdctGAN) | **NOASSERTION** | No resolvable license. Its MDCT front-end also relies on `torch.fft`, which exports unreliably. |

`callenhancer` is the one CC-BY-NC engine that does ship. It is opt-in, its license is
stated at every point of use, and the restriction covers the weights rather than audio
processed with them.

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
| [MossFormerGAN_SE_16K](https://huggingface.co/alibabasglab/MossFormerGAN_SE_16K) | Apache-2.0 | The strongest reported PESQ (3.47) of any candidate surveyed. Blocked on an upstream checkpoint download that does not currently succeed. |
| [LiSenNet](https://github.com/hyyan2k/LiSenNet) | MIT | 56 K parameters, in the same ultra-light class as `gtcrn`, with a third-party ONNX port already published. Would only earn a slot by beating `gtcrn` at a comparable size. |
| [Fast-ULCNet](https://github.com/narrietal/Fast-ULCNet) | MIT | Low-complexity CNN + FastGRNN. Same class and same question as LiSenNet. |

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
