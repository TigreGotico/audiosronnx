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

## Rejected: nothing to export

| Model | License | Blocker |
|-------|---------|---------|
| [RNNoise](https://github.com/xiph/rnnoise) | BSD-3-Clause | Ships as a hand-written C inference engine, not a trained graph in an exportable framework. Reproducing it would mean retraining an equivalent network. Useful as a reference architecture. |
| [Fast-ULCNet](https://github.com/narrietal/Fast-ULCNet) | MIT | The repository publishes the **architecture only** — PyTorch and TensorFlow model definitions and a FastGRNN package, with no trained checkpoint anywhere in the tree. There is nothing to export without training it. |

Neither is a licensing or tracing problem: in both cases no trained graph exists to convert.

## Rejected: superseded

| Model | License | Notes |
|-------|---------|-------|
| [NSNet2](https://github.com/microsoft/DNS-Challenge) | CC-BY-4.0 (weights) | Already published as ONNX, so it would be trivial to add — but it is a 2020-era gain-mask RNN comfortably beaten by every shipped denoiser. Adding it would widen the menu without improving any outcome. |

## Under consideration

Not rejected — evaluated as viable and not yet integrated.

| Model | License | State |
|-------|---------|-------|

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

### VoiceFixer — the analysis graph bakes in a traced input

MIT, and structurally a good fit: a ResUNet mel predictor plus a TFGAN vocoder, the same
two-stage shape as `sidon`. Both stages export, and the **vocoder is clean** — max abs err
1.5e-05 against torch, at 133 MB.

The analysis stage is not. Exported from `forward(sp, mel_orig)`, the resulting graph
declares **only `mel_orig` as an input**: the tracer folded `sp` into a constant taken from
the trace. Parity lands at **6.8e-02**, far above the 1e-03 this library accepts, and the
error is consistent with a baked-in constant rather than rounding.

Shipping it would mean a graph that silently computes against one recording's spectrogram
for every input. Resolving it means establishing whether the generator genuinely consumes
`sp` — and if it does, exporting so that it stays a live input. Like the others here it is
also length-specialised, so it would need a fixed window regardless.

### NU-Wave2 — the transform is inside the model

BSD-3-Clause, 1.71 M parameters, and its diffusion sampler is not the problem: like
`flowhigh`'s, that loop would run in numpy outside the graph.

The weights load cleanly once two quirks are handled — the checkpoint carries a **double**
`model.model.` key prefix, and it pickles training metadata (a Lightning callback, an
omegaconf config) that must be stubbed rather than installed, since those pins clash with
the torch present.

What blocks it is structural: ``NuWave2.forward`` calls ``torch.stft`` and ``torch.istft``
**inside the model**. That is the one thing this library deliberately keeps out of its
graphs, and it is also what exports least reliably. Shipping it means splitting the model at
the transform boundary so the graph takes a spectrogram and returns one — real
restructuring rather than an operator rewrite, and the reason it is not done here yet.

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
  attention (`mossformergan`) and Python-computed padding (`metadenoiser`) both capture the
  traced length. Always sweep several lengths against the PyTorch module rather than
  checking one. Where it cannot be removed, export at a fixed window and slide it.
- **The transform inside the model** — a model whose own ``forward`` calls ``torch.stft``
  needs splitting at that boundary before it can export well. This is restructuring, not an
  operator rewrite, and is what currently holds back NU-Wave2.
- **Export memory** — tracing attention at a long fixed window can simply run out of
  memory and die without a useful message. `mossformergan` failed silently at 1601 frames
  and exports comfortably at 401.
- **Silently dropped inputs** — if an exported graph declares fewer inputs than the module
  took, the tracer folded one into a constant. `voicefixer`'s analysis stage lost its `sp`
  argument this way and still ran, just wrongly. Check the exported input list against the
  signature, not only the parity number at the traced shape.
