# DeepFilterNet3 → audiosronnx (`deepfilternet` denoise engine)

How the DeepFilterNet3 denoiser is wired into audiosronnx, and why it is assembled from
prebuilt ONNX rather than exported here.

## Why prebuilt ONNX

DeepFilterNet3 does not export from torch: the full model dies on `aten::view_as_complex`
(unsupported at every opset), and the per-module export hits `Unsupported value kind:
Tensor`. Its deep-filtering stage is complex-valued, which the tracer will not take.

DeepFilterNet publishes the graphs its own Rust/tract runtime uses, so audiosronnx consumes
those directly:

    https://github.com/Rikorose/DeepFilterNet/raw/1e96ef05e1ef75b3702f8c55ca065368deae637d/models/DeepFilterNet3_onnx.tar.gz

| graph | inputs | outputs |
|---|---|---|
| `enc.onnx` | `feat_erb[1,1,T,32]`, `feat_spec[1,2,T,96]` | `e0..e3`, `emb[1,T,512]`, `c0[1,64,T,96]`, `lsnr` |
| `erb_dec.onnx` | `emb`, `e3`, `e2`, `e1`, `e0` | `m[1,1,T,32]` — ERB gain mask |
| `df_dec.onnx` | `emb`, `c0` | `coefs[1,T,96,10]` — deep-filter taps |

Mirrored to `TigreGotico/audiosronnx-deepfilternet`.

## No torch

`libdf` (the `DeepFilterLib` wheel — Rust) supplies the STFT/ERB frontend and the synthesis,
and runs **without torch**; inference is onnxruntime only. Torch must *not* be installed
alongside it: every version tried (2.1.0, 2.4.1) segfaults `libdf` through an ABI clash.
That also rules out `df.enhance.enhance()` as a bit-exact reference, hence the signal-level
acceptance test below.

## Pipeline

Constants (`DeepFilterNet3/config.ini`): sr 48000 · fft 960 · hop 480 · nb_erb 32 ·
nb_df 96 · df_order 5 · df_lookahead 2 · conv_lookahead 2 · min_nb_erb_freqs 2 ·
norm_tau 1.0 → `alpha = exp(-(hop/sr)/tau) = 0.99`.

1. `spec = df.analysis(audio)` → `[1, T, 481]` complex.
2. `feat_erb = erb_norm(erb(spec, erb_widths), alpha)`; `feat_spec = unit_norm(spec[:, :96], alpha)` as re/im.
3. **`pad_feat` both features** — see below.
4. `enc` → `emb`, `c0`, `e0..e3`; `erb_dec` → `m`; `df_dec` → `coefs`.
5. Mask: `spec *= erb_inv(m, erb_widths)` (32 bands → 481 bins).
6. Deep filter over the low 96 bins, then `df.synthesis(spec)`.

## The two things that are easy to get wrong

Both are caught by the acceptance test; keep them in mind if the numbers ever move.

- **`coefs` are freq-major.** `df_dec` emits `[B, T, F=96, O=5, 2]` (bin, then tap, then
  re/im) — *not* `[B, T, O, F, 2]`. Reshape freq-first, then `transpose(0, 3, 1, 2)` to the
  `[B, O, T, F]` the einsum wants. Transposing O and F yields fluent-sounding but wrong
  output: `corr(clean, denoised)` collapses to **0.31**.
- **`pad_feat` lives outside the exported graph.** The model applies
  `ConstantPad2d((0, 0, -conv_lookahead, conv_lookahead))` to both features before `enc`: a
  *negative* pad — drop 2 frames from the front, append 2 zero frames. Skipping it puts every
  mask and coefficient 2 frames out of step (corr 0.9697 → 0.9533).

The deep filter itself is `einsum("tfn,ntf->tf", taps, coefs)` over time-unfolded frames,
padded `(df_order-1-df_lookahead)` before and `df_lookahead` after — matching
`df/multiframe.py`.

## Acceptance

Bit-exact parity against upstream is unavailable (torch segfaults libdf), so the engine is
validated at the signal level: real speech + white noise must come back cleaner *and* still
be the same speech.

| | SI-SDR | corr(clean, denoised) |
|---|---|---|
| noisy input | 2.82 dB | 0.8105 |
| **denoised** | **11.97 dB** | **0.9697** |

**+9.15 dB SI-SDR.** Stage ablations: mask-only 0.9231, mask+DF 0.9533, mask+DF+`pad_feat`
0.9697. The STFT round-trip is exact on its own (corr 0.9999, one hop of latency), so a
regression here points at the mask or the filter, not the frontend.

## Do not put this in front of ASR

Denoising **degrades** recognition. Measured on 16 real Common Voice Arabic clips against
their ground-truth sentences, corrupted with pink and babble noise at 0/5/10 dB and
transcribed with the project's Arabic ASR:

| noise | SNR | noisy WER | denoised WER |
|---|---|---|---|
| pink | 0 | 0.508 | **0.716** |
| pink | 5 | 0.437 | 0.477 |
| pink | 10 | 0.376 | 0.374 |
| babble | 0 | 0.831 | 0.804 |
| babble | 5 | 0.585 | 0.674 |
| babble | 10 | 0.462 | 0.454 |

Mean WER **0.533 → 0.583** (improved 19/96 conditions, worse 35/96); the loss is largest at
low SNR, where help is most wanted. This is the expected trade — the model optimizes
perceptual quality, and the artifacts it leaves cost a recognizer more than the noise it
removes. The engine is correct (see the SI-SDR/ablation results above); the metric simply
disagrees with the use case.

So: use it for **perceptual** cleanup — reference clips for cloning, listening sets — and
judge it by ear or speaker similarity. Keep it out of any path feeding ASR or a WER gate.
(Caveat: the clean-audio baseline is already 0.242 WER, so the recognizer is weak on this
material; the direction is solid, the magnitudes are soft.)
