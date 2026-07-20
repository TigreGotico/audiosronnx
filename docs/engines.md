# Bandwidth-extension engines

Every engine here takes narrowband or low-fidelity speech and produces **48 kHz** output
through `load_sr()`. They differ in how much they invent, how much they cost, and whether
they merely extend the band or resynthesise the speech outright.

| Engine | Input | Size | CPU speed | License |
|--------|-------|------|-----------|---------|
| [lavasr](#lavasr) (default) | 8–48 kHz | ~52 MB | ~50× realtime | Apache-2.0 |
| [novasr](#novasr) | 16 kHz | ~0.2 MB | ~1000× realtime | Apache-2.0 |
| [hifiganbwe](#hifiganbwe) | any | ~4 MB | fast | MIT |
| [apbwe](#apbwe) | any (12 kHz band) | ~120 MB | moderate | MIT |
| [sidon](#sidon) | 16 kHz | ~410 MB | ~0.6× realtime | MIT |
| [callenhancer](#callenhancer) | 8–16 kHz | ~3 GB fp32 / ~1.3 GB int8 | ~0.3–0.4× realtime | CC-BY-NC-4.0 |

## Choosing

- **General purpose** — `lavasr`. It accepts any rate from 8 to 48 kHz, preserves the
  original low band through a Linkwitz-Riley merge, and runs far faster than realtime.
- **Tight footprint** — `novasr` at ~0.2 MB, or `hifiganbwe` at ~4 MB.
- **Best spectral accuracy** — `apbwe`.
- **Damaged, not merely narrowband** — `sidon` or `callenhancer`. These *resynthesise*
  speech rather than extend its band, which fixes codec artefacts and noise that a
  bandwidth extender cannot, at a much higher cost.
- **Telephony specifically** — `callenhancer`, trained on G.711/GSM call audio.

A restoration engine reconstructs the signal, so its output is **enhanced-real**, not
ground truth. That is fine as an acoustic-model target and fine for listening; treat any
high band it produces as invented.

## lavasr

Vocos-based bandwidth extension with a Linkwitz-Riley spectral merge that keeps the
original low band intact, plus an optional UL-UNAS denoiser stage. Accepts any input rate
from 8 to 48 kHz. All spectral DSP — STFT, ISTFT, mel filterbank, resampling, merge —
runs in numpy/scipy.

```python
sr = load_sr("lavasr", denoise=True, cutoff_hz=6000)
```

- `denoise` (bool) — run the UL-UNAS denoiser before extension.
- `cutoff_hz` (int) — crossover for the spectral merge.

## novasr

A ~52 KB conv1d / BigVGAN-snake generator that upsamples 16 kHz to 48 kHz in one
time-domain pass. Extremely fast and memory-light, at lower fidelity than lavasr. Suited
to on-device use and quick dataset passes.

## hifiganbwe

HiFi-GAN+ (Su et al., ICASSP 2021): bandlimited kaiser interpolation to 48 kHz followed by
a non-causal WaveNet that reconstructs the high band. Time-domain, ~1M parameters, no
spectral front-end, accepts any input rate.

## apbwe

AP-BWE (Lu et al.): dual-ConvNeXt amplitude-and-phase prediction in the STFT domain, and
the strongest log-spectral-distance accuracy of these engines. The packaged checkpoint is
the 12 kHz→48 kHz model.

## sidon

Sidon (SARULab-Speech): full speech *restoration*. An 8-layer w2v-BERT 2.0 feature
predictor, LoRA-adapted to denoise SSL representations, feeds a DAC vocoder that
resynthesises clean 48 kHz audio from degraded 16 kHz input. The SeamlessM4T log-mel
front-end runs in numpy; the feature extractor ships int8-quantized.

At roughly 0.6× realtime on CPU it suits offline dataset work rather than interactive use.

## callenhancer

CallEnhancer (Scicom-intl): call-centre and telephony restoration. The same two-graph
shape as sidon, but with the *full* 24-layer w2v-BERT 2.0 feature predictor (LoRA merged
into the base weights) driving a 188M-parameter DAC vocoder, trained on narrowband,
codec'd call audio (G.711/GSM, 8–16 kHz).

```python
sr = load_sr("callenhancer")                     # fp32 feature extractor
sr = load_sr("callenhancer", precision="int8")   # ~4x smaller, audibly lossy
sr = load_sr("callenhancer", chunk_seconds=60)   # window very long calls
```

- `precision` — `"fp32"` (default) or `"int8"`. See [Precision](#precision).
- `chunk_seconds` — `0` (default) runs one length-invariant pass. A positive value windows
  the audio with a 2 s crossfaded overlap, trading a little seam overhead for much lower
  peak memory on long calls, since attention is O(T²).

The ONNX weights are **CC-BY-NC-4.0** — that covers the model, not audio restored with it.

## Precision

Both transformer engines can ship an int8-quantized feature extractor. Quantization cost
scales with encoder depth, so it is a per-engine decision:

| Engine | fp32 | int8 | int8 vs fp32 | Default |
|--------|------|------|--------------|---------|
| sidon (8-layer) | — | ~410 MB | near-lossless | int8 |
| callenhancer (24-layer) | ~2.3 GB | ~580 MB | corr 0.969, ~12 dB SNR | **fp32** |

Per-layer rounding error accumulates with depth. Sidon's shallow encoder quantizes
cleanly; CallEnhancer's 24-layer encoder loses about 12 dB SNR end-to-end, which is
audible. CallEnhancer therefore defaults to fp32, whose weights ride in an external
`.onnx.data` sidecar fetched automatically alongside the graph.

Decoders always ship fp32 — the DAC vocoder is small and quantizes poorly.

## Attribution

lavasr and novasr derive from [LavaSR](https://github.com/ysharma3501/LavaSR) and
[NovaSR](https://github.com/ysharma3501/NovaSR) by Yatharth Sharma (Apache-2.0);
hifiganbwe from [brentspell/hifi-gan-bwe](https://github.com/brentspell/hifi-gan-bwe)
(MIT); apbwe from [yxlu-0102/AP-BWE](https://github.com/yxlu-0102/AP-BWE) (MIT); sidon
from [sarulab-speech/Sidon](https://github.com/sarulab-speech/Sidon) (MIT); callenhancer
from [Scicom-intl/CallEnhancer](https://huggingface.co/Scicom-intl/CallEnhancer)
(CC-BY-NC-4.0).
