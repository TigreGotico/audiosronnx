# Denoising

A denoiser removes background noise and **preserves the sample rate**. That makes it a
different job from bandwidth extension, and it lives behind its own entry point:

```python
from audiosronnx import load_denoise

dn = load_denoise("dpdfnet")
clean, rate = dn.denoise("noisy_call.wav")

dn.denoise_file("in.wav", "clean.wav")
dn.denoise_dir("clips_in/", "clips_out/")
```

`load_sr()` and `load_denoise()` reject each other's engines, so a bandwidth extender
cannot be loaded as a denoiser by accident.

| Engine | Rate | Size | License | Notes |
|--------|------|------|---------|-------|
| [dpdfnet](#dpdfnet) (default) | 8 / 16 / **48 kHz** | 8.7–14.9 MB | Apache-2.0 | streaming, 8 variants, no extra deps |
| [mossformer2](#mossformer2) | **48 kHz** | 229 MB | Apache-2.0 | strongest fullband |
| [frcrn](#frcrn) | 16 kHz | 57.5 MB | Apache-2.0 | highest published benchmark scores |
| [mpsenet](#mpsenet) | 16 kHz | 9.7 MB | MIT | parallel magnitude + phase |
| [gtcrn](#gtcrn) | 16 kHz | **0.54 MB** | MIT | ultra-light, on-device |
| [deepfilternet](#deepfilternet) | 48 kHz | ~2 MB | MIT | needs the `deepfilternet` extra |

## Choosing

**Start with `dpdfnet`.** It is the default, needs no optional dependencies, covers 8, 16
and 48 kHz, and is among the strongest measured here.

Then adjust:

- **Fullband source (48 kHz) and quality matters most** — `mossformer2`. It is the only
  engine besides dpdfnet and deepfilternet that keeps content above 8 kHz, and the
  strongest of those.
- **Footprint is the constraint** — `gtcrn` at 0.54 MB, or `mpsenet` at 9.7 MB if you want
  a full spectral model.
- **Reproducing published benchmarks** — `frcrn` (PESQ 3.23 on VoiceBank+DEMAND).
- **Already depend on `libdf`** — `deepfilternet` is the smallest 48 kHz option.

### Measured performance

SNR gain on one speech clip corrupted with broadband Gaussian noise, measured at three
input SNRs against a fixed noise seed:

| Engine | 19.1 dB in | 11.1 dB in | 5.1 dB in |
|--------|-----------|------------|-----------|
| dpdfnet | +4.9 dB | **+10.3 dB** | **+13.7 dB** |
| mossformer2 | **+5.9 dB** | +10.4 dB | +13.4 dB |
| mpsenet (`dns`) | +4.8 dB | +8.9 dB | +11.7 dB |
| frcrn | +4.6 dB | +8.8 dB | +11.7 dB |
| gtcrn | +3.5 dB | +7.3 dB | +7.5 dB |

`dpdfnet` and `mossformer2` are close, trading places either side of ~11 dB input.
`gtcrn` **saturates**: its gain stops scaling with noise level, which is the cost of
23.7 K parameters.

Read these as one operating point, not a ranking. Broadband Gaussian noise is a
deliberately hostile synthetic case and says little about speech babble, room noise or
codec artefacts, and the published PESQ ordering disagrees with it — `frcrn` leads the
benchmarks while measuring mid-pack here. Test on your own material.

## dpdfnet

DPDFNet (Ceva), *Dual-Path RNN-based DeepFilterNet*, built on DeepFilterNet2. The whole
enhancement stage is one **stateful** ONNX graph consuming a single STFT frame at a time,
so only a Vorbis-windowed STFT/ISTFT runs outside it, in numpy. It needs no `libdf` wheel,
and is the only engine here with 8 kHz and 16 kHz variants alongside 48 kHz.

```python
load_denoise("dpdfnet", model="dpdfnet8")        # 16 kHz, highest quality
load_denoise("dpdfnet", model="dpdfnet2_8khz")   # narrowband telephony
load_denoise("dpdfnet", attn_limit_db=12)        # keep a natural noise floor
```

| `model` | Rate | Notes |
|---------|------|-------|
| `dpdfnet2_48khz` (default) | 48 kHz | balanced, fullband |
| `dpdfnet8_48khz` | 48 kHz | highest quality, ~1.5× the compute |
| `baseline` | 16 kHz | fastest, lowest compute |
| `dpdfnet2` / `dpdfnet4` / `dpdfnet8` | 16 kHz | increasing quality |
| `dpdfnet2_8khz` / `dpdfnet8_8khz` | 8 kHz | narrowband |

`attn_limit_db` caps how much the model may attenuate, blending the noisy spectrum back
in. Aggressive gating can sound unnatural on speech pauses; a finite value such as 12
leaves a residual floor.

The adapter reproduces the upstream reference implementation to max abs err **1.7e-8**.

## mossformer2

MossFormer2 (Alibaba / ClearerVoice-Studio): a 55M-parameter hybrid transformer and
recurrent mask predictor, and the strongest **fullband** engine here (PESQ 3.16 / STOI 0.95
/ SI-SDR 19.38 on VoiceBank+DEMAND). The graph predicts a mask from 60 Kaldi mel bins plus
their first and second deltas; that front-end, the mask application and the STFT/ISTFT all
run in numpy.

```python
load_denoise("mossformer2", dither=1.0)   # match upstream's stochastic front-end
```

`dither` defaults to `0` so the engine is deterministic. Upstream uses `1.0`, which moves
the output by roughly 1e-5.

At 229 MB it is the heaviest denoiser here. Matches the upstream pipeline to correlation
0.99999999.

## frcrn

FRCRN (Alibaba / ClearerVoice-Studio): a complex-mask denoiser built from two stacked UNets
with frequency-recurrent layers. It holds the strongest published benchmark scores here —
**PESQ 3.23 / STOI 0.95 / SI-SDR 19.22** on VoiceBank+DEMAND, and second place in the 2022
DNS Challenge.

Unlike the others it is not a streaming graph: its `ConvSTFT`/`ConviSTFT` are
Fourier-kernel convolutions, so the whole model is a single waveform-to-waveform graph.
Matches the upstream pipeline to correlation 0.99999994.

## mpsenet

MP-SENet (Lu et al., same authors as `apbwe`): predicts the **magnitude and phase spectra
in parallel**, rather than masking magnitudes and reusing the noisy phase. At 2.26M
parameters it is the smallest full spectral model here.

```python
load_denoise("mpsenet")               # dns checkpoint (default)
load_denoise("mpsenet", model="vb")   # VoiceBank+DEMAND checkpoint
```

The two upstream checkpoints are **not** interchangeable. On broadband noise the DNS
Challenge checkpoint recovers **+4.8 / +8.8 / +11.7 dB** where the VoiceBank+DEMAND one
manages only +1.5 / +2.0 / +4.1 dB — it generalises poorly beyond its training noise. Use
`vb` only to reproduce the published VoiceBank PESQ figures.

## gtcrn

GTCRN (Rong et al.): an ultra-light 16 kHz denoiser at **23.7 K parameters and 33 MMACs/s**
— half a megabyte, two orders of magnitude smaller than the rest. A single stateful graph
threading three recurrent caches per frame, with the ERB filterbank and subband features
inside the graph.

It denoises less aggressively than dpdfnet. The trade is size, not quality parity; reach
for it when footprint is the binding constraint.

## deepfilternet

DeepFilterNet3 (Schröter et al.): an ERB-band gain mask followed by *deep filtering* — a
short complex FIR filter per low-frequency bin across neighbouring frames, recovering
detail a real-valued mask cannot.

It runs as three ONNX graphs, and its STFT/ERB analysis and synthesis come from the `libdf`
Rust wheel:

```bash
pip install 'audiosronnx[deepfilternet]'
```

```python
load_denoise("deepfilternet", mask_only=True)   # skip deep filtering
```

## Chaining with bandwidth extension

Denoise **before** upscaling. A bandwidth extender asked to work from noisy input will
reconstruct a high band from the noise:

```python
clean, rate = load_denoise("dpdfnet").denoise("noisy_8k.wav")
wide, _ = load_sr("lavasr").upscale(clean, rate)
```

The restoration engines (`sidon`, `callenhancer`) already denoise as part of resynthesis,
so they do not need a separate denoise pass.
