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
| [cmgan](#cmgan) | 16 kHz | 7.8 MB | MIT | conformer metric-GAN; PESQ over waveform fidelity |
| [metadenoiser](#metadenoiser) | 16 kHz | 19–34 MB | **CC-BY-NC-4.0** | time-domain Demucs; non-commercial weights |
| [mossformergan](#mossformergan) | 16 kHz | 17.7 MB | Apache-2.0 | highest published PESQ (3.47) |
| [voicefixer](#voicefixer) | 44.1 kHz | 415 MB | MIT | general *restoration*, not denoising alone |
| [voiceisolator](#voiceisolator) | 16 kHz | 28–89 MB | **other** | keeps one speaker, removes other voices; streaming |
| [deepfilternet](#deepfilternet) | 48 kHz | ~2 MB | MIT | needs the `deepfilternet` extra |

## Recommendations

If you only read one line: **use `dpdfnet`**. It needs no optional dependencies, covers
8/16/48 kHz, and is among the strongest measured here.

| If you want… | Use | Why |
|---|---|---|
| a sensible default | **dpdfnet** | no extra deps, all three rates, top-tier measured gain |
| the best fullband quality | **mossformer2** | strongest 48 kHz; keeps content above 8 kHz |
| the best benchmark scores | **mossformergan** | highest published PESQ (3.47) |
| the smallest footprint | **gtcrn** | 0.54 MB, 23.7 K params |
| a small *spectral* model | **mpsenet** | 9.7 MB, predicts magnitude and phase in parallel |
| a time-domain model | **metadenoiser** | the only waveform-domain option — fails differently |
| several defects at once | **voicefixer** | restoration, not denoising: noise + reverb + clipping + band loss |
| to reproduce published results | **frcrn**, **cmgan**, **mpsenet (`vb`)** | these are the checkpoints those numbers came from |
| a recording damaged several ways at once | **voicefixer** | restores noise, reverb, clipping and band loss together |
| to remove other voices, not only noise | **voiceisolator** | target-speaker extraction, enrolled live or from a clip |

Engines are kept even when something else beats them, so a result can be reproduced or an
architecture compared. `cmgan` is the clearest case: it is dominated on both PESQ and SNR by
`gtcrn` at a fourteenth of the size, and it stays because a conformer metric-GAN is a
distinct thing to be able to run.

**Licensing is yours to judge.** `metadenoiser` is CC-BY-NC-4.0. `voiceisolator` declares
"other" upstream and publishes no license text. Everything else here is MIT or Apache-2.0. The restriction covers the weights, not audio processed with them.
`audiosronnx list` reports the license of every engine.

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
| cmgan | −7.7 dB | −1.5 dB | +3.5 dB |
| metadenoiser (`dns64`) | +3.2 dB | +6.9 dB | +9.3 dB |
| mossformergan | +2.6 dB | +8.3 dB | +11.6 dB |
| voiceisolator | +0.7 dB | +5.8 dB | +7.8 dB |

`dpdfnet` and `mossformer2` are close, trading places either side of ~11 dB input.
`gtcrn` **saturates**: its gain stops scaling with noise level, which is the cost of
23.7 K parameters. `cmgan` *loses* SNR on lightly-corrupted input, which is not a fault —
see below.

SNR is the wrong yardstick for every engine. Measured at 11 dB input, with PESQ alongside:

| Engine | PESQ | ΔPESQ | ΔSNR |
|--------|------|-------|------|
| mpsenet | 2.64 | +1.47 | +8.9 dB |
| frcrn | 2.46 | +1.29 | +8.8 dB |
| mossformer2 | 2.41 | +0.94 | +10.4 dB |
| gtcrn | 2.28 | +1.11 | +7.3 dB |
| dpdfnet | 2.27 | +0.80 | +10.3 dB |
| cmgan | 1.93 | +0.76 | −1.5 dB |

`cmgan` is the clearest case of the two metrics disagreeing: it raises perceptual quality
while moving the waveform *away* from the reference.

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

## cmgan

CMGAN (Cao et al.): a conformer-based **metric GAN**. A dense encoder feeds four two-stage
conformer blocks attending along time and frequency in turn; two decoders emit a magnitude
mask plus a complex residual, so it corrects phase rather than reusing the noisy phase. At
1.83 M parameters it is among the smallest engines here, and the only conformer one.

Being a *metric* GAN it optimises PESQ, and that objective diverges from waveform fidelity:
at 11 dB input it gains **+0.76 PESQ while losing 1.5 dB SNR**.

Note the trade honestly — on this material `gtcrn` scores higher on **both** metrics at a
fourteenth of the size. Reach for cmgan when you want a conformer/metric-GAN specifically,
or are reproducing its published results; otherwise start elsewhere.

## mossformergan

MossFormerGAN (Alibaba / ClearerVoice-Studio): the MossFormer attention backbone with a
metric-GAN objective, and the **highest published PESQ of any model surveyed for this
library — 3.47** on VoiceBank+DEMAND. Like `cmgan` it predicts a masked magnitude plus an
additive complex residual, so it corrects phase rather than reusing it — but unlike `cmgan`
it also gains on waveform SNR.

Note the gap between benchmark and measurement: it leads on PESQ, while `dpdfnet` and
`mossformer2` recover more SNR on the clip measured above. Both are true; pick on the axis
you care about.

MossFormer's group attention captures the traced sequence length in a reshape, so the graph
takes a fixed **401-frame** window (~2.5 s) and the adapter slides it with a crossfaded
overlap. Upstream's own decode segments long audio for the same reason.

## voicefixer

VoiceFixer (Liu et al.): general speech **restoration** at 44.1 kHz. Where every other
engine here targets one defect, this is trained to undo noise, reverberation, clipping and
bandwidth loss *together* — reach for it when a recording is damaged in several ways and you
do not want to chain three engines.

Two stages, the same shape as `sidon`: a ResUNet predicts a clean log-mel, and a TFGAN
vocoder resynthesises the waveform. The STFT, the mel projection and the log inverse run in
numpy.

It is **generative**: it resynthesises speech rather than filtering it, so the detail it
restores is invented. Treat the output as enhanced-real — fine to listen to and reasonable
as an acoustic-model target, but not a faithful recovery of what was there. That also means
SNR against a clean reference is the wrong yardstick for it, which is why it is absent from
the table above.

At 415 MB across two graphs it is the heaviest engine in the library, and it works over a
fixed 5 s window that the adapter slides. Verified against upstream on real speech at
correlation 0.99999996.

## metadenoiser

Meta's *denoiser* (Defossez et al.): a **causal Demucs** working directly on the waveform —
a convolutional encoder/decoder around an LSTM bottleneck, with no spectral front-end. It
is the only time-domain denoiser here; every other one masks or predicts a spectrum, which
makes it a genuinely different failure mode to have available.

```python
load_denoise("metadenoiser")                  # dns64, 33.5M params
load_denoise("metadenoiser", model="dns48")   # 18.9M, roughly half the size
```

**Its weights are CC-BY-NC-4.0** — research and non-commercial use only. That covers the
model, not audio processed with it. Every other denoiser here is MIT or Apache-2.0, so
choose this one deliberately.

Demucs bakes its padding arithmetic into the trace, so the graphs take a fixed **10 s
window** and the adapter slides that window with a crossfaded overlap. Reproduces upstream
to correlation 1.00000000.

## voiceisolator

Scicom's voice isolator (Zolkepli, *The Voice on the Line*, arXiv:2610.35272) keeps **one
speaker** and removes every other voice as well as the noise. Every other engine here keeps
all speech, so it is the one to use when the problem is a second person, a TV or a meeting
room rather than hiss.

```python
load_denoise("voiceisolator")                          # enrolls live from the input
load_denoise("voiceisolator", enrollment="caller.wav") # locks onto the voice in a clip
load_denoise("voiceisolator", model="agent")           # adds the presence gate
load_denoise("voiceisolator", quantized=True)          # 8-bit graphs, 28 MB
```

The graphs are the CPU release: a 22 M-parameter distillation of the 555 M GPU model, a
frame transformer over the last 10 s plus frame and band LSTMs. It is causal, with 10 ms
hops and 10 ms of delay.

**Enrollment** decides whose voice survives. By default the speaker vector comes from the
loudest 3 s of the input heard so far. It is re-made every half second until 15.5 s, then
fixed. Before 3 s the model runs on its learned "no enrollment" vector. An `enrollment`
clip of the speaker alone replaces the live rule.

**The `agent` model** adds a presence head. Where the speaker is not talking, the output is
exact digital silence, so a downstream VAD fires less on other voices. `gate="off"` turns
it off, and a `"THETA:HOLD:THETA_OFF:KEEP_DB"` string sets it by hand.

**Streaming.** `stream()` returns a stream you `push()` 16 kHz audio into, any amount at a
time. Output lags input by 160 samples, and `flush()` returns the tail:

```python
iso = load_denoise("voiceisolator")
s = iso.stream()
for chunk in mic_chunks:          # 16 kHz float32
    play(s.push(chunk))
play(s.flush())
```

The numpy side reproduces upstream's reference runtime bit for bit, for both models in fp32
and int8; the tests check this on 22 s of audio. Upstream's voicebot greeting mode is not included.

**The 555 M model.** The downloaded graphs are the CPU release. The full GPU release
(`Scicom-intl/voice-isolator`, WavLM-Large made causal beside a wider transformer) is not
hosted; export it with `conversion/export_voiceisolator.py` and load the directory:

```python
load_denoise("voiceisolator", model_dir="out/voiceisolator-large")
load_denoise("voiceisolator", model_dir="out/voiceisolator-large-agent")  # with the gate
```

Its step graph is 2.2 GB and takes a fixed 50 frames, so a stream emits output in 0.5 s
blocks. On 8 CPU threads it takes about 2 s per second of audio; it was built for a GPU.
Both models match upstream's PyTorch streaming step to 1.2e-7 or better, with and without
the gate, past the point where the 10 s attention rings wrap.

Two measured cautions. On broadband noise alone it gains less than `dpdfnet` (table above),
because removing voices is its job. On a 0 dB mix of a fullband voice and a telephone-band
voice, it kept the fullband voice even when enrolled on the telephone-band one. The
enrollment still moved the result by 17 dB toward the enrolled voice.

**License:** the upstream model card declares `license: other` and publishes no license
text. The weights download from the upstream repository at a pinned revision and are not
mirrored. Check the terms with the authors before you use it beyond evaluation.

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
