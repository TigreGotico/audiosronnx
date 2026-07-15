# audiosronnx

Pure-ONNX, multi-engine **audio super-resolution / bandwidth extension**. Upscale
low-sample-rate or low-bandwidth speech — 8 kHz telephony, muffled recordings,
low-fidelity TTS — to a clean **48 kHz** signal. Inference runs entirely on
onnxruntime; there is **no torch at runtime**.

audiosronnx is the super-resolution sibling of the TigreGotico pure-ONNX speech
libraries ([`vadonnx`](https://github.com/TigreGotico/vadonnx),
[`voiceclonnx`](https://github.com/TigreGotico/voiceclonnx),
[`speakeronnx`](https://github.com/TigreGotico/speakeronnx)) and mirrors their
conventions: a single `load_sr(engine=...)` entry point, an engine registry, a
HuggingFace-backed model resolver with an XDG cache, and a small CLI.

## Install

```bash
pip install audiosronnx
```

Runtime dependencies: `numpy`, `onnxruntime`, `scipy`, `soundfile`, `huggingface_hub`.
ONNX models are downloaded on first use from the Hugging Face Hub and cached under
`~/.local/share/audiosronnx`.

## Engines

| Engine | Input SR | Output SR | Size | CPU speed | License | Status |
|--------|----------|-----------|------|-----------|---------|--------|
| **lavasr** (default) | 8–48 kHz | 48 kHz | ~52 MB | ~50× realtime | Apache-2.0 | shipped |
| **novasr** | 16 kHz | 48 kHz | ~0.2 MB | ~1000× realtime | Apache-2.0 | shipped |
| **hifiganbwe** | any | 48 kHz | ~4 MB | fast | MIT | shipped |
| **apbwe** | any (12 kHz band) | 48 kHz | ~120 MB | moderate | MIT | shipped |

- **lavasr** — a Vocos-based bandwidth-extension model with a Linkwitz-Riley spectral
  merge that preserves the original low band, plus an optional UL-UNAS denoiser. It
  accepts any input rate from 8 to 48 kHz. All spectral DSP (STFT, ISTFT, mel
  filterbank, resampling, merge) runs in numpy/scipy, so no `torch.stft` ever enters an
  ONNX graph.
- **novasr** — a tiny conv1d / BigVGAN-snake generator that upsamples 16 kHz to 48 kHz
  in a single time-domain pass. Extremely fast and memory-light, at lower fidelity than
  lavasr; useful for on-device enhancement and quick dataset restoration.
- **hifiganbwe** — HiFi-GAN+ (Su et al., ICASSP 2021): bandlimited (kaiser)
  interpolation to 48 kHz followed by a non-causal WaveNet that reconstructs the high
  band. Time-domain, ~1M params, no spectral front-end. Accepts any input rate.
- **apbwe** — AP-BWE (Lu et al.): dual-ConvNeXt amplitude-and-phase prediction in the
  STFT domain; the strongest log-spectral-distance accuracy of the shipped engines. The
  packaged checkpoint is the 12 kHz→48 kHz model.

lavasr/novasr derive from the LavaSR/NovaSR projects by Yatharth Sharma
([LavaSR](https://github.com/ysharma3501/LavaSR),
[NovaSR](https://github.com/ysharma3501/NovaSR), Apache-2.0); hifiganbwe from
[brentspell/hifi-gan-bwe](https://github.com/brentspell/hifi-gan-bwe) (MIT); apbwe from
[yxlu-0102/AP-BWE](https://github.com/yxlu-0102/AP-BWE) (MIT). Every neural component
runs through onnxruntime with all STFT/ISTFT/resampling kept in numpy/scipy — the ONNX
graphs are validated against the original PyTorch models (HiFi-GAN+ end-to-end
correlation 1.0000, AP-BWE 0.9998, per-graph max error ≤ 5e-4).

## Quickstart

```python
from audiosronnx import load_sr

sr = load_sr("lavasr")                      # default engine

# low-level primitive: numpy array or path in, (float32 mono, 48000) out
out, rate = sr.upscale("telephone_8k.wav")  # -> (np.ndarray float32, 48000)

# array input (you supply the sample rate)
import soundfile as sf
audio, in_sr = sf.read("input.wav")
out, rate = sr.upscale(audio, in_sr)

# file / directory helpers
sr.upscale_file("in.wav", "out_48k.wav")
sr.upscale_dir("clips_in/", "clips_out_48k/")

# the fast tiny engine instead
fast = load_sr("novasr")
fast.upscale_file("in_16k.wav", "out_48k.wav")
```

LavaSR options:

```python
sr = load_sr("lavasr", denoise=True, cutoff_hz=6000)
```

## CLI

```bash
audiosronnx list                              # list engines
audiosronnx probe input.wav                   # print format / duration
audiosronnx upscale in.wav out.wav            # default engine (lavasr)
audiosronnx upscale in.wav out.wav --engine novasr
audiosronnx upscale in.wav out.wav --denoise  # lavasr denoiser stage
audiosronnx upscale-dir clips_in/ clips_out/
```

## API

- `load_sr(engine="lavasr", *, providers=None, cache_dir=None, revision=None, **kwargs) -> SRModel`
- `SRModel.upscale(audio, sample_rate=None) -> (np.ndarray float32 mono, 48000)` —
  `audio` is a path, raw int16 PCM bytes, or a numpy array. The low-level primitive.
- `SRModel.upscale_file(in_path, out_path) -> str`
- `SRModel.upscale_dir(in_dir, out_dir) -> list[str]`
- `available_models()`, `get_engine(alias)`, `register_engine(entry)`, `ENGINE_REGISTRY`

### Custom / third-party engines

Subclass `SRModel`, implement `_upscale_array(audio, sample_rate) -> np.ndarray` (a
48 kHz mono float32 array), and register an `EngineEntry`:

```python
from audiosronnx import SRModel, EngineEntry, register_engine

class MySR(SRModel):
    output_sample_rate = 48000
    def _upscale_array(self, audio, sample_rate):
        ...  # run your ONNX session, return a 48 kHz float32 array

register_engine(EngineEntry(alias="mysr", adapter_class=MySR, license="MIT"))
```

## Model hosting

Exported ONNX weights are hosted on the Hugging Face Hub and pinned by revision:

- `TigreGotico/audiosronnx-lavasr` — `backbone.onnx`, `spec_head.onnx`, `denoiser_core.onnx`
- `TigreGotico/audiosronnx-novasr` — `novasr.onnx`
- `TigreGotico/audiosronnx-hifiganbwe` — `hifiganbwe_wavenet.onnx`
- `TigreGotico/audiosronnx-apbwe` — `apbwe.onnx`

The export scripts under `conversion/` reproduce these from the upstream PyTorch
checkpoints and validate each ONNX graph against its PyTorch submodule (max absolute
error).

## Evaluated but not shipped

The audio super-resolution landscape was surveyed for other models to export. These
were evaluated and are **not** shipped, for the reasons given:

| Model | License | Reason not shipped |
|-------|---------|--------------------|
| **FLowHigh** | MIT | Single-step flow matching, but depends on an external BigVGAN vocoder plus a mel/STFT front-end — a multi-component export rather than one clean graph. |
| **AudioSR** | MIT | ~6 GB latent-diffusion model (VAE + LDM + vocoder, iterative sampler, ~0.6× realtime on GPU). Not CPU-runnable at usable latency; impractical to export. |
| **resemble-enhance** | MIT | Four-network 44.1 kHz restoration pipeline: UNet denoiser + IRMAE autoencoder + a CFM iterative ODE sampler (32–64 velocity-net evals per utterance) + a UnivNet vocoder. The vocoder is LVCNet — a *location-variable* convolution whose kernels are predicted per location and applied via `einsum` over `unfold`s, so it does not fold into a static ONNX graph. Speech restoration/denoising, not single-pass bandwidth extension (its hparams hard-assert 44.1 kHz). Same diffusion-style, multi-graph, dynamic-op class as AudioSR. |
| **NU-Wave2** | none | Diffusion (iterative sampler) and the repository ships no license file. |
| **mdctGAN** | unclear (NOASSERTION) | Unclear license, and its MDCT front-end relies on `torch.fft`, which exports to ONNX unreliably. |
| **VoiceFixer / NVSR** | MIT | Speech *restoration* rather than pure bandwidth extension; two-stage mel-predictor + neural vocoder, heavier and less focused than the shipped engines. |

An engine is only shipped when it exports cleanly to ONNX, runs on CPU via
onnxruntime, carries a permissive license, and produces verified 48 kHz output.

## License

Apache-2.0. The shipped model weights are Apache-2.0 (LavaSR, NovaSR).
