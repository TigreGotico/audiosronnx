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
| **novasr** | 16 kHz | 48 kHz | ~52 KB | ~1000× realtime | Apache-2.0 | shipped |

- **lavasr** — a Vocos-based bandwidth-extension model with a Linkwitz-Riley spectral
  merge that preserves the original low band, plus an optional UL-UNAS denoiser. It
  accepts any input rate from 8 to 48 kHz and is the best-quality engine. All spectral
  DSP (STFT, ISTFT, mel filterbank, resampling, merge) runs in numpy/scipy, so no
  `torch.stft` ever enters an ONNX graph.
- **novasr** — a tiny conv1d / BigVGAN-snake generator that upsamples 16 kHz to 48 kHz
  in a single time-domain pass. Extremely fast and memory-light, at lower fidelity than
  lavasr; useful for on-device enhancement and quick dataset restoration.

Both engines are derived from the LavaSR/NovaSR projects by Yatharth Sharma
([LavaSR](https://github.com/ysharma3501/LavaSR),
[NovaSR](https://github.com/ysharma3501/NovaSR)), Apache-2.0.

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

The export scripts under `conversion/` reproduce these from the upstream PyTorch
checkpoints and validate each ONNX graph against its PyTorch submodule (max absolute
error).

## Evaluated but not shipped

The audio super-resolution landscape was surveyed for other models to export. These
were evaluated and are **not** shipped, for the reasons given:

| Model | License | Reason not shipped |
|-------|---------|--------------------|
| **AP-BWE** | MIT | Permissive and ONNX-friendly (convolutional amplitude/phase predictor; STFT stays outside the graph). A strong candidate not yet packaged here. |
| **HiFi-GAN-BWE** | MIT | Permissive, time-domain, ships its own export script. A strong candidate not yet packaged here. |
| **FLowHigh** | MIT | Single-step flow matching, but depends on an external BigVGAN vocoder plus a mel/STFT front-end — a multi-component export rather than one clean graph. |
| **AudioSR** | MIT | ~6 GB latent-diffusion model (VAE + LDM + vocoder, iterative sampler, ~0.6× realtime on GPU). Not CPU-runnable at usable latency; impractical to export. |
| **NU-Wave2** | none | Diffusion (iterative sampler) and the repository ships no license file. |
| **mdctGAN** | unclear (NOASSERTION) | Unclear license, and its MDCT front-end relies on `torch.fft`, which exports to ONNX unreliably. |
| **VoiceFixer / NVSR** | MIT | Speech *restoration* rather than pure bandwidth extension; two-stage mel-predictor + neural vocoder, heavier and less focused than the shipped engines. |

An engine is only shipped when it exports cleanly to ONNX, runs on CPU via
onnxruntime, carries a permissive license, and produces verified 48 kHz output.

## License

Apache-2.0. The shipped model weights are Apache-2.0 (LavaSR, NovaSR).
