# audiosronnx

Speech **restoration, denoising and bandwidth extension** in pure ONNX. Clean up noisy
recordings, and upscale narrowband speech — 8 kHz telephony, muffled captures,
low-fidelity TTS — to a full **48 kHz** signal.

Inference runs entirely on onnxruntime. **There is no torch at runtime**: every STFT,
filterbank and resampler lives in numpy/scipy, so the ONNX graphs stay portable and the
front-ends stay inspectable.

```bash
pip install audiosronnx
```

```python
from audiosronnx import load_denoise, load_sr

clean, rate = load_denoise("dpdfnet").denoise("noisy_call.wav")   # remove noise
wide,  _    = load_sr("lavasr").upscale(clean, rate)              # extend to 48 kHz
```

```bash
audiosronnx denoise noisy.wav clean.wav
audiosronnx upscale clean.wav wide_48k.wav
```

Runtime dependencies are `numpy`, `onnxruntime`, `scipy`, `soundfile` and
`huggingface_hub`. Weights download on first use and cache under
`~/.local/share/audiosronnx`, pinned by revision.

## Two jobs

**Denoising** removes background noise and keeps the sample rate. **Bandwidth extension**
invents high-frequency content that was never in the input and always outputs 48 kHz. They
have separate entry points — `load_denoise()` and `load_sr()` — which refuse each other's
engines.

They compose, in that order: denoise first, then extend. Run them the other way round and
the extender reconstructs a high band out of the noise.

## Denoisers

| Engine | Rate | Size | License | Reach for it when |
|--------|------|------|---------|-------------------|
| **dpdfnet** (default) | 8 / 16 / **48 kHz** | 8.7–14.9 MB | Apache-2.0 | you have no other reason to choose |
| **mossformer2** | **48 kHz** | 229 MB | Apache-2.0 | the source is fullband and quality wins |
| **frcrn** | 16 kHz | 57.5 MB | Apache-2.0 | reproducing published benchmarks |
| **mpsenet** | 16 kHz | 9.7 MB | MIT | you want a spectral model that stays small |
| **gtcrn** | 16 kHz | **0.54 MB** | MIT | footprint is the binding constraint |
| **cmgan** | 16 kHz | 7.8 MB | MIT | you want a conformer metric-GAN specifically |
| **metadenoiser** | 16 kHz | 19–34 MB | **CC-BY-NC-4.0** | you want a time-domain model and NC is acceptable |
| **mossformergan** | 16 kHz | 17.7 MB | Apache-2.0 | highest published PESQ (3.47) |
| **voicefixer** | 44.1 kHz | 415 MB | MIT | *restoration* — noise, reverb, clipping and band loss at once |
| **deepfilternet** | 48 kHz | ~2 MB | MIT | you already depend on `libdf` |

SNR recovered on one clip at 19 / 11 / 5 dB input SNR: dpdfnet **+4.9 / +10.3 / +13.7 dB**,
mossformer2 +5.9 / +10.4 / +13.4, mpsenet +4.8 / +8.9 / +11.7, frcrn +4.6 / +8.8 / +11.7,
gtcrn +3.5 / +7.3 / +7.5. That is broadband Gaussian noise — a hostile synthetic case that
ranks engines consistently but predicts little about babble or codec artefacts.

**Recommendations:** start with **dpdfnet** — no extra dependencies, 8/16/48 kHz, top-tier
measured gain. Take **mossformer2** for the best fullband quality, **mossformergan** for the
best benchmark scores, **gtcrn** when footprint is the constraint, **metadenoiser** for a
time-domain model. Engines are kept even when something else beats them, so published
results stay reproducible — `cmgan` is dominated and stays for that reason.

Full detail in [docs/denoising.md](docs/denoising.md).

## Bandwidth extension

| Engine | Input | Size | CPU speed | License |
|--------|-------|------|-----------|---------|
| **lavasr** (default) | 8–48 kHz | ~52 MB | ~50× realtime | Apache-2.0 |
| **novasr** | 16 kHz | ~0.2 MB | ~1000× realtime | Apache-2.0 |
| **flowhigh** | any | ~200 MB | slow (4 ODE steps) | MIT |
| **hifiganbwe** | any | ~4 MB | fast | MIT |
| **apbwe** | any (12 kHz band) | ~120 MB | moderate | MIT |
| **sidon** | 16 kHz | ~410 MB | ~0.6× realtime | MIT |
| **callenhancer** | 8–16 kHz | ~3 GB / ~1.3 GB int8 | ~0.3–0.4× realtime | CC-BY-NC-4.0 |

`lavasr` is the general-purpose choice. `sidon` and `callenhancer` do something different:
they *resynthesise* speech rather than extend its band, repairing codec damage a bandwidth
extender cannot, at a much higher cost. `callenhancer` is trained on telephony
specifically, and its weights are non-commercial.

Full detail in [docs/engines.md](docs/engines.md).

## Choosing an engine

Three questions settle it.

**1. What is wrong with the audio?**

| Problem | Reach for |
|---------|-----------|
| background noise | a **denoiser** — `dpdfnet` |
| narrowband / muffled, but clean | a **bandwidth extender** — `lavasr` |
| noisy *and* narrowband | denoise then extend — `dpdfnet` → `lavasr` |
| telephony, codec-damaged | `callenhancer` (restoration, not extension) |
| damaged several ways at once | `voicefixer` (restoration) |

**2. What constrains you?**

| Constraint | Denoise | Extend |
|------------|---------|--------|
| nothing in particular | `dpdfnet` | `lavasr` |
| footprint | `gtcrn` (0.54 MB) | `novasr` (0.2 MB) |
| best quality, fullband | `mossformer2` | `apbwe` |
| best benchmark scores | `mossformergan` | — |
| license must be permissive | anything but `metadenoiser` | anything but `callenhancer` |

**3. Do you mind invented detail?**

`sidon`, `callenhancer`, `flowhigh` and `voicefixer` are **generative** — they resynthesise
speech rather than filter it, so the detail they add is plausible rather than recovered.
That is fine for listening and reasonable as an acoustic-model target; it is not a faithful
reconstruction, and SNR against a clean reference is the wrong yardstick for them.

Engines are kept even when something else beats them, so published results stay
reproducible and distinct architectures stay runnable. `cmgan` is the clearest example: it
is dominated on both PESQ and SNR by `gtcrn` at a fourteenth of the size, and it stays.

These answers are backed by a reproducible board: every engine is scored on a fixed,
externally-hosted gold set with the `speechonnxmetrics` library, the same files for all of
them. The measured tables live in [docs/denoising.md](docs/denoising.md) (denoising and
restoration) and [docs/engines.md](docs/engines.md) (bandwidth extension); regenerate them
with `python -m benchmarks.run` (see [benchmarks/](benchmarks/)).

## Documentation

| Page | Contents |
|------|----------|
| [docs/engines.md](docs/engines.md) | Every bandwidth-extension engine, options, precision |
| [docs/denoising.md](docs/denoising.md) | Every denoiser, and how to choose |
| [docs/api.md](docs/api.md) | `load_sr`, `load_denoise`, the model contracts, the registry |
| [docs/cli.md](docs/cli.md) | Command-line reference |
| [docs/custom-engines.md](docs/custom-engines.md) | Registering your own engine |
| [docs/conversion.md](docs/conversion.md) | How the ONNX graphs are produced and validated |
| [docs/not-shipped.md](docs/not-shipped.md) | Models evaluated and rejected, with reasons |

Runnable scripts are in [examples/](examples).

## API

```python
sr = load_sr("lavasr")
out, rate = sr.upscale("telephone_8k.wav")   # path, int16 bytes, or numpy array
sr.upscale_file("in.wav", "out_48k.wav")
sr.upscale_dir("clips_in/", "clips_out/")

dn = load_denoise("dpdfnet")
clean, rate = dn.denoise(audio, in_rate)     # returns the ENGINE's rate
dn.denoise_file("in.wav", "clean.wav")
dn.denoise_dir("clips_in/", "clips_out/")

available_models(); available_denoisers(); get_engine(alias); register_engine(entry)
```

Output is always mono float32. Empty, silent and non-finite input produce finite output
rather than raising. See [docs/api.md](docs/api.md).

## Which engines ship

An engine ships when it exports to a **single static ONNX graph**, runs on CPU through
onnxruntime, carries a permissive license, and is **validated end-to-end against the
upstream pipeline** — not merely against the model, since reproducing a network but not
its surrounding normalisation yields audio that sounds plausible and is quietly wrong.

That rules out diffusion and flow-matching models, location-variable convolutions, and
unlicensed checkpoints. [docs/not-shipped.md](docs/not-shipped.md) names every candidate
considered and which condition it failed.

## License

Apache-2.0.

Model weights carry their upstream licenses, listed per engine in the docs and reported by
`audiosronnx list`. Most are MIT or Apache-2.0; **`callenhancer` is CC-BY-NC-4.0**, which
covers the weights rather than audio processed with them.
