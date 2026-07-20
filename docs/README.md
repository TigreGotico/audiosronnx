# audiosronnx documentation

| Page | Contents |
|------|----------|
| [engines.md](engines.md) | Every bandwidth-extension engine: architecture, rates, size, speed, options |
| [denoising.md](denoising.md) | Every denoise engine, and how to pick one |
| [api.md](api.md) | `load_sr`, `load_denoise`, the `SRModel` / `Denoiser` contracts, the registry |
| [cli.md](cli.md) | Command-line reference |
| [custom-engines.md](custom-engines.md) | Registering your own engine |
| [conversion.md](conversion.md) | How the ONNX graphs are produced and validated |
| [not-shipped.md](not-shipped.md) | Models that were evaluated and rejected, with reasons |

Runnable scripts live in [`../examples/`](../examples).

## Which do I want?

Two different jobs, two different entry points:

- **Bandwidth extension / super-resolution** (`load_sr`) invents high-frequency content
  that is not in the input and always outputs 48 kHz. Use it on narrowband sources —
  8 kHz telephony, 16 kHz recordings, low-fidelity TTS.
- **Denoising** (`load_denoise`) removes background noise and keeps the sample rate. Use
  it when the band is fine but the recording is dirty.

They compose: denoise first, then upscale. Doing it the other way round asks the
bandwidth extender to invent detail from noise, and it will happily do so.

```python
from audiosronnx import load_denoise, load_sr

clean, rate = load_denoise("dpdfnet").denoise("noisy_8k.wav")
wide, _ = load_sr("lavasr").upscale(clean, rate)
```

See [`examples/denoise_then_upscale.py`](../examples/denoise_then_upscale.py).
