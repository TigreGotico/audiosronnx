# Examples

Runnable scripts. Each takes real audio paths and needs only the base install unless noted.

| Script | Shows |
|--------|-------|
| [upscale.py](upscale.py) | Bandwidth extension to 48 kHz, array and file forms |
| [denoise.py](denoise.py) | Noise removal, and reading the returned sample rate |
| [denoise_then_upscale.py](denoise_then_upscale.py) | Chaining the two in the order that matters |
| [batch_folder.py](batch_folder.py) | Processing a directory, with progress |
| [compare_denoisers.py](compare_denoisers.py) | Scoring every denoiser on your own audio |
| [custom_engine.py](custom_engine.py) | Registering a third-party engine |

```bash
python examples/upscale.py input.wav out_48k.wav
python examples/denoise.py noisy.wav clean.wav --engine mossformer2
python examples/compare_denoisers.py clean_reference.wav
```

Weights download on first use and cache under `~/.local/share/audiosronnx`, so the first
run of an engine is slower than the rest.
