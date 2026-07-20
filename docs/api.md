# API reference

Two entry points, one per job. Both return an object with array, file and directory
methods; the array form is the primitive and the others are built on it.

## Bandwidth extension

```python
load_sr(engine="lavasr", *, providers=None, cache_dir=None, revision=None, **engine_kwargs) -> SRModel
```

| Argument | Meaning |
|----------|---------|
| `engine` | registry alias — see [engines.md](engines.md) |
| `providers` | onnxruntime execution providers (default `["CPUExecutionProvider"]`) |
| `cache_dir` | override the model download cache |
| `revision` | pin a different HuggingFace revision |
| `**engine_kwargs` | forwarded to the adapter, e.g. `denoise=True`, `precision="int8"` |

### `SRModel`

| Method | Returns |
|--------|---------|
| `upscale(audio, sample_rate=None)` | `(np.ndarray float32 mono, 48000)` |
| `upscale_file(in_path, out_path)` | `out_path` |
| `upscale_dir(in_dir, out_dir, *, pattern=None)` | list of written paths |
| `sample_rate` | output rate (always 48000) |

`audio` is a path, raw `int16` PCM bytes, or a numpy array. `sample_rate` is required for
arrays and bytes, and ignored for paths.

## Denoising

```python
load_denoise(engine="dpdfnet", *, providers=None, cache_dir=None, revision=None, **engine_kwargs) -> Denoiser
```

Same arguments. Passing a bandwidth-extension alias raises `ValueError`.

### `Denoiser`

| Method | Returns |
|--------|---------|
| `denoise(audio, sample_rate=None)` | `(np.ndarray float32 mono, rate)` |
| `denoise_file(in_path, out_path)` | `out_path` |
| `denoise_dir(in_dir, out_dir, *, pattern=None)` | list of written paths |
| `sample_rate` | the engine's native rate |

A denoiser returns audio at **its own native rate**, not the input rate. Input at another
rate is resampled in. Check the returned rate rather than assuming it round-trips:

```python
clean, rate = load_denoise("gtcrn").denoise("input_48k.wav")   # rate == 16000
```

## Registry

```python
available_models()      # every engine alias
available_denoisers()   # denoise/enhance aliases only
get_engine(alias)       # -> EngineEntry
register_engine(entry)  # add your own
ENGINE_REGISTRY         # dict of alias -> EngineEntry
```

`EngineEntry` carries `alias`, `adapter_class`, `description`, `input_sample_rate`,
`output_sample_rate`, `license`, `extras`, and `kind` (`"sr"`, `"denoise"` or `"enhance"`).
`kind` is what `load_sr` and `load_denoise` filter on.

## Behaviour worth knowing

- **Output is always mono float32.** Stereo input is averaged down.
- **Empty input returns empty output**, not an error.
- **Length is preserved** by denoisers, and scaled by `48000/input_rate` by SR engines.
- **Silent or non-finite input yields finite output.** Engines that normalise by RMS or
  peak short-circuit rather than dividing by zero.
- **Models download on first use** to `~/.local/share/audiosronnx` (or `$XDG_DATA_HOME`),
  pinned by revision.
- **Sessions are created lazily**, on the first call rather than at construction.
