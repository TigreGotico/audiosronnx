# Custom engines

The registry is public. A third-party engine is a subclass plus an `EngineEntry`, and it
then works with `load_sr`/`load_denoise`, the CLI, and everything else.

## A bandwidth extender

Implement `_upscale_array`: mono float32 in at a known rate, 48 kHz mono float32 out. The
base class handles path/bytes/array dispatch, file and directory helpers.

```python
import numpy as np
from audiosronnx import EngineEntry, SRModel, register_engine


class MySR(SRModel):
    input_sample_rate = 16000     # 0 means "accepts any rate"
    output_sample_rate = 48000

    def __init__(self, *, providers=None, cache_dir=None, revision=None, **cfg):
        super().__init__(**cfg)
        self._sess = None

    def _ensure_models(self):
        if self._sess is None:
            import onnxruntime as ort
            self._sess = ort.InferenceSession(
                "my_model.onnx", providers=providers or ["CPUExecutionProvider"])

    def _upscale_array(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        self._ensure_models()
        out = self._sess.run(None, {"input": audio[None]})[0]
        return out.reshape(-1).astype(np.float32)


register_engine(EngineEntry(
    alias="mysr",
    adapter_class=MySR,
    description="what it does, in one line",
    input_sample_rate=16000,
    output_sample_rate=48000,
    license="MIT",
    kind="sr",
))
```

## A denoiser

Same shape, but subclass `Denoiser`, implement `_denoise_array`, and set `kind="denoise"`.
A denoiser returns audio at its own `input_sample_rate` rather than 48 kHz.

```python
from audiosronnx.base import Denoiser

class MyDenoiser(Denoiser):
    input_sample_rate = 16000

    def _denoise_array(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        ...
```

Register it with `output_sample_rate` equal to `input_sample_rate` — denoisers preserve the
rate, and that invariant is tested.

## Conventions worth following

These are what the shipped engines do, and what makes them predictable:

- **Keep spectral work outside the graph.** STFT, ISTFT, mel filterbanks and resampling
  belong in numpy/scipy — `audiosronnx._stft` and `audiosronnx._kaiser` provide
  torch-faithful implementations, and `_kaldi_fbank` a Kaldi-compatible filterbank. This
  keeps the graph portable and the front-end inspectable.
- **Resample on the way in.** Accept any rate, resample to the model's native rate, and
  say what you return.
- **Load lazily.** Build the session on first use, not in `__init__`, so listing engines
  stays cheap.
- **Resolve weights through `audiosronnx.resolver.resolve`**, pinned by revision, so
  downloads are cached and reproducible.
- **Survive degenerate input.** Empty, single-sample, silent, and non-finite input must
  return finite output of the contracted length rather than raising.
