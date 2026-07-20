# Command line

```bash
audiosronnx list                    # every engine, grouped by job
audiosronnx probe input.wav         # sample rate, duration, peak
```

## Bandwidth extension

```bash
audiosronnx upscale in.wav out_48k.wav
audiosronnx upscale in.wav out.wav --engine novasr
audiosronnx upscale in.wav out.wav --denoise                  # lavasr's denoiser stage
audiosronnx upscale in.wav out.wav --engine callenhancer --precision int8
audiosronnx upscale-dir clips_in/ clips_out/ --engine apbwe
```

| Flag | Applies to | Meaning |
|------|-----------|---------|
| `--engine` | all | engine alias (default `lavasr`) |
| `--denoise` | lavasr | run its internal UL-UNAS denoiser first |
| `--precision` | callenhancer | `fp32` (default) or `int8` |

## Denoising

```bash
audiosronnx denoise noisy.wav clean.wav
audiosronnx denoise noisy.wav clean.wav --engine mossformer2
audiosronnx denoise call.wav clean.wav --engine dpdfnet --model dpdfnet2_8khz
audiosronnx denoise-dir clips_in/ clips_out/ --engine gtcrn
```

| Flag | Meaning |
|------|---------|
| `--engine` | denoise engine alias (default `dpdfnet`) |
| `--model` | engine-specific checkpoint, e.g. `dpdfnet8`, `vb` |

Output lands at the engine's native rate, which is not necessarily the input rate — see
[api.md](api.md). `probe` the result if it matters.

## Denoise, then upscale

The two commands compose in the order that matters — clean first, extend second:

```bash
audiosronnx denoise noisy_8k.wav clean_8k.wav --engine dpdfnet
audiosronnx upscale clean_8k.wav final_48k.wav --engine lavasr
```

## Directory processing

`upscale-dir` and `denoise-dir` read `.wav`, `.flac`, `.ogg`, `.mp3`, `.opus` and `.m4a`,
and write `.wav`. Written paths go to stdout and the summary to stderr, so a pipeline can
consume the list:

```bash
audiosronnx denoise-dir raw/ clean/ 2>/dev/null | wc -l
```
