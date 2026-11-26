# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses semantic
versioning driven by conventional commits.

## [Unreleased]

### Added
- Cross-engine **benchmark harness** (`benchmarks/`, `benchmark` install extra): scores
  every registered engine on a fixed, externally-hosted gold set with the
  `speechonnxmetrics` library — VoiceBank+DEMAND for denoising, VCTK for bandwidth
  extension and a seeded compound-degradation restoration board. `python -m benchmarks.run`
  writes a summary; `--promote` publishes it; `benchmarks/render.py` renders the measured
  tables into the docs idempotently. An opt-in `pytest -m benchmark` floor test guards the
  default engines against regressions.
- `load_sr(engine=...)` facade over an `SRModel` ABC and an engine registry
  (`ENGINE_REGISTRY`, `register_engine`, `get_engine`, `available_models`).
- HuggingFace / URL / local ONNX resolver with an XDG-based download cache.
- CLI: `list`, `probe`, `upscale`, `upscale-dir`, `denoise`, `denoise-dir`. `list`
  groups engines by job; `upscale` takes `--precision`, `denoise` takes `--model`.
- `docs/` covering engines, denoising, the API, the CLI, custom engines, conversion, and
  the models evaluated but not shipped.
- `examples/` with runnable scripts for upscaling, denoising, chaining the two, batch
  directories, scoring every denoiser on your own audio, and registering an engine.
- **lavasr** engine — Vocos-based bandwidth extension, any 8-48 kHz input to
  48 kHz, with an optional UL-UNAS denoiser and Linkwitz-Riley spectral merge.
- **novasr** engine — tiny conv1d generator, 16 kHz to 48 kHz.
- **hifiganbwe** engine — HiFi-GAN+ WaveNet bandwidth extension, any input to 48 kHz.
- **apbwe** engine — AP-BWE dual-ConvNeXt amplitude/phase bandwidth extension.
- **callenhancer** engine — CallEnhancer (Scicom-intl) call-centre / telephony
  restoration: full 24-layer w2v-BERT 2.0 feature predictor (LoRA-merged) + 188M-param
  DAC vocoder, 8–16 kHz to 48 kHz. Reuses Sidon's numpy SeamlessM4T front-end; single
  length-invariant pass by default with an optional crossfaded windowing fallback for
  very long calls. Ships both an fp32 feature extractor (default, full fidelity) and an
  int8 one (`precision="int8"`, ~4x smaller but ~12 dB SNR lossy on this 24-layer model);
  README documents the per-engine quantization trade-off. ONNX weights are CC-BY-NC-4.0.
- **deepfilternet** engine — DeepFilterNet3 denoiser (ERB gain mask + deep filtering),
  48 kHz, behind the new `load_denoise()` / `Denoiser` API.
- **dpdfnet** engine — DPDFNet (Ceva) streaming denoiser as a single stateful ONNX
  graph, with 8 / 16 / 48 kHz variants and an `attn_limit_db` control. Needs no extra
  dependencies; matches the upstream reference to 1.7e-8.
- **gtcrn** engine — ultra-light 16 kHz denoiser (23.7 K params, ~0.5 MB) as a single
  stateful ONNX graph, for embedded / on-device use.
- **frcrn** engine — FRCRN (ClearerVoice-Studio) 16 kHz complex-mask denoiser exported
  to a single waveform-to-waveform ONNX graph; matches the upstream pipeline to
  correlation 0.99999994.
- **mossformer2** engine — MossFormer2 (ClearerVoice-Studio) 48 kHz fullband denoiser;
  matches the upstream pipeline to correlation 0.99999999.
- **mpsenet** engine — MP-SENet 16 kHz denoiser with parallel magnitude/phase prediction,
  in DNS and VoiceBank checkpoints via `model=`.
- **cmgan** engine — CMGAN conformer metric-GAN denoiser, 16 kHz.
- **voicefixer** engine — general 44.1 kHz speech restoration (noise, reverb, clipping and
  bandwidth loss together) as a ResUNet mel predictor plus a TFGAN vocoder, over a fixed
  5 s window. Registered as an `enhance` engine.
- **mossformergan** engine — MossFormer attention with a metric-GAN objective, the
  highest published PESQ of the shipped denoisers, over a fixed 401-frame window.
- **metadenoiser** engine — Meta's causal-Demucs waveform denoiser (dns64 / dns48), run
  over a fixed 10 s window with a crossfaded slide. Weights are CC-BY-NC-4.0.
- **flowhigh** engine — FLowHigh conditional flow-matching bandwidth extension to 48 kHz
  in 4 Euler steps, with a BigVGAN vocoder; `steps`, `seed` and `merge` controls.
- `_stft.hamming_window` — periodic Hamming, distinct from the symmetric window Kaldi
  feature extraction uses.
- Kaldi-compatible log-mel filterbank and deltas in numpy (`_kaldi_fbank`), ported from
  `torchaudio.compliance.kaldi` to within 3.2e-05.
- `_stft.stft`/`istft` accept an explicit analysis `window` (plus a `vorbis_window`
  helper), so window choice is no longer hard-coded to Hann.
- Pure-numpy torch-faithful STFT/ISTFT (`_stft`) and kaiser resampler (`_kaiser`),
  keeping every spectral operation out of the ONNX graphs.
- Maintainer export scripts under `conversion/` with per-graph and end-to-end parity
  checks against the original PyTorch models.
