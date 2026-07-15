# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses semantic
versioning driven by conventional commits.

## [Unreleased]

### Added
- `load_sr(engine=...)` facade over an `SRModel` ABC and an engine registry
  (`ENGINE_REGISTRY`, `register_engine`, `get_engine`, `available_models`).
- HuggingFace / URL / local ONNX resolver with an XDG-based download cache.
- CLI: `list`, `probe`, `upscale`, `upscale-dir`.
- **lavasr** engine — Vocos-based bandwidth extension, any 8-48 kHz input to
  48 kHz, with an optional UL-UNAS denoiser and Linkwitz-Riley spectral merge.
- **novasr** engine — tiny conv1d generator, 16 kHz to 48 kHz.
- **hifiganbwe** engine — HiFi-GAN+ WaveNet bandwidth extension, any input to 48 kHz.
- **apbwe** engine — AP-BWE dual-ConvNeXt amplitude/phase bandwidth extension.
- Pure-numpy torch-faithful STFT/ISTFT (`_stft`) and kaiser resampler (`_kaiser`),
  keeping every spectral operation out of the ONNX graphs.
- Maintainer export scripts under `conversion/` with per-graph and end-to-end parity
  checks against the original PyTorch models.
