# DeepFilterNet3 → audiosronnx (WIP, ~85% — finish is deterministic)

DFN3 will NOT export from torch (`view_as_complex` unsupported; separate-graph hits
`Unsupported value kind: Tensor`). Instead use DeepFilterNet's **prebuilt ONNX** and wrap
it — the plan below is fully mapped; only the DSP marshalling remains.

## Prebuilt ONNX (obtained, load in pure onnxruntime, no torch)
`https://github.com/Rikorose/DeepFilterNet/raw/1e96ef05e1ef75b3702f8c55ca065368deae637d/models/DeepFilterNet3_onnx.tar.gz`
→ `enc.onnx`, `erb_dec.onnx`, `df_dec.onnx`.
- enc: (feat_erb[1,1,T,32], feat_spec[1,2,T,96]) → e0,e1,e2,e3, emb[1,T,512], c0[1,64,T,96], lsnr
- erb_dec: (emb, e3,e2,e1,e0) → m[1,1,T,32]  (sigmoid ERB mask)
- df_dec: (emb, c0) → coefs[1,T,·,10]  (df_order=5 complex)

## Config (DeepFilterNet3/config.ini)
sr 48000 · fft 960 · hop 480 · nb_erb 32 · nb_df 96 · df_order 5 · df_lookahead 2 · min_nb_erb_freqs 2 · norm alpha 0.99

## Algorithm (verified against df/deepfilternet3.py + multiframe.py)
1. spec = STFT(audio)                                   # [T, F=481] complex
2. feat_erb = erb_norm(erb(|spec|², erb_widths), α)      # [T,32]
   feat_spec = unit_norm(spec[:, :96], α) as_real        # [T,96,2]
3. enc → emb,c0,e*  ;  erb_dec → m  ;  df_dec → coefs
4. mask: spec_m = spec * upscale(m, erb_widths)          # repeat each ERB band across its width
5. deep filter (low 96 bins): pad spec_m time by (df_order-1-lookahead)=2 left, 2 right;
   unfold into frames of 5; `einsum("tfn,ntf->tf", frames, coefs_complex)`; coefs reshaped
   [T,5,96,2]→complex→transpose→[5,T,96]. Replace spec_m[:, :96] with the result.
6. audio = ISTFT(spec_m)

## Verified findings (debug round 2)

- **`libdf` works WITHOUT torch** — the earlier segfaults were a torch/libdf ABI clash, not a
  shape bug. `DF.analysis/synthesis/erb_widths/fft_window`, `erb`, `erb_norm`, `unit_norm`,
  `erb_inv` all run clean in a torch-free venv. **So the adapter is libdf + onnxruntime +
  numpy, zero torch** — exactly audiosronnx's ethos, and it does not segfault.
- **Any torch co-loaded with libdf segfaults** (tried 2.4.1 and 2.1.0; the `deepfilterlib`
  wheel is ABI-incompatible with installable torch here). Consequence: `df.enhance.enhance()`
  cannot run on this box, so **bit-exact parity vs the torch reference is not available**.
  Validate instead with real speech: noisy = clean + noise, require SNR improvement > 3 dB
  and best-lag `corr(clean, denoised) > 0.9`.
- **STFT round-trip verified**: `d.synthesis(d.analysis(x)) ≈ x`, corr **0.9999** at lag
  **-480** (one hop of latency). Analysis/synthesis usage is correct; the bug is elsewhere.
- **Config**: `norm_tau = 1.0` → `alpha = exp(-(hop/sr)/tau) = 0.99`.
- **Current bug**: mask+deep-filter output scores `corr(clean, denoised) = 0.15` on real
  speech (must be > 0.9). Noise floor drops correctly, but speech is destroyed too.
- **Leading hypothesis (test first)**: `enc.onnx` expects features **padded by
  `conv_lookahead` (=2)**. The model's forward does `feat_erb = self.pad_feat(feat_erb)` /
  `feat_spec = self.pad_feat(feat_spec)` *before* `self.enc(...)`, and `pad_feat` is OUTSIDE
  the exported `enc` graph. Feeding unpadded features misaligns every enc output, so the mask
  and coefs land on the wrong frames. Fix: replicate `pad_feat` (see `deepfilternet3.py`
  `__init__`/`forward`) before the enc call, then re-run the real-speech check. Isolate by
  testing **mask-only** vs **mask+df** separately.

## Remaining blocker + the finish
`conversion/deepfilternet_reference_wip.py` implements all of the above but SEGFAULTS in
`libdf`'s analysis/synthesis FFI (opaque Rust boundary, array layout mismatch). Two finishes:
- **(preferred, matches audiosronnx ethos — no libdf dep)** replace analysis/synthesis with
  audiosronnx's own `_stft.py` (fft 960 / hop 480 / window = `df_state.fft_window()`, a
  sqrt-Hann/Vorbis window) and port `erb`/`erb_norm`/`unit_norm` to numpy (erb = band-sum
  matrix; norms = exponential-average filters with α=0.99). Then no segfault, pure numpy+onnx.
- (quick) match libdf's exact expected array shape/dtype/contiguity for analysis/synthesis.

Acceptance: parity vs `df.enhance.enhance()` (corr > 0.99) on a noisy clip. Then wrap as
`audiosronnx/engines/deepfilternet.py` (Denoiser subclass, kind="denoise", input_sr=48000),
publish the 3 ONNX to `TigreGotico/audiosronnx-deepfilternet`, register.
