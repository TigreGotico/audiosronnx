"""Export VoiceFixer (haoheliu/voicefixer) to ONNX.

VoiceFixer is general 44.1 kHz speech restoration — noise, reverberation, clipping and
bandwidth loss together. Two stages, the same shape as sidon: a ResUNet predicts a clean
log-mel and a TFGAN vocoder resynthesises the waveform. The STFT, the mel projection and
the ``10**clip(x, max=5)`` log inverse stay in numpy.

Two things are worth knowing before reading the output of this script.

**The analysis graph declares only ``mel_orig``.** The module signature is
``forward(sp, mel_orig)``, but ``sp`` is genuinely unused: zeroing or scaling it changes the
module's output by exactly zero. The tracer is right to drop it. Verify by perturbing the
argument rather than assuming a folded constant.

**Parity must be measured on speech.** The same exported graph reads 6.8e-02 against random
noise and 2.3e-04 against real speech. Noise is out of domain for a restoration model and
drives it into a regime where float error through a 70M-parameter ResUNet amplifies. This
script therefore checks relative error and correlation, and warns if only noise is available.

The analysis stage is also length-specialised, so it is exported at a fixed 5 s window and
the adapter slides it.

Usage::

    python export_voicefixer.py --output-dir ./out/voicefixer [--speech reference.wav]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

SR = 44100
SECONDS = 5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("./out/voicefixer"))
    ap.add_argument("--speech", default=None,
                    help="a real speech wav; parity on noise is not representative")
    a = ap.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)

    from voicefixer import VoiceFixer

    vf = VoiceFixer()
    model = vf._model
    model.eval()

    if a.speech:
        import soundfile as sf
        from scipy.signal import resample_poly

        raw, rate = sf.read(a.speech, dtype="float32")
        if raw.ndim > 1:
            raw = raw.mean(axis=1)
        if rate != SR:
            raw = resample_poly(raw, SR, rate).astype(np.float32)
        probe = np.zeros(SR * SECONDS, dtype=np.float32)
        probe[:min(raw.size, probe.size)] = raw[:probe.size]
        wav = torch.from_numpy(probe)[None, None]
    else:
        print("WARNING: no --speech given; parity on noise understates this model badly")
        wav = torch.randn(1, 1, SR * SECONDS) * 0.1

    sp, _, _ = model.f_helper.wav_to_spectrogram_phase(wav)
    mel_orig = model.mel(sp.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
    print(f"fixed window {SECONDS}s -> {mel_orig.shape[2]} frames")

    # sp is unused; prove it rather than assume it
    with torch.no_grad():
        base = model(sp, mel_orig)["mel"]
        zeroed = model(torch.zeros_like(sp), mel_orig)["mel"]
    print(f"sp influence on output: {float((base - zeroed).abs().max()):.2e} "
          f"(zero means the tracer may drop it)")

    class Analysis(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, sp, mel_orig):
            return self.m(sp, mel_orig)["mel"]

    class Vocoder(torch.nn.Module):
        def __init__(self, v):
            super().__init__()
            self.v = v

        def forward(self, mel):
            return self.v(mel, cuda=False)

    analysis = Analysis(model).eval()
    vocoder = Vocoder(model.vocoder).eval()

    apath = a.output_dir / "analysis.onnx"
    torch.onnx.export(
        analysis, (sp, mel_orig), str(apath),
        input_names=["sp", "mel_orig"], output_names=["mel"],
        dynamic_axes={"sp": {0: "batch"}, "mel_orig": {0: "batch"}, "mel": {0: "batch"}},
        opset_version=17, do_constant_folding=False, dynamo=False)
    print(f"{apath.name}  {apath.stat().st_size / 1e6:.1f} MB")

    with torch.no_grad():
        mel_pred = analysis(sp, mel_orig)
    vpath = a.output_dir / "vocoder.onnx"
    torch.onnx.export(
        vocoder, (mel_pred,), str(vpath),
        input_names=["mel"], output_names=["wav"],
        dynamic_axes={"mel": {0: "batch"}, "wav": {0: "batch"}},
        opset_version=17, do_constant_folding=False, dynamo=False)
    print(f"{vpath.name}  {vpath.stat().st_size / 1e6:.1f} MB")

    np.save(a.output_dir / "mel_filters.npy", model.mel.fb.numpy())
    print("mel_filters.npy saved for the numpy front-end")

    import onnxruntime as ort

    asess = ort.InferenceSession(str(apath), providers=["CPUExecutionProvider"])
    vsess = ort.InferenceSession(str(vpath), providers=["CPUExecutionProvider"])
    print("analysis inputs:", [i.name for i in asess.get_inputs()])

    feeds = {"mel_orig": mel_orig.numpy()}
    if any(i.name == "sp" for i in asess.get_inputs()):
        feeds["sp"] = sp.numpy()
    got = asess.run(None, feeds)[0]
    ref = mel_pred.numpy()
    rel = float(np.abs(ref - got).max() / max(float(np.abs(ref).max()), 1e-9))
    corr = float(np.corrcoef(ref.ravel(), got.ravel())[0, 1])
    print(f"analysis parity: rel={rel:.2e} corr={corr:.8f} {'PASS' if corr > 0.9999 else 'FAIL'}")

    with torch.no_grad():
        rw = vocoder(torch.from_numpy(got)).numpy().reshape(-1)
    gw = vsess.run(None, {"mel": got})[0].reshape(-1)
    n = min(rw.size, gw.size)
    vcorr = float(np.corrcoef(rw[:n], gw[:n])[0, 1])
    print(f"vocoder parity: max abs err={np.abs(rw[:n] - gw[:n]).max():.2e} corr={vcorr:.8f} "
          f"{'PASS' if vcorr > 0.9999 else 'FAIL'}")


if __name__ == "__main__":
    main()
