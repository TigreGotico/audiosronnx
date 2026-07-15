"""Export HiFi-GAN+ bandwidth extender (brentspell/hifi-gan-bwe) to ONNX.

The upstream generator is ``BandwidthExtender`` = a bandlimited-interpolation
upsampler followed by a non-causal WaveNet (stacked gated dilated 1-D convs) and a
``tanh``. Only the WaveNet is neural; the resampling and edge padding are plain DSP.

This exports just the WaveNet as one ONNX graph ``[1, 1, T] -> [1, 1, T]``; the
adapter does the kaiser resample and padding in numpy/scipy. The exported graph plus
the numpy pipeline are validated end-to-end against the reference TorchScript model.

Requires torch + torchaudio and the ``hifi_gan_bwe`` package on the path.

Usage:
    python export_hifiganbwe.py --model /path/hifi-gan-bwe-10.pt --output-dir ./out
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def export(model_pt: str, output_dir: Path):
    from hifi_gan_bwe.models import BandwidthExtender

    output_dir.mkdir(parents=True, exist_ok=True)
    ref = BandwidthExtender.from_pretrained(str(model_pt))  # TorchScript module
    ref.eval()

    # Rebuild a fresh (weightnorm-free) model and copy the WaveNet weights so we can
    # export the WaveNet submodule cleanly.
    fresh = BandwidthExtender()
    missing, unexpected = fresh.load_state_dict(ref.state_dict(), strict=False)
    wavenet = fresh._wavenet.eval()

    # onnxruntime cannot run a dilated conv with "same" auto-padding; convert every
    # such conv to the equivalent explicit symmetric padding (odd kernels only).
    for m in wavenet.modules():
        if isinstance(m, torch.nn.Conv1d) and m.padding == "same":
            k, d = m.kernel_size[0], m.dilation[0]
            m.padding = (d * (k - 1) // 2,)

    rf = int(wavenet.receptive_field)
    train_sr = int(ref.sample_rate.item())
    print(f"receptive_field={rf}  sample_rate={train_sr}  "
          f"missing={len(missing)} unexpected={len(unexpected)}")

    dummy = torch.randn(1, 1, 24000)
    path = output_dir / "hifiganbwe_wavenet.onnx"
    kw = dict(
        input_names=["audio"], output_names=["wavenet_out"],
        dynamic_axes={"audio": {0: "batch", 2: "samples"},
                      "wavenet_out": {0: "batch", 2: "samples"}},
        opset_version=17, do_constant_folding=True,
    )
    try:
        torch.onnx.export(wavenet, dummy, str(path), dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(wavenet, dummy, str(path), **kw)

    # end-to-end parity: reference jit model vs numpy pipeline over the ONNX wavenet
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import onnxruntime as ort

    from audiosronnx._kaiser import kaiser_resample

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    src_sr = 16000
    wave = np.random.randn(src_sr).astype(np.float32) * 0.1
    with torch.no_grad():
        ref_out = ref(torch.from_numpy(wave), src_sr).numpy()

    # numpy replica: resample -> pad rf/2 -> wavenet -> tanh -> unpad
    up = kaiser_resample(wave, src_sr, train_sr)
    pad = rf // 2
    p = np.pad(up, (pad, pad))[None, None, :]
    y = sess.run(None, {"audio": p.astype(np.float32)})[0]
    y = np.tanh(y)[0, 0, pad:-pad]
    n = min(len(y), len(ref_out))
    err = float(np.max(np.abs(y[:n] - ref_out[:n])))
    mb = path.stat().st_size / 1e6
    print(f"hifiganbwe_wavenet.onnx {mb:.2f} MB  end-to-end parity max abs err={err:.2e} "
          f"{'PASS' if err < 5e-3 else 'FAIL'}")
    return err


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("./out/hifiganbwe"))
    a = ap.parse_args()
    export(a.model, a.output_dir)
