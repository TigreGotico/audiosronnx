"""Upscale a low-bandwidth WAV to 48 kHz with the default engine."""
from audiosronnx import load_sr

sr = load_sr("lavasr")  # or "novasr" for the tiny fast engine

# path in, 48 kHz WAV out
sr.upscale_file("input_8k.wav", "output_48k.wav")

# or work with numpy arrays
import soundfile as sf  # noqa: E402

audio, in_sr = sf.read("input_8k.wav")
out, out_sr = sr.upscale(audio, in_sr)
sf.write("output_48k.wav", out, out_sr)
print(f"upscaled to {out_sr} Hz, {len(out)} samples")
