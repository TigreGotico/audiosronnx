"""End-to-end voice isolator + adversarial inputs.

The isolator keeps one speaker and removes other voices, so the inference tests mix two
voices and check that the enrollment decides which one survives. The target is a fullband
LibriTTS-R utterance shipped in the upstream repository; the interferer is the repository's
own narrowband ``speech16k.wav``.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from audiosronnx.engines.voiceisolator import (
    _HF_REPO,
    _HF_REVISION,
    _MODELS,
    _SR,
    VoiceIsolatorAdapter,
    VoiceIsolatorStream,
    parse_gate,
)


# --------------------------------------------------------------------------- #
# Configuration (no weights) — always runs.
# --------------------------------------------------------------------------- #
def test_default_model_is_release():
    assert VoiceIsolatorAdapter()._model == "release"


def test_both_models_selectable():
    assert set(_MODELS) == {"release", "agent"}
    assert VoiceIsolatorAdapter(model="agent")._model == "agent"


def test_unknown_model_rejected():
    with pytest.raises(ValueError):
        VoiceIsolatorAdapter(model="gpu")


def test_engine_states_its_license():
    """Upstream publishes no license text; the registry must not claim a permissive one."""
    from audiosronnx import get_engine

    entry = get_engine("voiceisolator")
    assert entry.license.startswith("other")
    assert entry.kind == "denoise"
    assert entry.input_sample_rate == entry.output_sample_rate == _SR


@pytest.mark.parametrize("spec, expected", [
    ("0.9:30:0.2:1", dict(theta=0.9, hold=30, theta_off=0.2, keep_db=1.0)),
    ("0.5", dict(theta=0.5, hold=10, theta_off=None, keep_db=None)),
    ("0.7::0.3", dict(theta=0.7, hold=10, theta_off=0.3, keep_db=None)),
])
def test_parse_gate(spec, expected):
    assert parse_gate(spec) == expected


@pytest.mark.parametrize("spec", [None, "", "off"])
def test_parse_gate_off(spec):
    assert parse_gate(spec) is None


def test_gate_on_a_model_without_presence_head_rejected():
    """The release graph has no gate output; asking for a gate must fail loudly."""
    cfg = {
        "win": 320, "hop": 160, "window": 1000, "layers": 1, "heads": 1,
        "head_dim": 2, "rope_base": 10000.0, "null": [0.0], "states": [],
    }
    with pytest.raises(ValueError, match="presence head"):
        VoiceIsolatorStream(cfg, None, None, gate="0.9:30")


# --- the 555 M model's runtime: a fixed chunk and the WavLM branch (no weights) --- #
_C, _WWIN = 10, 8         # frames per step (5 WavLM frames); WavLM window (ring of 7)


class _FakeStep:
    """Records each feed; returns zeros shaped like the 555 M step graph's outputs."""

    def __init__(self, cfg):
        self.cfg, self.feeds = cfg, []

    def run(self, _, feed):
        self.feeds.append(feed)
        c, wl = self.cfg, self.cfg["wavlm"]
        C, k = feed["re"].shape[1], feed["wvalid"].size
        outs = [np.zeros_like(feed["re"]), np.zeros_like(feed["im"]),
                np.ones((c["layers"], 1, c["heads"], C, c["head_dim"]), np.float32),
                np.ones((c["layers"], 1, c["heads"], C, c["head_dim"]), np.float32)]
        outs += [feed[st["name"]] for st in c["states"]]
        jj = np.arange(k, dtype=np.float32)[None, None, :, None] + len(self.feeds) * 100
        wk = np.broadcast_to(jj, (wl["layers"], 1, wl["heads"], k, wl["head_dim"])).copy()
        outs += [feed["wpos"] + 1, wk, wk]
        return outs


def _large_stream():
    cfg = {
        "win": 320, "hop": 160, "window": 8, "layers": 1, "heads": 1, "head_dim": 2,
        "rope_base": 10000.0, "null": [0.0], "states": [{"name": "hb0", "shape": [1, 1, 1]}],
        "chunk_frames": _C,
        "wavlm": {"layers": 1, "heads": 1, "head_dim": 2, "window": _WWIN, "d": 1, "pos_taps": 3},
    }
    step = _FakeStep(cfg)
    return VoiceIsolatorStream(cfg, step, None, live_enroll=False, gate=None), step


def test_large_steps_only_whole_chunks():
    s, step = _large_stream()
    assert s.push(np.zeros(160 * (_C - 1), np.float32)).size == 0
    assert step.feeds == []
    assert s.push(np.zeros(160, np.float32)).size == 160 * _C
    assert len(step.feeds) == 1


def test_large_chunk_must_divide_a_half_second():
    cfg = dict(_large_stream()[0].cfg, chunk_frames=12)   # 50 frames / 12
    with pytest.raises(ValueError, match="half second"):
        VoiceIsolatorStream(cfg, None, None, live_enroll=False, gate=None)


def test_large_first_chunk_has_no_wavlm_frame_minus_one():
    """The first chunk finishes WavLM frame -1: it does not exist and may see only itself."""
    s, step = _large_stream()
    s.push(np.zeros(160 * _C, np.float32))
    f = step.feeds[0]
    k = _C // 2
    np.testing.assert_array_equal(f["wvalid"], [0.0] + [1.0] * (k - 1))
    allowed = f["wmask"][0, 0] == 0
    ring = _WWIN - 1
    assert not allowed[:, :ring].any()                   # an empty ring
    new = allowed[:, ring:]
    expected = np.tril(np.ones((k, k), bool))
    expected[1:, 0] = False                              # no frame sees frame -1
    expected[0, 0] = True                                # frame -1 sees only itself
    np.testing.assert_array_equal(new, expected)


def test_large_ring_holds_the_window():
    """Frames enter the ring after their step, and a query sees only the last window - 1."""
    s, step = _large_stream()
    for _ in range(3):
        s.push(np.zeros(160 * _C, np.float32))
    # the chunks finish WavLM frames -1..3, 4..8, 9..13; a ring of 7 keeps the last 7
    np.testing.assert_array_equal(sorted(s.w_pos[s.w_pos >= 0]), list(range(7, 14)))
    f = step.feeds[2]                                   # queries 9..13, the ring holding 2..8
    queries = np.arange(9, 14)
    ring_frames = f["wrel"][0, : _WWIN - 1] + queries[0]
    np.testing.assert_array_equal(sorted(ring_frames), list(range(2, 9)))
    allowed = f["wmask"][0, 0, :, : _WWIN - 1] == 0
    for q, row in zip(queries, allowed):
        np.testing.assert_array_equal(row, ring_frames > q - _WWIN)
    assert s.w_j == 14


def test_large_rms_normalisation_carries_across_chunks():
    """WavLM's input is divided by the RMS of everything up to each sample, not per chunk."""
    s, step = _large_stream()
    a = np.full(160 * _C, 0.5, np.float32)
    s.push(a)
    s.push(2 * a)
    second = step.feeds[1]["wx"][0, 320:]
    # sample i of the second chunk: 1.0 / sqrt((n0 * 0.25 + i * 1.0) / (n0 + i))
    n0 = a.size
    i = np.arange(1, a.size + 1)
    expected = 1.0 / np.sqrt((n0 * 0.25 + i * 1.0) / (n0 + i) + 1e-10)
    np.testing.assert_allclose(second, expected, rtol=1e-6)


def test_large_flush_pads_to_a_whole_chunk():
    s, _ = _large_stream()
    s.push(np.zeros(160 * _C + 7, np.float32))
    assert s.flush().size % (160 * _C) == 0


# --------------------------------------------------------------------------- #
# Inference (weights) — skipped if unavailable.
# --------------------------------------------------------------------------- #
def _isolator(**kw):
    from audiosronnx import load_denoise

    try:
        eng = load_denoise("voiceisolator", **kw)
        eng._ensure_models()
        return eng
    except Exception as exc:
        pytest.skip(f"voiceisolator weights unavailable: {exc}")


@pytest.fixture(scope="module")
def voices():
    """(target, interferer): a fullband voice and a narrowband one, equal RMS, 3 s."""
    import soundfile as sf

    from audiosronnx.resolver import resolve

    try:
        path = resolve("voice_isolator/resources/voicebot_decoy.npy",
                       hf_repo=_HF_REPO, revision=_HF_REVISION)
    except Exception as exc:
        pytest.skip(f"target voice unavailable: {exc}")
    interferer, _ = sf.read(
        os.path.join(os.path.dirname(__file__), "resources", "speech16k.wav"), dtype="float32")
    target = np.load(path, allow_pickle=False).astype(np.float32)[: interferer.size] / 32768.0
    target *= np.sqrt((interferer ** 2).mean() / (target ** 2).mean())
    return target.astype(np.float32), interferer


def _sisdr(est, ref):
    k = (est @ ref) / (ref @ ref)
    t = k * ref
    return 10 * np.log10((t @ t) / (((est - t) ** 2).sum() + 1e-20))


def _snr(x, clean):
    n = min(x.size, clean.size)
    return 10 * np.log10((clean[:n] ** 2).sum() / (((x[:n] - clean[:n]) ** 2).sum() + 1e-20))


@pytest.mark.parametrize("model", ["release", "agent"])
@pytest.mark.parametrize("quantized", [False, True])
def test_keeps_the_enrolled_speaker(voices, model, quantized):
    target, interferer = voices
    mix = target + interferer
    out, rate = _isolator(model=model, quantized=quantized, enrollment=target).denoise(mix, _SR)
    assert rate == _SR and out.size == mix.size
    assert _sisdr(out, target) > _sisdr(mix, target) + 6.0     # measured +12.8 to +13.2 dB


def test_enrollment_steers_which_voice_survives(voices):
    """Enrolling the other voice must keep far more of it: the vector is actually used."""
    target, interferer = voices
    mix = target + interferer
    on_target, _ = _isolator(enrollment=target).denoise(mix, _SR)
    on_interferer, _ = _isolator(enrollment=interferer).denoise(mix, _SR)
    gain = _sisdr(on_interferer, interferer) - _sisdr(on_target, interferer)
    assert gain > 10.0                                          # measured +17.0 dB


def test_enrollment_from_file_matches_array(voices, tmp_path):
    import soundfile as sf

    target, interferer = voices
    clip = tmp_path / "enroll.wav"
    sf.write(clip, target, _SR, subtype="FLOAT")
    mix = target + interferer
    from_file, _ = _isolator(enrollment=str(clip)).denoise(mix, _SR)
    from_array, _ = _isolator(enrollment=target).denoise(mix, _SR)
    assert np.array_equal(from_file, from_array)


def test_stereo_enrollment_is_downmixed(voices):
    target, interferer = voices
    mix = target + interferer
    stereo = np.stack([target * 1.5, target * 0.5], 1)          # mean: the target
    from_stereo, _ = _isolator(enrollment=stereo).denoise(mix, _SR)
    from_mono, _ = _isolator(enrollment=stereo.mean(1)).denoise(mix, _SR)
    assert np.array_equal(from_stereo, from_mono)


def test_int16_enrollment_is_scaled(voices):
    target, interferer = voices
    pcm = np.round(np.clip(target, -1, 1) * 32767).astype(np.int16)
    mix = target + interferer
    from_int, _ = _isolator(enrollment=pcm).denoise(mix, _SR)
    from_float, _ = _isolator(enrollment=pcm.astype(np.float32) / 32768.0).denoise(mix, _SR)
    assert np.array_equal(from_int, from_float)


def test_empty_enrollment_rejected():
    with pytest.raises(ValueError, match="empty"):
        _isolator(enrollment=np.zeros(0, np.float32)).stream()


def test_reset_keeps_a_clip_enrollment(voices):
    target, interferer = voices
    s = _isolator(enrollment=target).stream()
    first = s.push(target + interferer)
    s.reset()
    assert not s.live
    assert np.array_equal(s.push(target + interferer), first)


@pytest.fixture(scope="module")
def upstream_runtime():
    """Upstream's own numpy runtime, ``voice_isolator/onnx_stream.py`` at the pinned revision."""
    import importlib.util

    from audiosronnx.resolver import resolve

    try:
        path = resolve("voice_isolator/onnx_stream.py", hf_repo=_HF_REPO, revision=_HF_REVISION)
    except Exception as exc:
        pytest.skip(f"upstream runtime unavailable: {exc}")
    spec = importlib.util.spec_from_file_location("voice_isolator_onnx_stream", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("model", ["release", "agent"])
@pytest.mark.parametrize("quantized", [False, True])
def test_matches_upstream_runtime_past_the_rings(voices, upstream_runtime, model, quantized):
    """22 s of two voices: past the 10 s attention ring and the 15.5 s live enrollment."""
    target, interferer = voices
    x = np.concatenate([target, interferer, target + interferer, 0.5 * target, interferer,
                        target, target + 0.5 * interferer, interferer]).astype(np.float32)
    x = np.concatenate([x, x[: 22 * _SR - x.size]]) if x.size < 22 * _SR else x[: 22 * _SR]
    eng = _isolator(model=model, quantized=quantized)
    ours, _ = eng.denoise(x, _SR)
    theirs = upstream_runtime.isolate(os.path.dirname(eng._file("config.json")), x)
    assert ours.shape == theirs.shape == x.shape
    np.testing.assert_array_equal(ours, theirs)


def test_clip_enrollment_matches_upstream_runtime(voices, upstream_runtime):
    target, interferer = voices
    x = np.tile(target + interferer, 4)
    eng = _isolator(enrollment=target)
    ours, _ = eng.denoise(x, _SR)
    s = upstream_runtime.OnnxStream(os.path.dirname(eng._file("config.json")))
    s.set_enrollment(target)
    parts = [s.push(x[i:i + 800]) for i in range(0, x.size, 800)]
    parts.append(s.push(np.zeros(s.delay + s.hop, np.float32)))
    theirs = np.concatenate(parts)[s.delay:s.delay + x.size]
    np.testing.assert_array_equal(ours, theirs)


def test_removes_broadband_noise(speech16k_path):
    import soundfile as sf

    clean, sr = sf.read(speech16k_path, dtype="float32")
    noisy = (clean + 0.05 * np.random.RandomState(42).randn(clean.size)).astype(np.float32)
    out, _ = _isolator().denoise(noisy, sr)
    assert _snr(out, clean) > _snr(noisy, clean) + 3.0          # measured +5.8 dB


def test_live_enrollment_replaces_the_null_vector(speech16k_path):
    import soundfile as sf

    clean, _ = sf.read(speech16k_path, dtype="float32")
    x = np.tile(clean, 2)                                       # 6 s: past the 3 s mark
    live = _isolator().stream()
    live.push(x)
    assert not np.array_equal(live.emb, live.null)
    fixed = _isolator(live_enroll=False).stream()
    fixed.push(x)
    assert np.array_equal(fixed.emb, fixed.null)


def test_agent_gate_silences_a_line_without_speech():
    noise = (0.01 * np.random.RandomState(0).randn(3 * _SR)).astype(np.float32)
    gated, _ = _isolator(model="agent").denoise(noise, _SR)
    assert (gated == 0).mean() > 0.9                            # measured 1.0
    ungated, _ = _isolator(model="agent", gate="off").denoise(noise, _SR)
    assert (ungated == 0).mean() < 0.01


def test_release_has_no_gate_by_default():
    noise = (0.01 * np.random.RandomState(0).randn(3 * _SR)).astype(np.float32)
    out, _ = _isolator().denoise(noise, _SR)
    assert (out == 0).mean() < 0.01


def test_release_rejects_a_gate():
    with pytest.raises(ValueError, match="presence head"):
        _isolator(gate="0.9:30").stream()


def test_stream_matches_offline(speech16k_path):
    """Pushing ragged chunks gives the offline result, to float rounding."""
    import soundfile as sf

    clean, _ = sf.read(speech16k_path, dtype="float32")
    x = np.concatenate([clean, 0.5 * clean, clean]).astype(np.float32)
    eng = _isolator(model="agent")
    offline, _ = eng.denoise(x, _SR)
    s = eng.stream()
    rng = np.random.RandomState(3)
    parts, i = [], 0
    while i < x.size:
        n = int(rng.randint(1, 3000))
        parts.append(s.push(x[i:i + n]))
        i += n
    parts.append(s.flush())
    streamed = np.concatenate(parts)[s.delay:s.delay + x.size]
    np.testing.assert_allclose(streamed, offline, atol=1e-5)    # measured 2.4e-7


def test_stream_output_lags_by_the_delay():
    s = _isolator().stream()
    assert s.delay == 160
    assert s.push(np.zeros(1000, np.float32)).size == 960      # whole 10 ms hops only


def test_reset_reproduces_the_first_run(speech16k_path):
    import soundfile as sf

    clean, _ = sf.read(speech16k_path, dtype="float32")
    s = _isolator().stream()
    first = s.push(np.tile(clean, 2))
    s.reset()
    assert np.array_equal(s.push(np.tile(clean, 2)), first)


def test_resamples_foreign_rate():
    t = np.arange(48000, dtype=np.float32) / 48000.0
    out, rate = _isolator().denoise((0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 48000)
    assert rate == _SR and out.size == _SR
    assert np.isfinite(out).all()


def test_is_deterministic():
    eng = _isolator()
    x = (0.1 * np.random.RandomState(2).randn(_SR)).astype(np.float32)
    assert np.array_equal(eng.denoise(x, _SR)[0], eng.denoise(x, _SR)[0])


# --- adversarial --- #
def test_empty_input():
    out, _ = _isolator().denoise(np.zeros(0, dtype=np.float32), _SR)
    assert out.size == 0


def test_single_sample():
    out, _ = _isolator().denoise(np.array([0.5], dtype=np.float32), _SR)
    assert out.size == 1
    assert np.isfinite(out).all()


def test_silence_stays_finite():
    out, _ = _isolator().denoise(np.zeros(_SR, dtype=np.float32), _SR)
    assert out.size == _SR
    assert np.isfinite(out).all()


def test_non_finite_input_does_not_propagate():
    """A streaming state carries forward, so one NaN must not poison the rest."""
    x = np.array([0.1, np.nan, np.inf, -np.inf] * 4000, dtype=np.float32)
    out, _ = _isolator().denoise(x, _SR)
    assert np.isfinite(out).all()


def test_extremely_loud_input():
    x = (np.random.RandomState(1).randn(_SR) * 1e4).astype(np.float32)
    out, _ = _isolator().denoise(x, _SR)
    assert np.isfinite(out).all()
