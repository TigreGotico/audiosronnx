"""Voice isolator adapter: ``voiceisolator``.

Scicom's *voice isolator* (Zolkepli, *The Voice on the Line*, arXiv:2610.35272) keeps one
speaker — the caller — and removes every other voice and the noise. It is target-speaker
extraction, not plain denoising: background speech that a denoiser keeps as "speech" is
removed too. Causal and streaming, 16 kHz in and out, 10 ms hops, 10 ms of delay.

The ONNX graphs are the CPU release, a 22 M-parameter distillation of the 555 M GPU model
(``Scicom-intl/voice-isolator``): a frame transformer (8 layers, width 384) attending over
the last 10 s, a frame LSTM and band LSTMs. Two graphs::

    enroll:  logpow[1, T, 161], mask[1, T]                      -> emb[1, 256]
    step:    re, im[1, C, 161], emb, cos, sin, bias, k0..k7, v0..v7, LSTM states
             -> yre, yim, k_new[8, ...], v_new[8, ...], next states (, gate logits)

Everything a stream keeps is a graph input and output, so one session serves any number of
streams. Outside the graph, in numpy: the sqrt-Hann STFT and overlap-add, the key/value
rings, the rotary angles and the attention mask.

Enrollment
----------
The model needs to know whose voice to keep. By default it enrolls *live* from the input
itself: from 3 s in, the speaker vector comes from the loudest 3 s heard so far, re-made
every half second until 15.5 s, then fixed. Before 3 s the model runs on its learned
"no enrollment" vector. Pass ``enrollment=`` (a clip of the target speaker alone) to lock
the voice up front instead.

Models
------
``model="release"`` keeps the most of the caller's words. ``model="agent"`` adds a presence
head: where the caller is not talking the output is exact digital silence, so a downstream
VAD is triggered less by other voices. ``quantized=True`` takes the 8-bit export.

The 555 M GPU release itself exports with ``conversion/export_voiceisolator.py`` and loads
with ``model_dir=``. Its step graph carries the causal WavLM-Large branch as well, takes a
fixed 50 frames (0.5 s) per step, and is about 2.2 GB.

**License:** the upstream model card declares ``license: other`` and publishes no license
text. The weights download from the upstream repository at a pinned revision; they are not
mirrored. Check the terms with the authors before use beyond evaluation.

The numpy side reproduces upstream's reference runtime (``voice_isolator/onnx_stream.py`` in
the model repository) without its voicebot greeting mode.

Reference
---------
- https://huggingface.co/Scicom-intl/voice-isolator-cpu  (license: other)
- https://arxiv.org/abs/2610.35272
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Union

import numpy as np

from audiosronnx._kaiser import kaiser_resample
from audiosronnx.audio import read_wav, to_float32_mono
from audiosronnx.base import Denoiser, EngineEntry, register_engine
from audiosronnx.resolver import resolve

_HF_REPO = "Scicom-intl/voice-isolator-cpu"
_HF_REVISION: Optional[str] = "1dfac5977ba5c531b0972e5b66a10160d49d78f9"
#: ``model`` alias -> directory of its config.json and graphs in the repository.
_MODELS: Dict[str, str] = {"release": "", "agent": "agent/"}
_DEFAULT_MODEL = "release"

_SR = 16000
_LOUD_S = 3.0        # the live enrollment takes the loudest this-many seconds
_LOUD_WITHIN = 15.0  # ... heard within the first this-many seconds
_STEP_S = 0.5        # ... re-made at every half second
_CHUNK = 800         # samples per push when a whole recording runs through a stream


def parse_gate(spec: Optional[str]) -> Optional[dict]:
    """``"THETA[:HOLD[:THETA_OFF[:KEEP_DB]]]"`` -> gate settings; ``None``/``"off"`` -> none.

    HOLD is in 10 ms frames (default 10).
    """
    if spec is None or str(spec) in ("", "off"):
        return None
    f = str(spec).split(":")
    return dict(
        theta=float(f[0]),
        hold=int(f[1]) if len(f) > 1 and f[1] else 10,
        theta_off=float(f[2]) if len(f) > 2 and f[2] else None,
        keep_db=float(f[3]) if len(f) > 3 and f[3] else None,
    )


def _sqrt_hann(n: int) -> np.ndarray:
    """``torch.hann_window(n).sqrt()`` — periodic Hann, square-rooted."""
    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n, dtype=np.float64) / n)
    return np.sqrt(w).astype(np.float32)


class VoiceIsolatorStream:
    """One stream's state. :meth:`push` 16 kHz audio in, the isolated speaker out.

    Output sample ``n`` is the speaker at input sample ``n - delay`` (160 samples, the STFT
    window's look-ahead). Get one from :meth:`VoiceIsolatorAdapter.stream`.
    """

    def __init__(
        self,
        cfg: dict,
        step,
        enroll,
        *,
        live_enroll: Optional[bool] = None,
        gate: Optional[str] = "auto",
        max_frames: int = 8,
    ) -> None:
        self.cfg, self._step_sess, self._enroll_sess = cfg, step, enroll
        self.win, self.hop = cfg["win"], cfg["hop"]
        self.delay = self.win - self.hop
        self.window, self.R = cfg["window"], cfg["window"] - 1
        self.L, self.heads, self.hd = cfg["layers"], cfg["heads"], cfg["head_dim"]
        self.w = _sqrt_hann(self.win)
        ola = np.zeros(self.win)
        for k in range(0, self.win, self.hop):
            ola += np.roll(self.w.astype(np.float64) ** 2, k)
        self.norm = float(ola.mean())
        self.inv = cfg["rope_base"] ** (-np.arange(0, self.hd, 2, dtype=np.float64) / self.hd)
        self.null = np.asarray(cfg["null"], np.float32)[None]
        self.max_frames = max_frames
        self.n_states = len(cfg["states"])

        serve = cfg.get("serve") or {}
        self.live_default = (
            serve.get("enroll", "loud") != "none" if live_enroll is None else bool(live_enroll))
        self.loud_until = float(serve.get("loud_within", _LOUD_WITHIN)) + _STEP_S
        self.step_n = int(_STEP_S * _SR)

        # a graph of a fixed number of frames per step (the 555 M model, whose WavLM branch
        # also needs it even); its steps then land on the live rule's half seconds
        self.fixed = cfg.get("chunk_frames")
        if self.fixed:
            if (self.step_n // self.hop) % self.fixed:
                raise ValueError(f"chunk_frames {self.fixed} does not divide a half second")
            self.max_frames = self.fixed
        self.wl = cfg.get("wavlm")

        self.gate = parse_gate(serve.get("gate") if gate == "auto" else gate)
        if self.gate is not None:
            if not cfg.get("gate_out"):
                raise ValueError(
                    "gate= needs a model exported with its presence head (model='agent')")
            self.g_th = np.float32(self.gate["theta"])
            self.g_th_off = np.float32(self.gate["theta_off"] or self.gate["theta"])
            self.w2 = self.w * self.w
        self.clip_emb: Optional[np.ndarray] = None
        self.reset()

    def reset(self) -> None:
        """Start a new stream: every state back to its start. A clip enrollment stays;
        otherwise the live enrollment starts over."""
        self.live = self.live_default and self.clip_emb is None
        self.inbuf = np.zeros(self.delay, np.float32)
        self.outbuf = np.zeros(self.delay, np.float32)
        ring = (1, self.heads, self.R, self.hd)
        self.k = [np.zeros(ring, np.float32) for _ in range(self.L)]
        self.v = [np.zeros(ring, np.float32) for _ in range(self.L)]
        self.pos = np.full(self.R, -1, np.int64)  # the frame each ring slot holds, -1 empty
        self.t = 0                                 # the next frame
        self.states = {s["name"]: np.zeros(s["shape"], np.float32) for s in self.cfg["states"]}
        self.emb = (self.null if self.clip_emb is None else self.clip_emb).copy()
        self.pending = np.zeros(0, np.float32)
        self.n_in = 0                              # input samples stepped
        self.g_since = 1 << 30                     # frames since the gate last heard the speaker
        self.g_prev = np.float32(0.0)              # a stream starts with the gate closed
        self.g_krun = 0
        n_hist = int(self.loud_until * _SR) + self.step_n
        self.hist = np.zeros(n_hist, np.float32) if self.live else None
        self.loud_st = None
        if self.wl:
            wl = self.wl
            self.w_R = wl["window"] - 1
            ring = (1, wl["heads"], self.w_R, wl["head_dim"])
            self.wk = [np.zeros(ring, np.float32) for _ in range(wl["layers"])]
            self.wv = [np.zeros(ring, np.float32) for _ in range(wl["layers"])]
            self.w_pos = np.full(self.w_R, -1, np.int64)   # WavLM frame per ring slot
            self.w_j = -1                                   # the next WavLM frame to finish
            self.w_hist = np.zeros(320, np.float32)         # the last normalised samples
            self.w_esum, self.w_n = 0.0, 0.0                # running sum of squares, count
            self.w_conv = np.zeros((1, wl["pos_taps"] - 1, wl["d"]), np.float32)

    def set_enrollment(self, clip: np.ndarray) -> None:
        """Lock the stream to the voice in ``clip`` (16 kHz mono, the speaker alone, a few
        seconds). The live enrollment stays off, across :meth:`reset` too."""
        if np.asarray(clip).size == 0:
            raise ValueError("the enrollment clip is empty")
        self.clip_emb = self.embed(clip)
        self.emb = self.clip_emb.copy()
        self.live = False

    def embed(self, clip: np.ndarray) -> np.ndarray:
        """The speaker vector of a 16 kHz clip."""
        x = np.nan_to_num(np.asarray(clip, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        S = len(x)
        x = np.pad(x, (self.delay, self.delay + (-S) % self.hop))
        T = (len(x) - self.win) // self.hop + 1
        idx = np.arange(T)[:, None] * self.hop + np.arange(self.win)[None]
        X = np.fft.rfft(x[idx] * self.w, n=self.win)
        logp = np.log((X.real ** 2 + X.imag ** 2) + 1e-8).astype(np.float32)[None]
        mask = (np.arange(T) < min(T, max(1, S // self.hop))).astype(np.float32)[None]
        return self._enroll_sess.run(None, {"logpow": logp, "mask": mask})[0].astype(np.float32)

    def push(self, x: np.ndarray) -> np.ndarray:
        """Audio in (any length, 16 kHz); the isolated audio of every whole hop out."""
        x = np.nan_to_num(np.asarray(x, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        self.pending = np.concatenate([self.pending, x])
        out = []
        while len(self.pending) >= (self.fixed or 1) * self.hop:
            n = min(len(self.pending) // self.hop, self.max_frames)
            if self.live and self.n_in < self.loud_until * _SR and not self.fixed:
                # a run never crosses a half second, where the enrollment may change
                n = min(n, (self.step_n - self.n_in % self.step_n) // self.hop)
            chunk = self.pending[: n * self.hop]
            self.pending = self.pending[n * self.hop:]
            out.append(self._step(chunk))
            if self.live and self.n_in < len(self.hist):
                m = min(len(chunk), len(self.hist) - self.n_in)
                self.hist[self.n_in:self.n_in + m] = chunk[:m]
            self.n_in += len(chunk)
            if self.live and self.n_in % self.step_n == 0:
                self._loud_rule()
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def flush(self) -> np.ndarray:
        """Push enough silence to emit everything pushed so far: the pending samples and the
        last ``delay`` inside the STFT window, plus output for the padding itself (a hop, or
        up to a whole chunk on a fixed-chunk model). The stream stays usable."""
        n = self.delay + self.hop
        if self.fixed:
            n += -(len(self.pending) + n) % (self.fixed * self.hop)
        return self.push(np.zeros(n, np.float32))

    def _loud_rule(self) -> None:
        t = self.n_in / _SR
        if not (_LOUD_S <= t <= self.loud_until):
            return
        m = self.n_in // self.hop
        w = int(min(_LOUD_S, t) * _SR / self.hop)
        if m < w:
            return
        # float32 hop energies and a running sum by convolution, as upstream computes
        # them, so ties pick the same window
        e = (self.hist[: m * self.hop].reshape(m, self.hop) ** 2).mean(1)
        st = int(np.convolve(e, np.ones(w), "valid").argmax())
        if (st, w) != self.loud_st:
            self.emb = self.embed(self.hist[st * self.hop:(st + w) * self.hop])
            self.loud_st = (st, w)

    def _step(self, chunk: np.ndarray) -> np.ndarray:
        hop, win, R = self.hop, self.win, self.R
        C = len(chunk) // hop
        x = np.concatenate([self.inbuf, chunk])
        self.inbuf = x[-self.delay:].copy()
        frames = x[np.arange(C)[:, None] * hop + np.arange(win)[None]] * self.w
        X = np.fft.rfft(frames, n=win)
        p = self.t + np.arange(C)
        ang = np.remainder(p[:, None].astype(np.float64) * self.inv[None], 2.0 * np.pi)
        old = (self.pos[None] >= 0) & (self.pos[None] > p[:, None] - self.window)
        new = np.arange(C)[None] <= np.arange(C)[:, None]
        bias = np.where(np.concatenate([old, new], 1), 0.0, -1e9).astype(np.float32)[None, None]
        feed = {
            "re": X.real.astype(np.float32)[None],
            "im": X.imag.astype(np.float32)[None],
            "emb": self.emb,
            "cos": np.cos(ang).astype(np.float32),
            "sin": np.sin(ang).astype(np.float32),
            "bias": bias,
            **self.states,
        }
        for i in range(self.L):
            feed[f"k{i}"], feed[f"v{i}"] = self.k[i], self.v[i]
        if self.wl:
            feed.update(self._wavlm_feed(chunk))
        outs = self._step_sess.run(None, feed)
        yre, yim, knew, vnew = outs[:4]
        self.states = {
            s["name"]: v for s, v in zip(self.cfg["states"], outs[4:4 + self.n_states])}
        slots = p % R
        for i in range(self.L):
            self.k[i][:, :, slots] = knew[i]
            self.v[i][:, :, slots] = vnew[i]
        self.pos[slots] = p
        self.t += C
        n_out = 4 + self.n_states
        if self.wl:
            self._wavlm_update(*outs[n_out:n_out + 3])
            n_out += 3
        y = np.fft.irfft(yre[0] + 1j * yim[0], n=win).astype(np.float32)
        y *= self.w / self.norm
        acc = np.zeros((C - 1) * hop + win, np.float32)
        acc[: self.delay] += self.outbuf
        for c in range(C):
            acc[c * hop:c * hop + win] += y[c]
        self.outbuf = acc[C * hop:].copy()
        out = acc[: C * hop]
        if self.gate is not None:
            out = self._gate_out(out, x[: C * hop], outs[n_out][0])
        return out

    def _wavlm_feed(self, chunk: np.ndarray) -> dict:
        """The WavLM branch's inputs for one chunk: it finishes one 20 ms frame per two hops.

        Each sample is divided by the RMS of the input up to it (float64), and attention runs
        over the ring of the last ``window - 1`` frames plus the chunk's own. A stream's first
        chunk finishes frame -1, which does not exist: it reads zeros and attends to itself.
        """
        S, window = len(chunk), self.wl["window"]
        k = S // 320
        c = chunk.astype(np.float64)
        e = np.cumsum(c * c) + self.w_esum
        n = np.arange(1, S + 1, dtype=np.float64) + self.w_n
        xn = (c / np.sqrt(e / n + 1e-10)).astype(np.float32)
        self.w_esum, self.w_n = float(e[-1]), float(n[-1])
        wx = np.concatenate([self.w_hist, xn])
        self.w_hist = wx[-320:].copy()
        jj = self.w_j + np.arange(k)
        valid = jj >= 0
        old = (self.w_pos[None] >= 0) & (self.w_pos[None] > jj[:, None] - window)
        new = ((np.arange(k)[None] <= np.arange(k)[:, None]) & valid[None]) | np.eye(k, dtype=bool)
        rel = np.concatenate([self.w_pos[None] - jj[:, None], jj[None] - jj[:, None]], 1)
        self._w_jj = jj
        feed = {
            "wx": wx[None],
            "wvalid": valid.astype(np.float32),
            "wpos": self.w_conv,
            "wrel": rel.astype(np.int64),
            "wmask": np.where(np.concatenate([old, new], 1), 0.0, -1e9).astype(np.float32)[None, None],
        }
        for i in range(self.wl["layers"]):
            feed[f"wk{i}"], feed[f"wv{i}"] = self.wk[i], self.wv[i]
        return feed

    def _wavlm_update(self, wpos: np.ndarray, knew: np.ndarray, vnew: np.ndarray) -> None:
        jj = self._w_jj
        valid = jj >= 0
        slots = jj[valid] % self.w_R
        for i in range(self.wl["layers"]):
            self.wk[i][:, :, slots] = knew[i][:, :, valid]
            self.wv[i][:, :, slots] = vnew[i][:, :, valid]
        self.w_pos[slots] = jj[valid]
        self.w_conv = wpos
        self.w_j = int(jj[-1]) + 1

    def _gate_out(self, out: np.ndarray, line: np.ndarray, logit: np.ndarray) -> np.ndarray:
        """Apply the presence gate to one step's output.

        A closed gate opens at a frame whose probability reaches ``theta``; an open one
        counts a frame at ``theta_off`` or over as the speaker and stays open ``hold``
        frames after the last one. A frame the model kept within ``keep_db`` of the input
        counts as the speaker whatever the head says. Gate edges are shaped by the squared
        window, so a closed run is exact digital silence with no added delay.
        """
        hop, g = self.hop, self.gate
        C = len(out) // hop
        keep = None
        if g["keep_db"] is not None:
            eo = (out.reshape(C, hop) ** 2).mean(1)
            ei = (line.astype(np.float32).reshape(C, hop) ** 2).mean(1)
            q = np.float32(32768.0 ** -2)
            e = np.float32(1e-3) * q
            keep = (ei > q) & (10 * np.log10((eo + e) / (ei + e)) >= -np.float32(g["keep_db"]))
        with np.errstate(over="ignore"):  # a very sure "no" overflows exp: p = 0
            z = np.exp(-np.asarray(logit, np.float32))
            prob = (np.float32(1.0) / (np.float32(1.0) + z)).astype(np.float32)
        o = np.empty(C, np.float32)
        since, hold = self.g_since, g["hold"]
        is_open = since <= hold
        for t in range(C):
            heard = prob[t] >= (self.g_th_off if is_open else self.g_th)
            if keep is not None:
                self.g_krun = self.g_krun + 1 if keep[t] else 0
                heard = heard or self.g_krun >= 1
            since = 0 if heard else since + 1
            is_open = since <= hold
            o[t] = 1.0 if is_open else 0.0
        self.g_since = since
        gp = np.concatenate([[self.g_prev], o[:-1]]).astype(np.float32)
        self.g_prev = o[-1]
        env = gp[:, None] * self.w2[hop:] + o[:, None] * self.w2[:hop]
        return out * env.reshape(-1)


class VoiceIsolatorAdapter(Denoiser):
    """Scicom's voice isolator: keep one speaker, remove other voices and noise (16 kHz).

    Parameters
    ----------
    model:
        ``"release"`` (default) keeps the most of the speaker's words; ``"agent"`` adds a
        presence gate that outputs digital silence where the speaker is not talking.
    quantized:
        Use the 8-bit export (28 MB instead of 89 MB).
    enrollment:
        A clip of the target speaker alone — a file path, or a 16 kHz array (mono or
        ``(n, channels)``, any dtype ``denoise`` accepts). When omitted the speaker is
        enrolled live from the loudest 3 s of the input.
    live_enroll:
        Override the model's live-enrollment setting; ``False`` runs on the model's
        "no enrollment" vector throughout.
    gate:
        ``"auto"`` (default) takes the model's own setting (on for ``agent``, none for
        ``release``); ``None`` or ``"off"`` disables it; or a ``"THETA:HOLD:THETA_OFF:KEEP_DB"``
        spec. A gate needs ``model="agent"``.
    num_threads:
        onnxruntime intra-op threads. Default: one for the CPU release, whose step runs at
        most 8 frames, and every core for the 555 M model.
    model_dir:
        A local directory holding ``config.json`` and its graphs, instead of downloading —
        for example the 555 M model written by ``conversion/export_voiceisolator.py``.
    providers:
        onnxruntime execution providers (default CPU).
    cache_dir:
        Override the model download cache directory.
    """

    input_sample_rate = _SR

    def __init__(
        self,
        *,
        model: str = _DEFAULT_MODEL,
        quantized: bool = False,
        enrollment: Union[str, np.ndarray, None] = None,
        live_enroll: Optional[bool] = None,
        gate: Optional[str] = "auto",
        num_threads: Optional[int] = None,
        model_dir: Optional[str] = None,
        providers: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
        revision: Optional[str] = None,
        **cfg,
    ):
        super().__init__(**cfg)
        if model not in _MODELS:
            raise ValueError(
                f"unknown voiceisolator model {model!r}; choose one of {sorted(_MODELS)}")
        self._model = model
        self._quantized = quantized
        self._enrollment = enrollment
        self._live_enroll = live_enroll
        self._gate = gate
        self._num_threads = num_threads
        self._model_dir = model_dir
        self._providers = providers
        self._cache_dir = cache_dir
        self._revision = revision if revision is not None else _HF_REVISION
        self._cfg_json: Optional[dict] = None
        self._step = None
        self._enroll = None

    def _file(self, name: str) -> str:
        if self._model_dir:
            return resolve(f"{self._model_dir.rstrip('/')}/{name}")
        sub = _MODELS[self._model] + ("int8/" if self._quantized else "")
        return resolve(sub + name, hf_repo=_HF_REPO, revision=self._revision,
                       cache_dir=self._cache_dir)

    def _ensure_models(self) -> None:
        if self._step is not None:
            return
        import onnxruntime as ort

        with open(self._file("config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self._num_threads or (
            (os.cpu_count() or 1) if cfg.get("wavlm") else 1)
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        providers = self._providers or ["CPUExecutionProvider"]
        self._step = ort.InferenceSession(
            self._file(cfg["step_file"]), sess_options=opts, providers=providers)
        self._enroll = ort.InferenceSession(
            self._file(cfg["enroll_file"]), sess_options=opts, providers=providers)
        self._cfg_json = cfg

    def _load_clip(self, clip: Union[str, np.ndarray]) -> np.ndarray:
        """A file path, or a 16 kHz array of any dtype or channel layout ``denoise`` accepts."""
        if isinstance(clip, str):
            x, sr = read_wav(clip)
            if sr != _SR:
                x = kaiser_resample(x, sr, _SR)
            return np.asarray(x, np.float32)
        return to_float32_mono(np.asarray(clip))

    def stream(self, enrollment: Union[str, np.ndarray, None] = None) -> VoiceIsolatorStream:
        """A fresh push-based stream for live 16 kHz audio.

        ``enrollment`` overrides the adapter's own; with neither, the stream enrolls live.
        """
        self._ensure_models()
        s = VoiceIsolatorStream(
            self._cfg_json, self._step, self._enroll,
            live_enroll=self._live_enroll, gate=self._gate)
        clip = enrollment if enrollment is not None else self._enrollment
        if clip is not None:
            s.set_enrollment(self._load_clip(clip))
        return s

    def _denoise_array(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sample_rate != _SR:
            x = np.asarray(kaiser_resample(x, sample_rate, _SR), dtype=np.float32)
        if x.size == 0:
            return np.zeros(0, dtype=np.float32)
        s = self.stream()
        parts = [s.push(x[i:i + _CHUNK]) for i in range(0, x.size, _CHUNK)]
        parts.append(s.flush())
        out = np.concatenate(parts)[s.delay:s.delay + x.size]
        return np.ascontiguousarray(out, dtype=np.float32)


register_engine(EngineEntry(
    alias="voiceisolator",
    adapter_class=VoiceIsolatorAdapter,
    description=(
        "Scicom voice isolator: keeps one speaker and removes other voices and noise "
        "(target-speaker extraction), 16 kHz, causal and streaming with 10 ms delay. "
        "22 M-param CPU distillation; live enrollment from the loudest 3 s or from a clip; "
        "'agent' model adds a presence gate; int8 export. ONNX from "
        "Scicom-intl/voice-isolator-cpu (Zolkepli, arXiv:2610.35272; license: other)"
    ),
    input_sample_rate=_SR,
    output_sample_rate=_SR,
    license="other (no license text published upstream)",
    extras="",
    kind="denoise",
))
