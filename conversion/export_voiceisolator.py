#!/usr/bin/env python3
"""Export Scicom's 555 M voice isolator (the GPU release) to ONNX: ``enroll.onnx`` + ``step.onnx``.

    python conversion/export_voiceisolator.py --out out/release
    python conversion/export_voiceisolator.py --sub agent --out out/agent

The model is a FrameFormer (frame transformer over 10 ms frames, band and frame LSTMs) with
WavLM-Large made causal beside it on 20 ms frames. Upstream serves it with PyTorch and CUDA
graphs; its own ONNX exporter is not published, so this script writes the same graph
contract as the published CPU release (``Scicom-intl/voice-isolator-cpu``), extended with the
WavLM branch's state.

``step.onnx`` runs a fixed ``--chunk`` frames (even: the WavLM branch finishes one 20 ms frame
per two 10 ms frames). Everything a stream keeps is an input and an output:

    re, im [1, C, 161]          STFT of the C new frames (sqrt-Hann, 320/160)
    emb [1, 256]                speaker vector
    cos, sin [C, 32]            rotary angles of the C frame positions
    bias [1, 1, C, 999 + C]     additive mask: the ring of 999 past frames, then the new ones
    k0..k15, v0..v15            the FrameFormer's rotated keys and values, [1, 16, 999, 64]
    hb0, cb0, hb1, cb1, hm0, cm0  LSTM states
    wx [1, 320 + 160 C]         RMS-normalised samples: the previous 320, then the new ones
    wvalid [C/2]                1 for a WavLM frame that exists (the first chunk's first does not)
    wpos [1, 64, 1024]          the positional convolution's last 64 input frames
    wrel [C/2, 499 + C/2] int64 relative positions (key - query) for WavLM's position bias
    wmask [1, 1, C/2, 499 + C/2] additive mask over the WavLM ring and the new frames
    wk0..wk23, wv0..wv23        WavLM's keys and values, [1, 16, 499, 64]
  -> yre, yim [1, C, 161], knew, vnew [16, 1, 16, C, 64], next LSTM states,
     wpos_out, wknew, wvnew [24, 1, 16, C/2, 64] (, gate [1, C] for the agent model)

The ring writes, the masks, the rotary angles, the STFT and overlap-add and WavLM's running
RMS normalisation (float64) stay outside the graph, in the runtime.

Upstream's own modules are imported from the downloaded repository; the step reuses their
layers and re-wires only what ONNX cannot express: complex arithmetic becomes real and
imaginary parts, and attention reads a ring passed in instead of writing one in place.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 16000


def load_upstream(repo: str, sub: str, revision: str | None):
    from huggingface_hub import snapshot_download

    pat = [f"{sub}/*"] if sub else ["config.json", "model.safetensors"]
    root = Path(snapshot_download(repo, revision=revision, allow_patterns=pat + ["vi/*"]))
    sys.path.insert(0, str(root))
    from vi.infer import load_model  # noqa: E402

    d = root / sub if sub else root
    model = load_model(str(d), device="cpu")
    cfg = json.loads((d / "config.json").read_text())
    return model, cfg, root


def rotate(x, cos, sin):
    """Upstream's ``_rot``: interleaved pairs rotated by cos/sin (C, hd/2)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


def ring_attend(q, k_ring, k_new, v_ring, v_new, bias):
    """softmax(q K^T / sqrt(hd) + bias) V over [ring, new] keys; q (1, h, n, hd)."""
    K = torch.cat([k_ring, k_new], 2)
    V = torch.cat([v_ring, v_new], 2)
    s = torch.matmul(q, K.transpose(-1, -2)) * (q.shape[-1] ** -0.5) + bias
    return torch.matmul(torch.softmax(s, -1), V)


class Step(nn.Module):
    def __init__(self, m, chunk: int):
        super().__init__()
        # even: one WavLM frame per two hops; dividing 50: the runtime steps land on the
        # live enrollment's half seconds
        if chunk % 2 or 50 % chunk:
            raise ValueError(f"--chunk must be even and divide 50, got {chunk}")
        assert not getattr(m, "mask_floor", 0.0) and not getattr(m, "mask_shrink", 0.0)
        assert m.bg_mix is None and m.btime is not None and m.ftime is not None
        self.m, self.C, self.k = m, chunk, chunk // 2
        self.wl = m._ssl_encoder()
        self.L, self.WL = len(m.tblocks), len(self.wl.layers)
        self.n_btime, self.n_ftime = len(m.btime), len(m.ftime)

    # ---- the isolator's band features and mask, in real arithmetic ------------------------
    def features(self, re, im):
        m = self.m
        mag = torch.sqrt(re * re + im * im)
        s = mag.clamp(min=1e-8) ** (m.compress - 1)
        zero = re.new_zeros(re.shape[:-1] + (1,))
        rp, ip = torch.cat([re * s, zero], -1), torch.cat([im * s, zero], -1)
        gr, gi = rp[..., m.gidx], ip[..., m.gidx]                       # (1, C, K, bw)
        pw = (re * re + im * im)[..., m.gidx.clamp(max=m.F - 1)]
        le = torch.log(pw.mul(m.gvalid).sum(-1) / m.gvalid.sum(-1) + 1e-8) / 10
        f = torch.cat([gr, gi, le[..., None]], -1)
        return torch.einsum("btkf,kfn->btkn", f, m.w_in) + m.b_in

    def mask(self, z, re, im):
        m = self.m
        u = torch.einsum("btkn,knm->btkm", m.m_norm(z), m.w_m1) + m.b_m1
        u = torch.einsum("btkm,kmo->btko", torch.tanh(u), m.w_m2) + m.b_m2
        B, T = u.shape[:2]
        u = u.reshape(B, T, m.K, 2, 2, m.bw)
        mk = u[:, :, :, 0] * torch.sigmoid(u[:, :, :, 1])
        mk = mk.permute(0, 1, 3, 2, 4).reshape(B, T, 2, m.K * m.bw).index_select(-1, m.gsel)
        mr, mi = mk[:, :, 0], mk[:, :, 1]
        return mr * re - mi * im, mr * im + mi * re

    # ---- WavLM branch -----------------------------------------------------------------------
    def wavlm(self, wx, wvalid, wpos, wrel, wmask, wk, wv):
        from vi.causal_wavlm import rel_bucket

        wl = self.wl
        S = self.k * 320
        f = wl.cnn(wx[:, : S + 80]) * wvalid[None, :, None]           # (1, k, d)
        xp = torch.cat([wpos, f], 1)
        wpos_out = xp[:, -(wl.pos_taps - 1):]
        x = f + F.gelu(wl.pos(xp.transpose(1, 2))).transpose(1, 2)
        bias = wl.rel(rel_bucket(wrel, wl.nb, wl.md)).permute(2, 0, 1)[None]   # (1, h, k, R+k)
        hs, knew, vnew = [x], [], []
        for i, layer in enumerate(wl.layers):
            a = layer.att
            h = layer.n1(x)
            q, kk, vv = a._heads(h)
            am = a.gates(h) * bias + wmask
            y = ring_attend(q, wk[i], kk, wv[i], vv, am)
            B, n, D = h.shape
            x = x + a.o(y.transpose(1, 2).reshape(B, n, D))
            x = x + layer.ff2(F.gelu(layer.ff1(layer.n2(x))))
            hs.append(x)
            knew.append(kk)
            vnew.append(vv)
        hs[-1] = wl.norm(x)
        m = self.m
        extra = m.ssl_gate * m.ssl_proj(
            (m._ssl_mixed(hs) * wvalid[None, :, None]).repeat_interleave(2, dim=1))
        return extra, wpos_out, torch.stack(knew), torch.stack(vnew)

    # ---- one streaming step -------------------------------------------------------------------
    def forward(self, re, im, emb, cos, sin, bias, *rest):
        m, L, WL = self.m, self.L, self.WL
        k, v = rest[:L], rest[L:2 * L]
        rest = rest[2 * L:]
        n_states = 2 * (self.n_btime + self.n_ftime)
        states, rest = rest[:n_states], rest[n_states:]
        wx, wvalid, wpos, wrel, wmask = rest[:5]
        wk, wv = rest[5:5 + WL], rest[5 + WL:5 + 2 * WL]

        extra, wpos_out, wknew, wvnew = self.wavlm(wx, wvalid, wpos, wrel, wmask, wk, wv)

        z = m.band_in(self.features(re, im))
        B, T, K, N = z.shape
        out_states = list(states)   # replaced slot by slot, so they leave in input order

        def btime(i, z):
            h0, c0 = states[2 * i], states[2 * i + 1]
            z, (h1, c1) = m.btime[i](z, emb, (h0.reshape(1, K, -1), c0.reshape(1, K, -1)))
            out_states[2 * i], out_states[2 * i + 1] = h1.reshape(h0.shape), c1.reshape(c0.shape)
            return z

        z = btime(0, z)
        x = m.down(z.reshape(B, T, K * N)) + extra
        knew, vnew = [], []
        for i, blk in enumerate(m.tblocks):
            g, b = blk.film(emb).chunk(2, -1)
            x = x * (1 + g[:, None]) + b[:, None]
            a = blk.att
            q, kk, vv = a.qkv(blk.n1(x)).view(B, T, 3, a.h, a.hd).permute(2, 0, 3, 1, 4)
            qr, kr = rotate(q, cos, sin), rotate(kk, cos, sin)
            y = ring_attend(qr, k[i], kr, v[i], vv, bias)
            x = x + a.o(y.transpose(1, 2).reshape(B, T, -1))
            h = blk.n2(x)
            x = x + blk.w2(F.silu(blk.w1(h)) * blk.w3(h))
            knew.append(kr)
            vnew.append(vv)
            if i in m.mem_at:
                j = m.mem_at.index(i)
                s0 = 2 * (self.n_btime + j)
                x, (h1, c1) = m.ftime[j](x, emb, (states[s0].reshape(1, 1, -1),
                                                  states[s0 + 1].reshape(1, 1, -1)))
                out_states[s0], out_states[s0 + 1] = (h1.reshape(states[s0].shape),
                                                      c1.reshape(states[s0 + 1].shape))
        xn = m.t_norm_out(x)
        z = z + m.up(xn).reshape(B, T, K, N)
        z = btime(1, z)
        yre, yim = self.mask(m.band_out(z), re, im)
        outs = [yre, yim, torch.stack(knew), torch.stack(vnew), *out_states, wpos_out, wknew, wvnew]
        if m.gate_head is not None:
            outs.append(m.gate_head(xn).squeeze(-1))
        return tuple(outs)


class Enroll(nn.Module):
    """``EnrollEncoder`` from log-power frames (the runtime computes them; ONNX has no complex)."""

    def __init__(self, m):
        super().__init__()
        self.enc = m.enroll_enc

    def forward(self, logpow, mask):
        e = self.enc
        x = logpow.transpose(1, 2)
        x = x - (x * mask[:, None]).sum(-1, keepdim=True) / mask.sum(-1).clamp(min=1)[:, None, None]
        h = e.net(x)
        a = e.att(h).masked_fill(mask[:, None] < 0.5, -1e4)
        a = torch.softmax(a, dim=-1)
        mu = (a * h).sum(-1)
        sd = ((a * h * h).sum(-1) - mu * mu).clamp(min=1e-6).sqrt()
        return e.out(torch.cat([mu, sd], -1))


def io_spec(m, step: Step):
    """(input names, example inputs, output names, LSTM state specs) of ``step.onnx``."""
    att, wl = m.tblocks[0].att, step.wl
    wa = wl.layers[0].att
    C, k = step.C, step.k
    R, WR = att.window - 1, wl.window - 1
    states = []
    for i in range(len(m.btime)):
        states += [(f"hb{i}", [1, m.K, m.band_rnn]), (f"cb{i}", [1, m.K, m.band_rnn])]
    for j in range(len(m.ftime)):
        states += [(f"hm{j}", [1, 1, m.mem_rnn]), (f"cm{j}", [1, 1, m.mem_rnn])]
    z = torch.zeros
    spec = [("re", z(1, C, m.F)), ("im", z(1, C, m.F)), ("emb", m.null[None].detach().clone()),
            ("cos", torch.ones(C, att.hd // 2)), ("sin", z(C, att.hd // 2)), ("bias", z(1, 1, C, R + C))]
    spec += [(f"k{i}", z(1, att.h, R, att.hd)) for i in range(step.L)]
    spec += [(f"v{i}", z(1, att.h, R, att.hd)) for i in range(step.L)]
    spec += [(n, z(*shape)) for n, shape in states]
    spec += [("wx", z(1, 320 + 160 * C)), ("wvalid", torch.ones(k)), ("wpos", z(1, wl.pos_taps - 1, wl.d)),
             ("wrel", torch.zeros(k, WR + k, dtype=torch.long)), ("wmask", z(1, 1, k, WR + k))]
    spec += [(f"wk{i}", z(1, wa.h, WR, wa.hd)) for i in range(step.WL)]
    spec += [(f"wv{i}", z(1, wa.h, WR, wa.hd)) for i in range(step.WL)]
    out_names = ["yre", "yim", "knew", "vnew"] + [f"{n}_out" for n, _ in states] + \
        ["wpos_out", "wknew", "wvnew"] + (["gate"] if m.gate_head is not None else [])
    return [n for n, _ in spec], [t for _, t in spec], out_names, states


def runtime_config(m, step: Step, states, up_cfg: dict, source: str) -> dict:
    """The ``config.json`` the audiosronnx runtime reads."""
    att, wl = m.tblocks[0].att, step.wl
    wa = wl.layers[0].att
    cfg = {
        "win": m.stft.win, "hop": m.stft.hop, "window": att.window, "layers": step.L,
        "heads": att.h, "head_dim": att.hd, "rope_base": 10000.0,
        "null": m.null.detach().tolist(),
        "states": [{"name": n, "shape": s} for n, s in states],
        "step_file": "step.onnx", "enroll_file": "enroll.onnx",
        "chunk_frames": step.C,
        "wavlm": {"layers": step.WL, "heads": wa.h, "head_dim": wa.hd, "window": wl.window,
                  "d": wl.d, "pos_taps": wl.pos_taps},
        "source": source,
        "model": up_cfg["model"],
    }
    if m.gate_head is not None:
        cfg["gate_out"] = True
    if up_cfg.get("serve"):
        cfg["serve"] = up_cfg["serve"]
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="Scicom-intl/voice-isolator")
    ap.add_argument("--revision", default="d41d6d8166a6566c9c2df7ca17e9ca38abcd35d6")
    ap.add_argument("--sub", default="", help="'agent' for the agent model")
    ap.add_argument("--chunk", type=int, default=50, help="frames per step: 2, 10 or 50")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    m, up_cfg, _ = load_upstream(args.repo, args.sub, args.revision)
    m.float().eval()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    step = Step(m, args.chunk).eval()
    names, inputs, out_names, states = io_spec(m, step)
    # the band mixer's nn.MultiheadAttention: its fused inference kernel has no ONNX symbolic
    torch.backends.mha.set_fastpath_enabled(False)
    with torch.no_grad(), tempfile.TemporaryDirectory(dir=out) as tmp:
        # the exporter writes one external-data file per weight: gather them into one
        torch.onnx.export(step, tuple(inputs), f"{tmp}/step.onnx", input_names=names,
                          output_names=out_names, opset_version=17, do_constant_folding=False,
                          dynamo=False)
        onnx.save(onnx.load(f"{tmp}/step.onnx"), str(out / "step.onnx"), save_as_external_data=True,
                  all_tensors_to_one_file=True, location="step.onnx.data", size_threshold=1024)
    with torch.no_grad():
        torch.onnx.export(Enroll(m).eval(), (torch.zeros(1, 300, m.F), torch.ones(1, 300)),
                          str(out / "enroll.onnx"), input_names=["logpow", "mask"],
                          output_names=["emb"], opset_version=17,
                          dynamic_axes={"logpow": {1: "T"}, "mask": {1: "T"}}, dynamo=False)
    source = f"{args.repo}@{args.revision}" + (f"/{args.sub}" if args.sub else "")
    cfg = runtime_config(m, step, states, up_cfg, source)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
