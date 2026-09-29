"""CLM inference entirely in MLX: encoder, projection heads, scoring.

The heads mirror upstream `clm.heads.make_head` (hidden -> width -> [width, LayerNorm] x (depth-2) -> proj,
exact GELU, optional residual). By default the released head is fetched already converted to
safetensors (no PyTorch needed); a `.pt` checkpoint is converted once with PyTorch and cached.
Question rendering and answer math are upstream's `clm/schema.py` (vendored), so answers match.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from .schema import answer_from_logits, build_pairs

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "clm_tune_mlx")
# Contrastive-LM/CLM-v0.1-8B CLM_v0.1-8B.pt (sha256 b2b4a8c9...) converted to safetensors, weights unchanged.
HEAD_URL = "https://github.com/abe238/clm-tune-mlx/releases/download/v0.3.0/"
HEAD_FILES = {"CLM_v0.1-8B.safetensors": "5aa89d237523e423fc89f8c56f125a149c1cc2151c395842dc63d3bf6cd15151",
              "CLM_v0.1-8B.json": "ecc83670100e43601d4f5fa6e3547b435911a98244cb2307097fcdb877b83a1e"}


def default_head() -> str:
    """Download (once) and sha256-verify the released head as safetensors; returns its path."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    for name, sha in HEAD_FILES.items():
        dest = os.path.join(CACHE_DIR, name)
        if not (os.path.exists(dest) and hashlib.sha256(open(dest, "rb").read()).hexdigest() == sha):
            data = urllib.request.urlopen(HEAD_URL + name, timeout=300).read()
            if hashlib.sha256(data).hexdigest() != sha:
                raise RuntimeError(f"{name}: sha256 mismatch, refusing to use it")
            open(dest, "wb").write(data)
    return os.path.join(CACHE_DIR, "CLM_v0.1-8B.safetensors")


class Head(nn.Module):
    def __init__(self, hidden: int, width: int, depth: int, proj: int, layernorm: bool, residual: bool):
        super().__init__()
        self.inp = nn.Linear(hidden, width)
        self.hidden = [nn.Linear(width, width) for _ in range(depth - 2)]
        self.norms = [nn.LayerNorm(width) if layernorm else nn.Identity() for _ in range(depth - 2)]
        self.out = nn.Linear(width, proj)
        self.residual = residual

    def __call__(self, x):
        x = nn.gelu(self.inp(x))
        for lin, nrm in zip(self.hidden, self.norms):
            h = nn.gelu(nrm(lin(x)))
            x = x + h if self.residual else h
        return self.out(x)


def _cached_safetensors(path: str) -> tuple[str, dict]:
    """.pt -> (safetensors path, meta); converts once with PyTorch, then reads the cache."""
    if path.endswith(".safetensors"):
        return path, json.load(open(path[: -len(".safetensors")] + ".json"))
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
    st, meta_p = os.path.join(CACHE_DIR, f"{digest}.safetensors"), os.path.join(CACHE_DIR, f"{digest}.json")
    if not os.path.exists(st):
        import torch  # one-time conversion only
        ck = torch.load(path, map_location="cpu", weights_only=True)
        cfg = dict(ck["cfg"])
        meta = {"width": cfg["width"], "depth": cfg["depth"], "activation": cfg.get("activation", "gelu"),
                "proj": ck.get("projection_dim", cfg.get("projection_dim", 512)), "layernorm": cfg.get("layernorm", False),
                "residual": cfg.get("residual", False), "hidden": cfg.get("hidden_size", 4096),
                "logit_scale": float(torch.as_tensor(ck["logit_scale"]).float())}
        if meta["activation"] != "gelu":
            raise ValueError(f"unsupported activation {meta['activation']!r}")
        arrays = {f"{side}.{k}": mx.array(v.float().numpy()) for side in ("state", "action") for k, v in ck[f"{side}_head"].items()}
        os.makedirs(CACHE_DIR, exist_ok=True)
        mx.save_safetensors(st, arrays)
        json.dump(meta, open(meta_p, "w"))
    return st, json.load(open(meta_p))


class Heads:
    def __init__(self, checkpoint: str):
        st, m = _cached_safetensors(checkpoint)
        kw = dict(hidden=m["hidden"], width=m["width"], depth=m["depth"], proj=m["proj"], layernorm=m["layernorm"], residual=m["residual"])
        self.state, self.action = Head(**kw), Head(**kw)
        w = mx.load(st)
        for side, head in (("state", self.state), ("action", self.action)):
            head.load_weights([(k[len(side) + 1:], v) for k, v in w.items() if k.startswith(side + ".")], strict=True)
        self.scale = float(min(np.exp(m["logit_scale"]), 100.0))
        self.path, self.meta = checkpoint, m
        # optional input centering (train_heads(center=True)); absent in older files = no-op
        self.means = {s: np.asarray(w[f"center.{s}"]) for s in ("state", "action") if f"center.{s}" in w}

    def project(self, x, side: str):
        if side in self.means:                                   # subtract the train mean, then L2-renormalise
            x = x - mx.array(self.means[side])
            x = x / mx.maximum(mx.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
        z = (self.state if side == "state" else self.action)(x)
        return z / mx.maximum(mx.linalg.norm(z, axis=-1, keepdims=True), 1e-12)


class MLXEngine:
    """Drop-in for upstream `clm.Engine.answer` with the heads in MLX. Option projections are cached."""

    def __init__(self, embedder, checkpoint: str, cache_size: int = 50_000):
        self.embedder, self.heads = embedder, Heads(checkpoint)
        self.cache: dict[str, mx.array] = {}
        self.cache_size = cache_size

    def _project(self, texts: list[str], side: str):
        if side == "state":
            e, _ = self.embedder.embed(texts)
            return self.heads.project(mx.array(e), "state")
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if todo:
            e, _ = self.embedder.embed(todo)
            z = self.heads.project(mx.array(e), "action")
            mx.eval(z)
            for t, v in zip(todo, z):
                if len(self.cache) >= self.cache_size:
                    self.cache.pop(next(iter(self.cache)))  # ponytail: FIFO eviction
                self.cache[t] = v
        return mx.stack([self.cache[t] for t in texts])

    def answer(self, state, questions: dict, model: str | None = None, temperature: float = 1.0) -> dict:
        if not questions:
            raise ValueError("questions must not be empty")
        pairs = build_pairs(state, questions)
        zs = self._project([p[0] for p in pairs.values()], "state")
        za = self._project([t for p in pairs.values() for t in p[2]], "action")
        answers, k = {}, 0
        for i, (qid, (_, keys, texts)) in enumerate(pairs.items()):
            cos = za[k:k + len(texts)] @ zs[i]
            k += len(texts)
            answers[qid] = answer_from_logits(questions[qid], keys, (self.heads.scale * cos / temperature).tolist())
        return {"model": "clm-latest", "answers": answers, "usage": {"billing_units": len(questions), "output_tokens": 0}}

    def models(self) -> list[dict]:
        return [{"name": "clm-latest", "description": f"CLM heads in MLX ({os.path.basename(self.heads.path)})"}]
