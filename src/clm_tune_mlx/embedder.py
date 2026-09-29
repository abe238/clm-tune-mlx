"""MLX drop-in for upstream CLM's vLLM embedder.

Upstream CLM (`pip install contrastive-lm`) scores a (state, option) pair as the scaled cosine of
two small projection heads applied to Qwen3-8B last-token embeddings, which it gets from a vLLM
pooling server. This module computes the same embedding in MLX, so the unchanged upstream Engine,
schema and released heads run on a Mac with no server:

    from clm_tune_mlx import load_engine
    engine = load_engine()
    engine.answer({"ticket": "Charged twice"}, {"money": {"type": "noul", "instructions": "Is `ticket` about money?"}})
"""
from __future__ import annotations

import time

import numpy as np

DEFAULT_ENCODER = "mlx-community/Qwen3-8B-8bit"   # measured min cosine 0.9976 vs PyTorch Qwen/Qwen3-8B
MAX_TOKENS = 2048                                 # the truncation upstream asks vLLM for


class MLXEmbedder:
    """Upstream `Embedder` contract: embed(texts) -> ([n, 4096] L2-normalised float32, tokens spent)."""

    def __init__(self, encoder: str = DEFAULT_ENCODER, max_tokens: int = MAX_TOKENS, cache_size: int = 50_000,
                 batch_tokens: int = 4096):
        from mlx_lm import load
        t = time.time()
        self.model, self.tokenizer = load(encoder)
        self.load_seconds = time.time() - t
        self.max_tokens, self.cache_size, self.batch_tokens = max_tokens, cache_size, batch_tokens
        self.min_prefix = 64   # measured on M5 Pro: below ~60 shared tokens plain batching is faster
        if hasattr(self.model, "language_model"):   # Qwen3.5-family hybrid: linear-attention layers keep no KV cache to share
            self.min_prefix = 10**9
        self.cache: dict[str, np.ndarray] = {}

    def _ids(self, text: str) -> list[int]:
        # Tail kept; Qwen3's tokenizer adds no BOS/EOS, same as vLLM's default.
        return self.tokenizer.encode(text)[-self.max_tokens:] or self.tokenizer.encode(" ")

    @property
    def _body(self):
        """The text model whose output is the final-norm hidden states (nested one level deeper in Qwen3.5-family models)."""
        return getattr(self.model, "language_model", self.model).model

    def _forward(self, rows: list[list[int]], prefix: list[int] | None = None) -> np.ndarray:
        """Last-token, final-norm hidden states for a right-padded batch. Attention is causal, so
        padding after a row's last real token cannot change it: gather each row at its own length.
        With `prefix`, the shared prefix runs once and every row continues from its KV cache."""
        import mlx.core as mx
        cache = None
        if prefix:
            from mlx_lm.models.cache import make_prompt_cache
            cache = make_prompt_cache(self.model)
            self._body(mx.array([prefix]), cache=cache)
            for c in cache:                                         # share the prefix across the batch
                c.keys = mx.repeat(c.keys[..., :c.offset, :], len(rows), axis=0)
                c.values = mx.repeat(c.values[..., :c.offset, :], len(rows), axis=0)
        n = max(len(r) for r in rows)
        ids = mx.array([r + [r[-1]] * (n - len(r)) for r in rows])
        h = self._body(ids, cache=cache)                      # [b, n, hidden], after the final norm
        last = h[mx.arange(len(rows)), mx.array([len(r) - 1 for r in rows])]
        v = np.array(last.astype(mx.float32))
        return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)

    def _shared_prefix(self, rows: list[list[int]]) -> int:
        """Length of the token prefix every row shares, leaving each row at least one token."""
        k = min(len(r) for r in rows) - 1
        for i in range(k):
            if any(r[i] != rows[0][i] for r in rows):
                return i
        return max(k, 0)

    def embed_one(self, text: str) -> tuple[np.ndarray, int]:
        ids = self._ids(text)
        return self._forward([ids])[0], len(ids)

    def embed(self, texts: list[str]) -> tuple[np.ndarray, int]:
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        ids = {t: self._ids(t) for t in todo}
        tokens = 0
        # Several questions about one state share the state as a token prefix: run it once.
        groups: dict[tuple, list[str]] = {}
        for t in todo:
            groups.setdefault(tuple(ids[t][:self.min_prefix]), []).append(t)
        for g in groups.values():
            k = self._shared_prefix([ids[t] for t in g]) if len(g) > 1 else 0
            if k >= self.min_prefix and sum(len(ids[t]) - k for t in g) <= self.batch_tokens:
                for t, v in zip(g, self._forward([ids[t][k:] for t in g], prefix=ids[g[0]][:k])):
                    self._put(t, v)
                    tokens += len(ids[t])
                todo = [t for t in todo if t not in self.cache]
        todo.sort(key=lambda t: len(ids[t]))                        # similar lengths per batch: little padding
        i = 0
        while i < len(todo):
            j = i + 1                                               # grow the batch up to batch_tokens of padded work
            while j < len(todo) and len(ids[todo[j]]) * (j - i + 1) <= self.batch_tokens:
                j += 1
            chunk = todo[i:j]
            for t, v in zip(chunk, self._forward([ids[t] for t in chunk])):
                self._put(t, v)
                tokens += len(ids[t])
            i = j
        return np.stack([self.cache[t] for t in texts]), tokens

    def _put(self, text: str, vec: np.ndarray) -> None:
        if len(self.cache) >= self.cache_size:
            self.cache.pop(next(iter(self.cache)))   # ponytail: FIFO eviction, LRU if hit rates ever matter
        self.cache[text] = vec

    def healthy(self) -> bool:
        return True


def load_engine(encoder: str = DEFAULT_ENCODER, checkpoint: str | None = None, backend: str = "mlx"):
    """CLM with the MLX encoder. backend="mlx" (default) runs the heads in MLX too; backend="torch" uses
    the upstream PyTorch engine (the reference). Downloads the released head if no checkpoint is given."""
    if backend == "mlx":
        from .engine import MLXEngine, default_head
        return MLXEngine(MLXEmbedder(encoder), checkpoint or default_head())
    if checkpoint is None:
        from clm.heads import download   # needs the [torch] extra
        checkpoint = download()
    from clm.engine import Engine
    return Engine(embedder=MLXEmbedder(encoder), checkpoint=checkpoint, device="cpu", action_cache="0")
