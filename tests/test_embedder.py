"""Offline check of the embedding contract (no model download): last real token, final norm output,
L2-normalised, tail truncation, caching."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from clm_tune_mlx.embedder import MLXEmbedder


class FakeInner:
    def __call__(self, ids, cache=None):  # hidden state of token t = [t, 1, 0, ...]
        b, n = ids.shape
        h = mx.zeros((b, n, 4))
        return h + mx.stack([ids.astype(mx.float32), mx.ones_like(ids).astype(mx.float32)] + [mx.zeros_like(ids).astype(mx.float32)] * 2, axis=-1)


class FakeModel:
    model = FakeInner()


class FakeTok:
    def encode(self, text):
        return [len(w) for w in text.split()]


def make(max_tokens=2048):
    e = MLXEmbedder.__new__(MLXEmbedder)
    e.model, e.tokenizer, e.max_tokens, e.cache_size, e.cache, e.batch_tokens = FakeModel(), FakeTok(), max_tokens, 10, {}, 4096
    e.min_prefix = 10_000  # the fake model has no KV cache; prefix sharing is checked on the real model
    return e


def test_last_token_normalised():
    v, n = make().embed_one("aa bbbb ccc")        # tokens 2, 4, 3 -> last is 3
    assert n == 3 and np.allclose(v, np.array([3, 1, 0, 0]) / np.sqrt(10))


def test_tail_truncation_keeps_last_tokens():
    v, n = make(max_tokens=2).embed_one("aaaaa bb c")
    assert n == 2 and np.allclose(v, np.array([1, 1, 0, 0]) / np.sqrt(2))


def test_embed_caches_and_orders():
    e = make()
    out, tokens = e.embed(["a", "bb", "a"])
    assert out.shape == (3, 4) and np.allclose(out[0], out[2]) and tokens == 2 and len(e.cache) == 2


def test_batched_embed_matches_one_by_one():
    e = make()
    texts = ["aa bbbb ccc", "x", "hello there general kenobi"]
    batch, _ = e.embed(texts)                       # right-padded batch, gathered at each row's length
    for t, v in zip(texts, batch):
        assert np.allclose(v, make().embed_one(t)[0])
