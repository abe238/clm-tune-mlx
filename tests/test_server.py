"""Micro-batching: requests that queue while the model is busy share one encoder pass."""
import threading
import time

import pytest

pytest.importorskip("mlx.core")
from clm_tune_mlx.server import Batcher


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(len(texts))
        time.sleep(0.05)                       # a busy model, so a burst queues up
        return None, 0


class FakeEngine:
    def __init__(self):
        self.embedder = FakeEmbedder()

    def answer(self, state, questions, **kw):
        from clm_tune_mlx.schema import build_pairs
        build_pairs(state, questions)          # raises ValueError on a malformed question, like the real engine
        return {"answers": {q: {"type": "noul", "noul": 0.5} for q in questions}}


def test_burst_is_coalesced_and_every_caller_answered():
    b = Batcher(FakeEngine())
    q = {"q": {"type": "noul", "instructions": "Is it urgent?"}}
    out = [None] * 8
    threads = [threading.Thread(target=lambda i=i: out.__setitem__(i, b.answer({"s": i}, q))) for i in range(8)]
    for t in threads: t.start()
    for t in threads: t.join(5)
    assert all(o and "answers" in o for o in out)
    assert len(b.engine.embedder.calls) < 8 and max(b.batches) > 1   # fewer encoder passes than requests


def test_bad_request_fails_alone():
    b = Batcher(FakeEngine())
    with pytest.raises(ValueError):
        b.answer({"s": 1}, {"q": {"type": "nope"}})
    assert "answers" in b.answer({"s": 2}, {"q": {"type": "noul", "instructions": "ok?"}})
