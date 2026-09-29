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


class MLXEngine(FakeEngine):
    """Holds MLX arrays built at load time and evaluates them per answer, like the real engine."""
    def __init__(self):
        super().__init__()
        import mlx.core as mx
        self.w = mx.ones((4, 4))

    def answer(self, state, questions, **kw):
        import mlx.core as mx
        v = float(mx.sum(self.w @ mx.ones((4,))).item())
        return {"answers": {q: {"type": "noul", "noul": v} for q in questions}}


def test_loader_builds_engine_on_worker_thread():
    """Regression: with MLX >= 0.32, arrays created on the main thread could not be evaluated on the
    batcher thread ("There is no Stream(cpu, 0) in current thread"). The loader runs on the worker."""
    b = Batcher(loader=MLXEngine)
    out = b.answer({"x": 1}, {"q": {"type": "noul", "instructions": "Is `x` set?"}})
    assert out["answers"]["q"]["noul"] == 16.0


def test_loader_error_is_raised():
    def boom():
        raise RuntimeError("load failed")
    with pytest.raises(RuntimeError, match="load failed"):
        Batcher(loader=boom)


def test_engine_built_on_main_thread_reproduces_stream_bug():
    """The bug the loader fixes: an MLX engine built on the main thread fails on the batcher thread."""
    b = Batcher(MLXEngine())
    with pytest.raises(RuntimeError, match="no Stream"):
        b.answer({"x": 1}, {"q": {"type": "noul", "instructions": "Is `x` set?"}})
