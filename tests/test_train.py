"""Pure-MLX head training on tiny synthetic embeddings: no model download."""
import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from mlx.utils import tree_flatten

from clm_tune_mlx.engine import Head, Heads
from clm_tune_mlx.train import bm25_pick, evaluate_heads, report_card, train_heads

D, K = 16, 5


@pytest.fixture
def init(tmp_path):
    """A tiny random 'released' head in the layout engine.Heads loads."""
    mx.random.seed(1)
    meta = dict(width=32, depth=3, activation="gelu", proj=8, layernorm=True, residual=False, hidden=D, logit_scale=2.0)
    kw = {k: meta[k] for k in ("width", "depth", "proj", "layernorm", "residual")}
    arrays = {f"{s}.{k}": v for s in ("state", "action") for k, v in tree_flatten(Head(hidden=D, **kw).parameters())}
    p = str(tmp_path / "init.safetensors")
    mx.save_safetensors(p, arrays)
    json.dump(meta, open(p[:-len(".safetensors")] + ".json", "w"))
    return p


def data(n=20, seed=0):
    r = np.random.default_rng(seed)
    y = np.arange(n) % K
    protos = r.normal(size=(K, D)).astype(np.float32)
    return (protos[y] + 0.1 * r.normal(size=(n, D))).astype(np.float32), protos, y


def test_loss_decreases_and_overfits(init):
    x, protos, y = data()
    h, m = train_heads(x, protos, y, val_frac=0, init=init, epochs=150, lr=5e-3, patience=150, batch=8)
    assert m["loss_history"][-1] < m["loss_history"][0] * 0.5
    assert evaluate_heads(h, x, protos, y)["acc"] == 1.0


def test_save_load_roundtrip(init, tmp_path):
    x, protos, y = data()
    out = str(tmp_path / "t.safetensors")
    h, _ = train_heads(x, protos, y, init=init, epochs=5, out=out)
    h2 = Heads(out)                                  # engine.py's own loader, no conversion
    assert h2.scale == pytest.approx(h.scale, rel=1e-5)
    a = h.project(mx.array(x), "state") @ h.project(mx.array(protos), "action").T
    b = h2.project(mx.array(x), "state") @ h2.project(mx.array(protos), "action").T
    assert np.allclose(np.array(a), np.array(b), atol=1e-5)
    assert (np.array(a).argmax(1) == np.array(b).argmax(1)).all()


def test_per_example_option_lists(init):
    r = np.random.default_rng(3)
    n = 24
    x = r.normal(size=(n, D)).astype(np.float32)
    ks = [3 + i % 3 for i in range(n)]                # ragged: 3..5 options each
    y = np.array([i % k for i, k in enumerate(ks)])
    opts = [r.normal(size=(k, D)).astype(np.float32) for k in ks]
    for i in range(n):                                # make the right option the state itself
        opts[i][y[i]] = x[i]
    h, m = train_heads(x, opts, y, init=init, epochs=30, lr=5e-3, patience=30, batch=8)
    assert m["loss_history"][-1] < m["loss_history"][0]
    assert evaluate_heads(h, x, opts, y)["acc"] >= 0.9


def test_report_card_says_no_gain():
    d = {"y": np.array([0, 0, 1]), "states": ["a", "b", "c"], "option_texts": [["x", "y"]] * 3}
    rep = report_card(d, d, {"acc": 0.5, "top3": 1.0})
    assert not rep["beats_baselines"] and rep["verdict"].startswith("No gain over baselines")
    assert bm25_pick("refund money", ["card lost", "money refund"]) == 1


class FakeEmbedder:
    def embed(self, texts):
        return np.stack([np.random.default_rng(abs(hash(t)) % 2**32).normal(size=D).astype(np.float32) for t in texts]), 0


def test_jsonl_encode_cache_and_cli(init, tmp_path):
    from clm_tune_mlx.train import encode_jsonl, main
    p = tmp_path / "d.jsonl"
    rows = [{"state": f"ticket {i} refund" if i % 2 else f"ticket {i} cat", "options": ["refund", "cat"], "label": i % 2} for i in range(40)]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    d = encode_jsonl(str(p), embedder=FakeEmbedder())
    assert d["x"].shape == (40, D) and d["options"].shape == (2, D)            # one shared list stays [K,D]
    assert encode_jsonl(str(p), embedder=None)["x"].shape == (40, D)           # cache hit: no embedder needed
    out = tmp_path / "h.safetensors"
    main([str(p), "--out", str(out), "--init", init, "--epochs", "3"])
    rep = json.load(open(tmp_path / "h.report.json"))
    assert rep["n_test"] == 8 and "verdict" in rep and Heads(str(out)).scale > 0


def test_warmup_cosine_runs(init):
    x, protos, y = data(40)
    h, m = train_heads(x, protos, y, init=init, epochs=20, patience=20, batch=8, lr=5e-3, schedule="warmup_cosine")
    assert m["epochs"] == 20 and 1 <= m["kept_epoch"] <= 20 and len(m["val_acc_history"]) == 20
    _, m2 = train_heads(x, protos, y, init=init, epochs=20, patience=1, batch=8, schedule="warmup_cosine", test=(x, protos, y))
    assert m2["epochs"] >= 2 and 0 <= m2["last_epoch_test_acc"] <= 1     # no stop inside the 2 warm-up epochs


def test_fresh_init_seeded(init):
    x, protos, y = data()
    w = lambda h: np.array(h.state.inp.weight)
    kw = dict(init=init, epochs=1, lr=0.0, init_mode="fresh")     # lr 0: weights stay at their initial values
    a, _ = train_heads(x, protos, y, seed=1, **kw)
    b, _ = train_heads(x, protos, y, seed=1, **kw)
    c, _ = train_heads(x, protos, y, seed=2, **kw)
    r, _ = train_heads(x, protos, y, seed=1, init=init, epochs=1, lr=0.0)
    assert np.array_equal(w(a), w(b)) and not np.allclose(w(a), w(c)) and not np.allclose(w(a), w(r))
    assert a.scale == pytest.approx(r.scale)
