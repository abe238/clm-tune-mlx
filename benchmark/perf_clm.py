"""Headline performance table for clm_tune_mlx (batched + prefix-shared embedder), one encoder per run.
Usage: /tmp/clmv/bin/python perf_clm.py <encoder> <tag>"""
import json, os, sys, time, statistics
import mlx.core as mx
from clm.engine import Engine
from clm.schema import build_pairs
from clm_tune_mlx import MLXEmbedder
from common import load_cases, HERE
enc, tag = sys.argv[1], sys.argv[2]
t0 = time.time(); emb = MLXEmbedder(enc); load_s = time.time() - t0
eng = Engine(embedder=emb, checkpoint=(os.environ.get("CLM_CKPT") or __import__("clm.heads", fromlist=["download"]).download()), device="cpu", action_cache="0")
cases = load_cases(); short = [c for c in cases if "long" not in c["flags"]]
eng.answer(cases[0]["state"], {"q": cases[0]["question"]}); mx.reset_peak_memory()
def timed(fn, reps):
    ts = []
    for _ in range(reps):
        t = time.time(); fn(); ts.append(time.time() - t)
    return ts
out = {"tag": tag, "encoder": enc, "load_s": load_s}
warm, cold = [], []
for c in short:
    st, _, opts = build_pairs(c["state"], {"q": c["question"]})["q"]
    emb.cache.clear(); emb.embed(opts)                                  # options known ahead (the common case)
    warm += timed(lambda: eng.answer(c["state"], {"q": c["question"]}), 1)
    emb.cache.clear()
    cold += timed(lambda: eng.answer(c["state"], {"q": c["question"]}), 1)
q = lambda xs, p: 1000 * sorted(xs)[min(len(xs) - 1, int(p * len(xs)))]
out["one_question_options_cached"] = {"p50_ms": q(warm, .5), "p95_ms": q(warm, .95)}
out["one_question_cold"] = {"p50_ms": q(cold, .5), "p95_ms": q(cold, .95)}
# many questions on one state: prefix sharing
qs = {f"q{i}": c["question"] for i, c in enumerate([c for c in short if c["domain"] == "support"][:8])}
state = {"email": "Hi, we were charged twice for March and the app also crashes on the reports page. Please fix both today or we will move to a competitor."}
pairs = build_pairs(state, qs); emb.cache.clear(); emb.embed([t for p in pairs.values() for t in p[2]])
ts = []
for _ in range(5):
    for s in [p[0] for p in pairs.values()]: emb.cache.pop(s, None)
    ts += timed(lambda: eng.answer(state, qs), 1)
out["eight_questions_one_state"] = {"p50_ms": 1000 * statistics.median(ts), "questions_per_s": len(qs) / statistics.median(ts)}
out["peak_gib"] = mx.get_peak_memory() / 2**30
json.dump(out, open(os.path.join(HERE, "results", f"perf-clm-{tag}.json"), "w"), indent=1)
print(json.dumps(out))
