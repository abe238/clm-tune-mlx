"""CLM as a first-stage shortlister (the use the CLM authors and reviewers recommend).
For each of the 150 frozen BFCL requests, rank ALL 589 functions; report recall@K, latency vs menu
size, and a two-stage pipeline (CLM top-10 -> laya-mlx picks one).
Usage: /tmp/clmv/bin/python retrieval.py <encoder> <tag>"""
import hashlib, json, math, os, re, sys, time, statistics as st
from collections import Counter
import numpy as np
from clm.schema import build_pairs
from clm_tune_mlx import load_engine
enc, tag = sys.argv[1], sys.argv[2]
cases = json.loads(open("tools_cases.json").read())
reqs = {}
for c in cases:                                   # 150 unique requests (each appears in 3 menus)
    reqs.setdefault(c["id"].split("|")[0], c)
load = lambda p: [json.loads(l) for l in open(p) if l.strip()]
pool = {}
for it in load("bfcl/BFCL_v3_simple.json") + load("bfcl/BFCL_v3_multiple.json"):
    for f in it["function"]:
        pool.setdefault(f["name"], f)
names = sorted(pool); text = {n: f"{n}: {' '.join(pool[n]['description'].split())[:300]}" for n in names}
eng = load_engine(enc, os.environ.get("CLM_CKPT")); head = eng.heads["clm-latest"].ensure()
Q = "Which function should be called to handle `request`?"
t0 = time.time(); E_opt, _ = eng.embedder.embed([text[n] for n in names]); cache_s = time.time() - t0   # one-time option cache
A = head.project_actions(E_opt).cpu().numpy(); Araw = E_opt / np.linalg.norm(E_opt, axis=1, keepdims=True)
def rank_clm(state):
    s = build_pairs(state, {"q": {"type": "choice", "instructions": Q, "criteria": {"x": "y"}}})["q"][0]
    e, _ = eng.embedder.embed([s]); z = head.project_states(e).cpu().numpy()[0]
    return np.argsort(-(A @ z)), np.argsort(-(Araw @ (e[0] / np.linalg.norm(e[0]))))
tok = lambda s: re.findall(r"[a-z0-9]+", s.lower().replace("_", " ").replace(".", " "))
docs = [tok(text[n]) for n in names]; df = Counter(w for d in docs for w in set(d)); N = len(docs); avg = sum(map(len, docs)) / N
def rank_bm25(q, k1=1.5, b=0.75):
    qt = tok(q); sc = []
    for d in docs:
        tf = Counter(d); sc.append(sum(math.log(1 + (N - df[w] + .5) / (df[w] + .5)) * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / avg)) for w in qt if w in tf))
    return np.argsort(-np.array(sc))
K = (1, 5, 10, 20); hits = {m: {k: 0 for k in K} for m in ("clm", "raw", "bm25")}; lat = []; top10 = {}
for rid, c in reqs.items():
    g = names.index(c["gold"]); t = time.time(); rc, rr = rank_clm(c["state"]); lat.append(time.time() - t)
    for m, r in (("clm", rc), ("raw", rr), ("bm25", rank_bm25(c["state"]["request"]))):
        pos = int(np.where(r == g)[0][0])
        for k in K: hits[m][k] += pos < k
    top10[rid] = [names[i] for i in rc[:10]]
n = len(reqs)
# latency vs menu size through the normal engine path (options already cached)
scal = {}
for size in (10, 100, len(names)):
    ts = []
    for rid, c in list(reqs.items())[:40]:
        opts = [c["gold"]] + [x for x in names if x != c["gold"]][:size - 1]
        q = {"type": "choice", "instructions": Q, "criteria": {x: text[x] for x in opts}}
        eng.embedder.cache.pop(build_pairs(c["state"], {"q": q})["q"][0], None)
        t = time.time(); eng.answer(c["state"], {"q": q}); ts.append(time.time() - t)
    scal[size] = 1000 * st.median(ts)
out = {"tag": tag, "requests": n, "pool": len(names), "option_cache_seconds": cache_s,
       "recall": {m: {f"@{k}": v / n for k, v in h.items()} for m, h in hits.items()},
       "rank_ms_p50": 1000 * st.median(lat), "latency_ms_by_menu_size": scal, "top10": top10}
json.dump(out, open(f"results/retrieval-{tag}.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "top10"}, indent=1))
