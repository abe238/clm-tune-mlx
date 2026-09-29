"""Metrics over results/*.jsonl -> results/data.json (embedded in the HTML report)."""
import json, glob, math, os, statistics
from itertools import combinations
from common import load_cases, HERE
cases = load_cases(); byid = {c["id"]: c for c in cases}; N = len(cases)
ORDER = ["clm-mlx-bf16", "clm-mlx-q8", "clm-raw-q8", "laya-english", "laya-typed-decisions", "laya-mlx-typed-decisions-fp16"]
models = {}
for f in glob.glob(os.path.join(HERE, "results", "*.jsonl")):
    m = os.path.basename(f)[:-6]
    rows = [json.loads(l) for l in open(f)]
    assert len(rows) == N and {r["id"] for r in rows} == set(byid), f"{m}: {len(rows)} rows, expected {N}"
    models[m] = {r["id"]: r for r in rows}
names = [m for m in ORDER if m in models] + sorted(set(models) - set(ORDER))

def conf(r):  # model-agnostic confidence: probability of the chosen option
    p = r.get("probs") or {}
    return max(p.values()) if p else None

def metrics(m, ids):
    rows = [models[m][i] for i in ids]
    ok = [r for r in rows if not r["error"]]
    acc = sum(r["correct"] for r in ok) / len(rows) if rows else None
    brier, ll, pts = [], [], []
    for r in ok:
        c, p = byid[r["id"]], r["probs"]
        if not p: continue
        keys = list(c["question"]["criteria"]) if c["type"] == "choice" else (["false", "true"] if c["type"] == "noul" else [str(i) for i in range(len(c["question"]["criteria"]))])
        brier.append(sum((p.get(k, 0.0) - (1.0 if k == c["gold"] else 0.0)) ** 2 for k in keys))
        ll.append(-math.log(max(p.get(c["gold"], 0.0), 1e-6)))
        pts.append((conf(r), r["correct"]))
    ece = 0.0
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        bin_ = [(cf, ok_) for cf, ok_ in pts if (lo < cf <= hi) or (b == 0 and cf == 0)]
        if bin_:
            ece += len(bin_) / len(pts) * abs(sum(o for _, o in bin_) / len(bin_) - sum(cf for cf, _ in bin_) / len(bin_))
    right = [cf for cf, o in pts if o]; wrong = [cf for cf, o in pts if not o]
    sec = [r["seconds"] for r in rows if r["seconds"] is not None]
    mae = [abs(r["expected_level"] - int(byid[r["id"]]["gold"])) for r in ok if "expected_level" in r]
    if m.startswith("clm-raw"): sec = []  # reuses the heads run's cached embeddings: its timing is not a real latency
    return {"n": len(rows), "errors": len(rows) - len(ok), "accuracy": acc,
            "brier": statistics.mean(brier) if brier else None, "logloss": statistics.mean(ll) if ll else None,
            "ece": ece if pts else None, "conf_right": statistics.mean(right) if right else None,
            "conf_wrong": statistics.mean(wrong) if wrong else None, "score_mae": statistics.mean(mae) if mae else None,
            "p50_ms": 1000 * statistics.median(sec) if sec else None,
            "p95_ms": 1000 * sorted(sec)[max(0, math.ceil(0.95 * len(sec)) - 1)] if sec else None}

def coverage(m):  # accuracy when acting only at confidence >= t
    pts = sorted(((conf(r), r["correct"]) for r in models[m].values() if not r["error"] and conf(r) is not None), key=lambda x: -x[0])
    out, hits = [], 0
    for k, (cf, o) in enumerate(pts, 1):
        hits += o
        out.append({"coverage": k / N, "accuracy": hits / k, "threshold": cf})
    return out

def calib(m):
    pts = [(conf(r), r["correct"]) for r in models[m].values() if not r["error"] and conf(r) is not None]
    bins = []
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        x = [(cf, o) for cf, o in pts if lo < cf <= hi]
        if x: bins.append({"bin": (lo + hi) / 2, "conf": sum(c for c, _ in x) / len(x), "acc": sum(o for _, o in x) / len(x), "n": len(x)})
    return bins

slices = {"all": [c["id"] for c in cases]}
for key in ("type", "difficulty", "domain"):
    for v in sorted({c[key] for c in cases}):
        slices[f"{key}:{v}"] = [c["id"] for c in cases if c[key] == v]
for fl in sorted({f for c in cases for f in c["flags"]}):
    slices[f"flag:{fl}"] = [c["id"] for c in cases if fl in c["flags"]]
slices["flag:plain"] = [c["id"] for c in cases if not c["flags"]]

agree = {f"{a}|{b}": sum(models[a][i].get("label") == models[b][i].get("label") for i in byid) / N for a, b in combinations(names, 2)}
metas = {os.path.basename(f)[5:-5]: json.load(open(f)) for f in glob.glob(os.path.join(HERE, "results", "meta-*.json"))}
parity = json.load(open(os.path.join(HERE, "results", "parity.json"))) if os.path.exists(os.path.join(HERE, "results", "parity.json")) else None
data = {"n": N, "models": names, "metas": metas, "parity": parity,
        "slices": {s: {m: metrics(m, ids) for m in names} | {"_n": len(ids)} for s, ids in slices.items()},
        "coverage": {m: coverage(m) for m in names}, "calibration": {m: calib(m) for m in names}, "agreement": agree,
        "cases": [c | {"results": {m: {k: models[m][c["id"]].get(k) for k in ("label", "probs", "correct", "seconds", "error", "expected_level")} for m in names}} for c in cases]}
json.dump(data, open(os.path.join(HERE, "results", "data.json"), "w"))
for m in names:
    s = data["slices"]["all"][m]
    print(f"{m:22s} acc={s['accuracy']:.3f} brier={s['brier']:.3f} ece={s['ece']:.3f} p50={s['p50_ms'] or 0:.0f}ms err={s['errors']}")
