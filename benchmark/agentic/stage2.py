"""Two-stage pipeline: a shortlister's top 10 -> laya-mlx picks one. Final accuracy per shortlister.
Usage: <laya-mlx venv python> stage2.py <ckpt dir> <tag>"""
import json, math, re, sys, time, statistics as st
from collections import Counter
import laya_mlx
ckpt, tag = sys.argv[1], sys.argv[2]
agent = laya_mlx.load(ckpt, dtype="float16")
load = lambda p: [json.loads(l) for l in open(p) if l.strip()]
pool = {}
for it in load("bfcl/BFCL_v3_simple.json") + load("bfcl/BFCL_v3_multiple.json"):
    for f in it["function"]: pool.setdefault(f["name"], f)
names = sorted(pool); text = {n: f"{n}: {' '.join(pool[n]['description'].split())[:300]}" for n in names}
tok = lambda s: re.findall(r"[a-z0-9]+", s.lower().replace("_", " ").replace(".", " "))
docs = [tok(text[n]) for n in names]; df = Counter(w for d in docs for w in set(d)); N = len(docs); avg = sum(map(len, docs)) / N
def bm25_top(q, k=10):
    qt = tok(q); sc = [sum(math.log(1 + (N - df[w] + .5) / (df[w] + .5)) * Counter(d)[w] * 2.5 / (Counter(d)[w] + 1.5 * (.25 + .75 * len(d) / avg)) for w in qt if w in d) for d in docs]
    return [names[i] for i in sorted(range(N), key=lambda i: -sc[i])[:k]]
def pick(state, instr, opts):
    keys = [f"o{i}" for i in range(len(opts))]
    a = agent.predict(state, {"q": {"type": "choice", "instructions": instr, "criteria": dict(zip(keys, opts))}})["answers"]["q"]
    return opts[keys.index(a["choice"])]
out = {}
ret = json.load(open(f"results/retrieval-{tag}.json")); reqs = {}
for c in json.loads(open("tools_cases.json").read()): reqs.setdefault(c["id"].split("|")[0], c)
for method in ("clm", "bm25"):
    ok = inshort = 0
    for rid, c in reqs.items():
        top = ret["top10"][rid] if method == "clm" else bm25_top(c["state"]["request"])
        inshort += c["gold"] in top
        ok += pick(c["state"], "Which function should be called to handle `request`?", [text[n] for n in top]) == text[c["gold"]]
    out[f"tools/{method}->laya"] = {"gold_in_top10": inshort / len(reqs), "final_accuracy": ok / len(reqs)}
try:
    ws = json.load(open(f"results/web-shortlists-{tag}.json"))
    for method, rows in ws.items():
        ok = sum(pick(r["state"], "Which page element should be used for `next_operation` to make progress on `task`?", r["options"]) == r["gold_text"] for r in rows)
        out[f"web/{method}->laya"] = {"gold_in_top10": sum(r["gold_text"] in r["options"] for r in rows) / len(rows), "final_accuracy": ok / len(rows)}
except FileNotFoundError:
    pass
json.dump(out, open(f"results/stage2-{tag}.json", "w"), indent=1)
print(json.dumps(out, indent=1))
