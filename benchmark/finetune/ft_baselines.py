"""laya-mlx / BM25 on the same held-out Banking77 test (77 routes)."""
import json, math, os, re, sys, time
from collections import Counter
kind = sys.argv[1]
d = json.load(open("banking77.json")); meta = json.load(open("embed-meta.json")); Q = meta["question"]; labels = meta["labels"]
test = d["test"]
if kind == "laya-mlx":
    import laya_mlx; agent = laya_mlx.load(sys.argv[2], dtype="float16"); ask = lambda m: agent.predict({"message": m}, {"q": Q})["answers"]["q"]["choice"]
else:
    tok = lambda t: re.findall(r"[a-z0-9]+", t.lower()); docs = [tok(v) for v in Q["criteria"].values()]; keys = list(Q["criteria"])
    df = Counter(w for dd in docs for w in set(dd)); N = len(docs); avg = sum(map(len, docs)) / N
    ask = lambda m: keys[max(range(N), key=lambda i: sum(math.log(1 + (N - df[w] + .5) / (df[w] + .5)) * Counter(docs[i])[w] * 2.5 / (Counter(docs[i])[w] + 1.5 * (.25 + .75 * len(docs[i]) / avg)) for w in tok(m) if w in docs[i]))]
t = time.time(); ok = err = 0
for r in test:
    try: ok += ask(r["text"]) == labels[r["label"]]
    except Exception: err += 1
out = {"model": kind, "n": len(test), "acc": ok / len(test), "errors": err, "ms_per_msg": 1000 * (time.time() - t) / len(test)}
json.dump(out, open(f"baseline-{kind}.json", "w")); print(out)
