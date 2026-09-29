"""Encode Banking77 with any qwen3-family MLX encoder into a NEW npz (never touches emb.npz): python exp_encode_any.py ENCODER OUT.npz META.json"""
import json, sys, time
import numpy as np
from clm_tune_mlx.schema import build_pairs
from clm_tune_mlx import MLXEmbedder
enc, out, meta = sys.argv[1:4]
d = json.load(open("banking77.json"))
labels = [d["labels"][str(i)] for i in range(77)]
route = lambda n: n.replace("_", " ").replace("?", "").strip().capitalize()
Q = {"type": "choice", "instructions": "Which support route should handle `message`?", "criteria": {l: route(l) for l in labels}}
state = lambda m: build_pairs({"message": m}, {"q": Q})["q"][0]
emb = MLXEmbedder(enc); t = time.time()
opts, _ = emb.embed(list(Q["criteria"].values()))
tr, _ = emb.embed([state(r["text"]) for r in d["train"]]); te, _ = emb.embed([state(r["text"]) for r in d["test"]])
secs = time.time() - t
np.savez(out, opts=opts, tr=tr, te=te, ytr=np.array([r["label"] for r in d["train"]]), yte=np.array([r["label"] for r in d["test"]]))
json.dump({"encoder": enc, "load_seconds": emb.load_seconds, "embed_seconds": secs, "dim": int(tr.shape[1])}, open(meta, "w"))
print(enc, "dim", tr.shape[1], f"encode {secs:.0f}s load {emb.load_seconds:.0f}s")
