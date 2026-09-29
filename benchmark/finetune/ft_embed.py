"""Encode Banking77 once with the MLX encoder (the only 8B work; training never touches it)."""
import json, os, sys, time, hashlib
import numpy as np
from clm_tune_mlx.schema import build_pairs
from clm_tune_mlx import MLXEmbedder
enc = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen3-8B-8bit"
d = json.load(open("banking77.json"))
labels = [d["labels"][str(i)] for i in range(77)]
route = lambda n: n.replace("_", " ").replace("?", "").strip().capitalize()
Q = {"type": "choice", "instructions": "Which support route should handle `message`?", "criteria": {l: route(l) for l in labels}}
state = lambda m: build_pairs({"message": m}, {"q": Q})["q"][0]
emb = MLXEmbedder(enc); t = time.time()
opts, _ = emb.embed(list(Q["criteria"].values()))
tr, _ = emb.embed([state(r["text"]) for r in d["train"]]); te, _ = emb.embed([state(r["text"]) for r in d["test"]])
secs = time.time() - t
np.savez("emb.npz", opts=opts, tr=tr, te=te, ytr=np.array([r["label"] for r in d["train"]]), yte=np.array([r["label"] for r in d["test"]]))
json.dump({"encoder": enc, "embed_seconds": secs, "n_train": len(tr), "n_test": len(te), "labels": labels, "question": Q}, open("embed-meta.json", "w"), indent=1)
print(f"encoded {len(tr)} train + {len(te)} test + 77 routes in {secs/60:.1f} min")
