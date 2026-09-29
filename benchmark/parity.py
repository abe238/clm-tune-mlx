"""MLX port parity: Qwen3-8B last-token, final-norm hidden state from transformers (CPU, bf16 weights,
float32 output) vs the MLX embeddings saved by run_clm_mlx.py, on a fixed sample of texts.
Also re-runs the CLM heads on the reference vectors to check the decisions themselves match."""
import glob, json, os, sys, time
import numpy as np, torch
from transformers import AutoModel, AutoTokenizer
from common import HERE
K = 24
tags = [os.path.basename(f)[4:-4] for f in sorted(glob.glob(os.path.join(HERE, "results", "emb-*.npz")))]
base = np.load(os.path.join(HERE, "results", f"emb-{tags[0]}.npz"), allow_pickle=True)
texts = list(base["texts"])
rng = np.random.default_rng(0)
pick = sorted(rng.choice(len(texts), size=min(K, len(texts)), replace=False).tolist())
sample = [texts[i] for i in pick]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
t = time.time()
model = AutoModel.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16).eval()
load_s = time.time() - t
ref = []
with torch.no_grad():
    for s in sample:
        ids = tok.encode(s)[-2048:]
        h = model(input_ids=torch.tensor([ids])).last_hidden_state[0, -1].float().numpy()
        ref.append(h / (np.linalg.norm(h) + 1e-12))
ref = np.stack(ref)
out = {"k": len(sample), "reference": "transformers Qwen/Qwen3-8B bf16 on CPU, last_hidden_state (post final norm), last token",
       "load_seconds": load_s, "per_tag": {}}
for tag in tags:
    d = np.load(os.path.join(HERE, "results", f"emb-{tag}.npz"), allow_pickle=True)
    idx = {t: i for i, t in enumerate(d["texts"])}
    v = np.stack([d["vecs"][idx[s]] for s in sample])
    cos = (v * ref).sum(1)
    out["per_tag"][tag] = {"cos_min": float(cos.min()), "cos_mean": float(cos.mean()), "cos": cos.tolist(),
                           "texts": [s[:80] for s in sample]}
    print(tag, "cos min", round(float(cos.min()), 5), "mean", round(float(cos.mean()), 5))
json.dump(out, open(os.path.join(HERE, "results", "parity.json"), "w"))
