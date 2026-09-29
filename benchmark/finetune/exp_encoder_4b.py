"""Experiment C: Banking77 with Qwen3-4B-8bit (dim 2560) vs Qwen3-8B-8bit: fresh heads, best config (A4), seeds 0-2.
Reads scratch npz produced by exp_encode_any.py (path args) and /usr/bin/time numbers passed as JSON via argv[3]."""
import json, sys, statistics as st
import numpy as np
from clm_tune_mlx.train import train_heads, evaluate_heads
S = sys.argv[1]; extra = json.loads(sys.argv[2])
cfg = dict(epochs=80, patience=20, schedule="warmup_cosine", init_mode="fresh")
out = {"config": cfg, "note": "8B rows: fresh heads on the stored 8B embeddings (A4 from exp-trainer.json is the same run)"}
for tag, f in (("qwen3-4b-8bit", S + "emb4b.npz"), ("qwen3-8b-8bit", S + "emb8b.npz")):
    Z = np.load(f); opts, tr, te, ytr, yte = (Z[k] for k in ("opts", "tr", "te", "ytr", "yte"))
    runs = []
    for seed in (0, 1, 2):
        h, m = train_heads(tr, opts, ytr, val_frac=0.1, lr=5e-4, batch=256, seed=seed, rng=np.random.default_rng(seed), test=(te, opts, yte), **cfg)
        runs.append({"seed": seed, "test_acc": evaluate_heads(h, te, opts, yte)["acc"], "last_epoch_test_acc": m["last_epoch_test_acc"],
                     "val_acc": m["val_acc"], "kept_epoch": m["kept_epoch"], "train_seconds": m["train_seconds"]})
        print(tag, runs[-1], flush=True)
    a = [r["test_acc"] for r in runs]
    out[tag] = {"dim": int(tr.shape[1]), "runs": runs, "test_acc_mean": st.mean(a), "test_acc_min": min(a), "test_acc_max": max(a), **extra[tag]}
json.dump(out, open("exp-encoder-4b.json", "w"), indent=1)
