"""Experiment D: Qwen3.8-27B 4-bit encoder on Banking77, fresh heads with A4 (the only config comparable across
encoder sizes, since the released heads are 4096-dim), seeds 0-2. 4B/8B rows come from exp-encoder-4b.json.
python exp_encoder_27b.py SCRATCH_DIR/  (reads emb27b.npz + emb27b-meta.json there)"""
import json, sys, statistics as st
import numpy as np
from clm_tune_mlx.train import train_heads, evaluate_heads
S = sys.argv[1]
cfg = dict(epochs=80, patience=20, schedule="warmup_cosine", init_mode="fresh")
Z = np.load(S + "emb27b.npz"); opts, tr, te, ytr, yte = (Z[k] for k in ("opts", "tr", "te", "ytr", "yte"))
runs = []
for seed in (0, 1, 2):
    h, m = train_heads(tr, opts, ytr, val_frac=0.1, lr=5e-4, batch=256, seed=seed, rng=np.random.default_rng(seed), **cfg)
    runs.append({"seed": seed, "test_acc": evaluate_heads(h, te, opts, yte)["acc"], "val_acc": m["val_acc"], "train_seconds": m["train_seconds"]})
    print(runs[-1], flush=True)
a = [r["test_acc"] for r in runs]
prev = json.load(open("exp-encoder-4b.json"))
out = {"config": cfg, "qwen3.8-27b-4bit": {"dim": int(tr.shape[1]), "runs": runs, "test_acc_mean": st.mean(a), "test_acc_min": min(a),
       "test_acc_max": max(a), **json.load(open(S + "emb27b-meta.json"))},
       **{k: {x: prev[k][x] for x in ("dim", "test_acc_mean", "test_acc_min", "test_acc_max")} for k in ("qwen3-4b-8bit", "qwen3-8b-8bit")}}
json.dump(out, open("exp-encoder-27b.json", "w"), indent=1)
