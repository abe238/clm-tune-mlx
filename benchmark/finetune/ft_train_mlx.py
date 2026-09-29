"""Same Banking77 experiment as ft_train.py, but the heads train in pure MLX (clm_tune_mlx.train_heads), on emb.npz.
Recipe and split logic match ft_train.py: warm start from the released head, AdamW lr 5e-4, batch 256, <=60 epochs,
patience 5, early stop on 10% of the chosen train subset, one shared seed-0 rng for subset choice and the val split.
Writes ft-results-mlx.json (compare with ft-results.json / ft-seeds.json)."""
import json, os
import numpy as np
from clm_tune_mlx.engine import Heads, default_head
from clm_tune_mlx.train import train_heads, evaluate_heads

rng = np.random.default_rng(0)
Z = np.load("emb.npz"); opts, tr, te, ytr, yte = (Z[k] for k in ("opts", "tr", "te", "ytr", "yte"))
CK = os.environ.get("CLM_CKPT") or default_head()   # a .pt needs the torch extra for the one-time conversion
res = {"zero_shot": evaluate_heads(Heads(CK), te, opts, yte)}
for per in (5, 20, None):
    idx = np.arange(len(ytr)) if per is None else np.concatenate([rng.permutation(np.where(ytr == c)[0])[:per] for c in range(77)])
    out = "clm-banking77-heads-mlx.safetensors" if per is None else None
    heads, info = train_heads(tr[idx], opts, ytr[idx], val_frac=0.1, init=CK, epochs=60, lr=5e-4, batch=256, patience=5, rng=rng, out=out)
    info.pop("loss_history"); info.pop("val_examples")
    res[f"{'all' if per is None else per}_per_route"] = {**info, **evaluate_heads(heads, te, opts, yte)}
json.dump(res, open("ft-results-mlx.json", "w"), indent=1)
print(json.dumps(res, indent=1))
