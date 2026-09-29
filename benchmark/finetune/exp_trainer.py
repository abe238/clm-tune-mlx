"""Experiment A: trainer configs A0-A4 on Banking77 emb.npz, seeds 0-2, all training data. Writes exp-trainer.json/.md.
Split logic as ft_train_mlx.py: fixed tr/te from emb.npz, 10% of train held out for early stopping (rng seeded per run)."""
import json, statistics as st
import numpy as np
from clm_tune_mlx.engine import Heads, default_head
from clm_tune_mlx.train import train_heads, evaluate_heads

Z = np.load("emb.npz"); opts, tr, te, ytr, yte = (Z[k] for k in ("opts", "tr", "te", "ytr", "yte"))
CK = default_head()
CFG = {"A0": dict(epochs=60, patience=5), "A1": dict(epochs=80, patience=20),
       "A2": dict(epochs=80, patience=20, schedule="warmup_cosine"),
       "A3": dict(epochs=80, patience=20, init_mode="fresh"),
       "A4": dict(epochs=80, patience=20, schedule="warmup_cosine", init_mode="fresh")}
res = {}
for name, cfg in CFG.items():
    runs = []
    for seed in (0, 1, 2):
        h, m = train_heads(tr, opts, ytr, val_frac=0.1, init=CK, lr=5e-4, batch=256, seed=seed, rng=np.random.default_rng(seed),
                           test=(te, opts, yte), **cfg)
        runs.append({"seed": seed, "test_acc": evaluate_heads(h, te, opts, yte)["acc"], "last_epoch_test_acc": m["last_epoch_test_acc"],
                     "val_acc": m["val_acc"], "kept_epoch": m["kept_epoch"], "epochs_run": m["epochs"], "train_seconds": m["train_seconds"],
                     "loss_history": m["loss_history"], "val_acc_history": m["val_acc_history"]})
        print(name, seed, {k: v for k, v in runs[-1].items() if "history" not in k}, flush=True)
    res[name] = {"config": cfg, "runs": runs}
    json.dump(res, open("exp-trainer.json", "w"), indent=1)
f = lambda v, pct=True: f"{st.mean(v)*100:.2f} ({min(v)*100:.2f}-{max(v)*100:.2f})" if pct else f"{st.mean(v):.1f} ({min(v):.0f}-{max(v):.0f})"
L = ["| cfg | config | test acc % mean (min-max) | last-epoch test % | val acc % | kept epoch | epochs run | train s (mean) |", "|---|---|---|---|---|---|---|---|"]
for n, r in res.items():
    R = r["runs"]; g = lambda k: [x[k] for x in R]
    L.append(f"| {n} | {json.dumps(r['config'])} | {f(g('test_acc'))} | {f(g('last_epoch_test_acc'))} | {f(g('val_acc'))} | {'/'.join(str(x) for x in g('kept_epoch'))} | {'/'.join(str(x) for x in g('epochs_run'))} | {st.mean(g('train_seconds')):.1f} |")
open("exp-trainer.md", "w").write("# Banking77 trainer experiments (3 seeds, all train data)\n\n" + "\n".join(L) + "\n")
print("\n".join(L))
