"""Experiment B: train A0 and A4 (best of A1-A4 on Banking77 validation) on typed-decisions choice questions, 3 seeds.
Evaluation only: never ship these heads. Val = 10% of train cases (grouped by case id) picks the kept epoch."""
import json, statistics as st
import numpy as np
import mlx.core as mx
mx.set_default_device(mx.cpu); assert mx.default_device() == mx.cpu   # CPU only: a GPU job may be running
from collections import Counter
from clm_tune_mlx.engine import default_head
from clm_tune_mlx.train import train_heads, evaluate_heads

Z = np.load("emb.npz"); x, y, split, case, uvec, idx = (Z[k] for k in ("x", "y", "split", "case", "uvec", "idx"))
opts = [uvec[r[r >= 0]] for r in idx]
tri, tei = np.where(split == "train")[0], np.where(split == "test")[0]
raw = {json.loads(l)["id"]: json.loads(l) for l in open("raw.jsonl")}
keys_of = lambda n: list(json.loads(raw[case[n]]["questions"])[str(Z["q"][n])]["criteria"])
lab = [keys_of(n)[y[n]] for n in range(len(y))]
qn = Z["q"]
top = Counter(lab[n] for n in tri).most_common(1)[0][0]
majority_test = float(np.mean([lab[n] == top for n in tei]))            # one global majority label (weak)
maj_q = {k: Counter(lab[n] for n in tri if qn[n] == k).most_common(1)[0][0] for k in set(qn)}
majority_per_q = float(np.mean([lab[n] == maj_q[qn[n]] for n in tei]))   # train-majority label per question name
CK = default_head()
CFG = {"A0": dict(epochs=60, patience=5),
       "A4": dict(epochs=80, patience=20, schedule="warmup_cosine", init_mode="fresh", center=False),   # existing row was uncentered (pre-`center`)
       "A1": dict(epochs=80, patience=20), "A2": dict(epochs=80, patience=20, schedule="warmup_cosine"),
       "A4c": dict(epochs=80, patience=20, schedule="warmup_cosine", init_mode="fresh", center=True),
       "A1fc": dict(epochs=80, patience=20, schedule="constant", init_mode="fresh", center=True)}
import os
old = json.load(open("results.json")) if os.path.exists("results.json") else {}   # keep existing rows, only run missing configs

class FixedPerm:
    """rng shim: first permutation(int) returns the grouped-val ordering (val rows first); no shuffle afterwards."""
    def __init__(self, perm): self.perm = perm
    def permutation(self, a): return self.perm

res = {"n_train": len(tri), "n_test": len(tei), "majority_class_test_acc": majority_test, "majority_per_question_test_acc": majority_per_q, "chance_test_acc": float(np.mean([1 / len(o) for o in [opts[n] for n in tei]])), "majority_label": top,
       "reference": {"study_clm_qwen3_8b": 0.753, "majority_class_all_types": 0.484, "teacher_self_agreement_ceiling": 0.735}}
for name, cfg in CFG.items():
    if name in old: res[name] = old[name]; continue
    runs = []
    for seed in (0, 1, 2):
        r = np.random.default_rng(seed)
        cases = np.array(sorted(set(case[tri]))); r.shuffle(cases)
        vc = set(cases[:max(1, round(0.1 * len(cases)))])
        isv = np.array([case[n] in vc for n in tri])
        order = np.concatenate([np.arange(len(tri))[isv], np.arange(len(tri))[~isv]])
        nv = int(isv.sum())
        xt, yt, ot = x[tri], y[tri], [opts[n] for n in tri]
        h, m = train_heads(xt, ot, yt, val_frac=(nv + .5) / len(tri), init=CK, lr=5e-4, batch=256, seed=seed, rng=FixedPerm(order),
                           test=(x[tei], [opts[n] for n in tei], y[tei]), **cfg)
        assert m["val_examples"] == nv
        ev = evaluate_heads(h, x[tei], [opts[n] for n in tei], y[tei])
        runs.append({"seed": seed, "test_acc": ev["acc"], "last_epoch_test_acc": m["last_epoch_test_acc"], "val_acc": m["val_acc"],
                     "kept_epoch": m["kept_epoch"], "epochs_run": m["epochs"], "train_seconds": m["train_seconds"],
                     "loss_history": m["loss_history"], "val_acc_history": m["val_acc_history"]})
        print(name, seed, {k: v for k, v in runs[-1].items() if "history" not in k}, flush=True)
    res[name] = {"config": cfg, "runs": runs}
json.dump(res, open("results.json", "w"), indent=1)
f = lambda v: f"{st.mean(v)*100:.1f} ({min(v)*100:.1f}-{max(v)*100:.1f})"
L = ["| cfg | test acc % mean (min-max) | last-epoch test % | val acc % | kept epoch | train s |", "|---|---|---|---|---|---|"]
for n in CFG:
    R = res[n]["runs"]; g = lambda k: [q[k] for q in R]
    L.append(f"| {n} {json.dumps(CFG[n])} | {f(g('test_acc'))} | {f(g('last_epoch_test_acc'))} | {f(g('val_acc'))} | {'/'.join(map(str, g('kept_epoch')))} | {st.mean(g('train_seconds')):.1f} |")
ref = ["", f"Choice-only: {len(tri)} train / {len(tei)} test questions. Baselines on our choice-only test: global majority label `{top}` {majority_test*100:.1f}%; per-question-name majority {majority_per_q*100:.1f}%; uniform chance {np.mean([1/len(opts[n]) for n in tei])*100:.1f}%.", "",
       "Reference points (from the study/card, NOT directly comparable): study CLM heads Qwen3-8B 75.3% (PyTorch, 80 epochs); majority class 48.4%; teacher self-agreement ceiling 73.5%.",
       "", "Caveat: the study scored ALL question types (noul, choice, score); this run scores choice questions only. Gold = `gold[q].label`. Evaluation only: weights trained on this data must not be shipped (Apache-2.0 data, eval use only per spec)."]
open("results.md", "w").write("# typed-decisions (choice-only) with A0, A4, A1, A2, A4c, A1fc\n\n" + "\n".join(L + ref) + "\n")
print("\n".join(L + ref))
