"""Pure-MLX training of CLM's two projection heads. No PyTorch anywhere on this path.

Recipe (mirrors benchmark/finetune/ft_train.py): warm start from a released head, L2-normalised state and
action projections, logit scale = exp(log_scale) clamped at 100 (log_scale is trained), cross-entropy over
the options, AdamW (bias-corrected, like torch), early stopping on a validation split, seed 0.

    clm-tune-mlx-train data.jsonl --out heads.safetensors      # data: {"state": ..., "options": [str], "label": int|str}
    clm-tune-mlx-serve --heads heads.safetensors
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from collections import Counter

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from .engine import Heads, default_head


def _norm(z):
    return z / mx.maximum(mx.linalg.norm(z, axis=-1, keepdims=True), 1e-12)


def _pad(options):
    """options -> (array [K,D] shared | [N,K,D] per example, mask [N,K] or None). Ragged lists are padded."""
    if isinstance(options, (np.ndarray, mx.array)):
        return np.asarray(options, dtype=np.float32), None
    ks = [len(o) for o in options]
    kmax = max(ks)
    out = np.zeros((len(options), kmax, np.asarray(options[0]).shape[-1]), np.float32)
    for i, o in enumerate(options):
        out[i, :len(o)] = o
    return out, (np.arange(kmax)[None] < np.array(ks)[:, None])


class _Pair(nn.Module):
    """Both heads plus the trainable log logit-scale. (Heads are shared with the `Heads` object.)"""

    def __init__(self, heads: Heads):
        super().__init__()
        self.s, self.a = heads.state, heads.action
        self.log_scale = mx.array(math.log(heads.scale), dtype=mx.float32)

    def logits(self, x, opts, mask=None):
        zs = _norm(self.s(x))
        scale = mx.minimum(mx.exp(self.log_scale), 100.0)
        if opts.ndim == 2:                                        # one shared option list
            return scale * zs @ _norm(self.a(opts)).T
        b, k, d = opts.shape                                      # per-example option lists
        za = _norm(self.a(opts.reshape(b * k, d))).reshape(b, k, -1)
        lg = scale * (za @ zs[:, :, None]).squeeze(-1)
        return lg if mask is None else mx.where(mask, lg, -1e9)


def _take(opts, mask, idx):
    return (opts, None) if opts.ndim == 2 else (opts[idx], mask[idx] if mask is not None else None)


def _evaluate(pair, x, opts, mask, y, chunk=1024):
    hit1 = hit3 = 0
    for i in range(0, len(y), chunk):
        j = np.arange(i, min(i + chunk, len(y)))
        o, m = _take(opts, mask, j)
        lg = pair.logits(mx.array(x[j]), mx.array(o), None if m is None else mx.array(m))
        top = np.array(mx.argsort(-lg, axis=-1)[:, :3])
        hit1 += int((top[:, 0] == y[j]).sum())
        hit3 += int((top == y[j][:, None]).any(-1).sum())
    return {"acc": hit1 / len(y), "top3": hit3 / len(y)}


def _center(v, mean):
    """Subtract `mean` (None = no-op) and L2-renormalise rows (numpy)."""
    if mean is None:
        return v
    v = v - mean
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def evaluate_heads(heads: Heads, x, options, y) -> dict:
    """Accuracy and top-3 of `heads` on precomputed embeddings (same option forms as train_heads).
    Applies the heads' saved centering means, if any."""
    o, m = _pad(options)
    x = _center(np.asarray(x, np.float32), heads.means.get("state"))
    return _evaluate(_Pair(heads), x, _center(o, heads.means.get("action")), m, np.asarray(y))


def save_heads(path: str, heads: Heads, log_scale: float | None = None) -> None:
    """Write `path` (.safetensors) plus the .json sidecar next to it, exactly the layout engine.Heads loads."""
    arrays = {f"{side}.{k}": v for side, h in (("state", heads.state), ("action", heads.action)) for k, v in tree_flatten(h.parameters())}
    arrays.update({f"center.{s}": mx.array(m) for s, m in heads.means.items()})
    meta = dict(heads.meta, logit_scale=float(math.log(heads.scale) if log_scale is None else log_scale), centered=bool(heads.means))
    mx.save_safetensors(path, arrays)
    json.dump(meta, open(path[: -len(".safetensors")] + ".json", "w"))


def train_heads(x, options, y, *, val_frac: float = 0.1, init: str | None = None, epochs: int = 80, lr: float = 5e-4,
                batch: int = 256, patience: int = 20, seed: int = 0, rng: np.random.Generator | None = None,
                shuffle: bool = False, out: str | None = None, schedule: str = "constant", min_epochs_before_stop: int = 0,
                init_mode: str = "released", test: tuple | None = None, center: bool | None = None) -> tuple[Heads, dict]:
    """Train state+action heads from precomputed embeddings.

    x [N,D] state vectors; y [N] option indices; options [K,D] (one list for everyone) or [N,K,D] / a list of
    [k_i,D] arrays (a list per example; y indexes into that example's list). `init` is a heads .safetensors/.pt
    (default: the released head). `val_frac` of the data (shuffled with `rng`, default seed) picks the best
    epoch (0 disables it and keeps the last epoch); stop after `patience` epochs without a strict improvement. Returns (heads at the best epoch, metrics).

    `schedule`: "constant" lr, or "warmup_cosine" (linear warm-up over the first 10% of steps, cosine decay to 0);
    with warmup_cosine no early stop fires during warm-up. `min_epochs_before_stop`: no early stop before that many epochs.
    `init_mode`: "released" (warm start from `init`) or "fresh" (same architecture, random weights seeded by `seed`,
    logit scale as the released head's). `test` = (x, options, y): also report last-epoch accuracy on it.
    `center`: subtract the state mean (train portion only, after the val split) and the option mean, then L2-renormalise,
    before training, validation and test; the means ride along on `heads.means` and are saved/applied at inference.
    None = auto: True for fresh heads, False for released ones (centering hurts released heads).
    """
    if schedule not in ("constant", "warmup_cosine") or init_mode not in ("released", "fresh"):
        raise ValueError("schedule: constant|warmup_cosine; init_mode: released|fresh")
    mx.random.seed(seed)
    rng = rng or np.random.default_rng(seed)
    x, y = np.asarray(x, np.float32), np.asarray(y)
    o, mask = _pad(options)
    heads = Heads(init or default_head())
    if init_mode == "fresh":
        from .engine import Head
        heads.meta = dict(heads.meta, hidden=x.shape[-1])       # fresh heads may target another encoder width
        kw = {k: heads.meta[k] for k in ("hidden", "width", "depth", "proj", "layernorm", "residual")}
        heads.state, heads.action = Head(**kw), Head(**kw)      # mx.random is seeded above: reproducible
        mx.eval(heads.state.parameters(), heads.action.parameters())
    pair = _Pair(heads)
    perm = rng.permutation(len(y))
    nv = max(1, int(len(y) * val_frac + 1e-9)) if val_frac > 0 else 0   # val_frac=0: no early stopping, keep the last epoch
    val, trn = perm[:nv], perm[nv:]
    if len(trn) == 0:
        raise ValueError("no training examples left after the validation split")
    if center is None:
        center = init_mode == "fresh"
    heads.means = {}
    if center:                                                   # means from the TRAIN rows only
        orows = o if o.ndim == 2 else o[trn][mask[trn]]
        heads.means = {"state": x[trn].mean(0), "action": orows.reshape(-1, o.shape[-1]).mean(0)}
        x, o = _center(x, heads.means["state"]), _center(o, heads.means["action"])
        if test is not None:
            test = (_center(np.asarray(test[0], np.float32), heads.means["state"]), test[1], test[2])
    spe = math.ceil(len(trn) / batch)
    warm = max(1, round(0.1 * epochs * spe))
    sched = lr if schedule == "constant" else optim.join_schedules(
        [optim.linear_schedule(0.0, lr, warm), optim.cosine_decay(lr, max(1, epochs * spe - warm))], [warm])
    no_stop_before = max(min_epochs_before_stop, math.ceil(warm / spe) if schedule == "warmup_cosine" else 0)
    opt = optim.AdamW(learning_rate=sched, bias_correction=True)     # torch AdamW: weight_decay 0.01, eps 1e-8

    def loss_fn(p, xb, ob, mb, yb):
        return nn.losses.cross_entropy(p.logits(xb, ob, mb), yb, reduction="mean")

    step = nn.value_and_grad(pair, loss_fn)
    vo, vm = _take(o, mask, val)
    best, best_params, bad, history, t0 = -1.0, None, 0, [], time.time()
    val_hist, kept = [], 0
    for ep in range(epochs):
        order = rng.permutation(trn) if shuffle else trn
        losses = []
        for b in range(0, len(order), batch):
            j = order[b:b + batch]
            ob, mb = _take(o, mask, j)
            loss, grads = step(pair, mx.array(x[j]), mx.array(ob), None if mb is None else mx.array(mb), mx.array(y[j]))
            opt.update(pair, grads)
            mx.eval(pair.parameters(), opt.state)
            losses.append(float(loss) * len(j))
        history.append(sum(losses) / len(order))
        if nv == 0:
            best_params, kept = pair.parameters(), ep + 1
            continue
        v = _evaluate(pair, x[val], vo, vm, y[val])["acc"]
        val_hist.append(v)
        if v > best:
            best, best_params, bad, kept = v, pair.parameters(), 0, ep + 1
        else:
            bad += 1
            if bad >= patience and ep + 1 >= no_stop_before:
                break
    last = None
    if test is not None:
        to, tm = _pad(test[1])
        to = _center(to, heads.means.get("action"))
        last = _evaluate(pair, np.asarray(test[0], np.float32), to, tm, np.asarray(test[2]))["acc"]
    pair.update(best_params)
    heads.scale = float(min(math.exp(float(pair.log_scale)), 100.0))
    metrics = {"train_examples": len(trn), "val_examples": nv, "val_acc": best, "epochs": ep + 1,
               "loss_history": history, "val_acc_history": val_hist, "kept_epoch": kept, "centered": bool(center), "last_epoch_test_acc": last,
               "train_seconds": time.time() - t0}
    if out:
        save_heads(out, heads, float(pair.log_scale))
    return heads, metrics


# ---- data: JSONL -> embeddings (once) -> .npz cache ------------------------------------------------------

def _read_jsonl(path: str):
    from .schema import build_pairs, to_text
    rows = [json.loads(l) for l in open(path) if l.strip()]
    states, texts, opts, y = [], [], [], []
    for n, r in enumerate(rows):
        o = list(r["options"])
        if len(set(o)) != len(o) or len(o) < 2:
            raise ValueError(f"line {n + 1}: options must be >= 2 distinct strings")
        lab = o.index(r["label"]) if isinstance(r["label"], str) else int(r["label"])
        if not 0 <= lab < len(o):
            raise ValueError(f"line {n + 1}: label out of range")
        q = {"type": "choice", "instructions": r.get("instructions"), "criteria": {t: t for t in o}}
        st, _, ot = build_pairs(r["state"], {"q": q})["q"]
        states.append(to_text(r["state"])), texts.append(st), opts.append(ot), y.append(lab)
    return states, texts, opts, np.array(y)


def encode_jsonl(path: str, cache: str | None = None, embedder=None, encoder: str | None = None) -> dict:
    """Encode `{state, options:[str], label}` lines once with the MLX embedder; cache to `.npz` (default `<path>.npz`).

    Returns {x [N,D], y [N], options ([K,D] if every example shares one list, else list of [k_i,D]),
    states [raw state text], option_texts [list per example]}. The cache is reused only if the file
    content and encoder name match.
    """
    from .embedder import DEFAULT_ENCODER
    encoder = encoder or DEFAULT_ENCODER
    cache = cache or path + ".npz"
    key = hashlib.sha256(open(path, "rb").read() + encoder.encode()).hexdigest()
    if os.path.exists(cache):
        z = np.load(cache)
        if str(z["key"]) == key:
            return _unpack(z)
    states, texts, opts, y = _read_jsonl(path)
    if embedder is None:
        from .embedder import MLXEmbedder
        embedder = MLXEmbedder(encoder)
    uniq = list(dict.fromkeys(t for o in opts for t in o))
    uv, _ = embedder.embed(uniq)
    x, _ = embedder.embed(texts)
    at = {t: i for i, t in enumerate(uniq)}
    kmax = max(map(len, opts))
    idx = np.full((len(opts), kmax), -1, np.int64)
    for i, o in enumerate(opts):
        idx[i, :len(o)] = [at[t] for t in o]
    np.savez(cache, key=key, x=x, y=y, uniq=np.array(uniq), uvec=uv, idx=idx, states=np.array(states))
    return _unpack(np.load(cache))


def _unpack(z) -> dict:
    uniq, uvec, idx = list(z["uniq"]), z["uvec"], z["idx"]
    rows = [r[r >= 0] for r in idx]
    same = all(len(r) == len(rows[0]) and (r == rows[0]).all() for r in rows)
    return {"x": z["x"], "y": z["y"], "states": list(z["states"]),
            "option_texts": [[uniq[i] for i in r] for r in rows],
            "options": uvec[rows[0]] if same else [uvec[r] for r in rows]}


# ---- baselines and the report card -----------------------------------------------------------------------

def _tok(t: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", t.lower())


def bm25_pick(query: str, options: list[str]) -> int:
    """Index of the option BM25-closest to `query` (options are the documents; k1=1.5, b=.75, as ft_baselines.py)."""
    docs = [Counter(_tok(o)) for o in options]
    lens = [sum(d.values()) for d in docs]
    n, avg = len(docs), max(sum(lens) / len(docs), 1e-9)
    df = Counter(w for d in docs for w in d)
    q = [w for w in _tok(query) if w in df]
    sc = [sum(math.log(1 + (n - df[w] + .5) / (df[w] + .5)) * d[w] * 2.5 / (d[w] + 1.5 * (.25 + .75 * ln / avg)) for w in q if w in d)
          for d, ln in zip(docs, lens)]
    return max(range(n), key=sc.__getitem__)


def report_card(train: dict, test: dict, trained: dict, zero_shot: dict | None = None) -> dict:
    """Trained accuracy on the held-out test vs majority-class and BM25 baselines. `train`/`test` are index-sliced
    encode_jsonl dicts (x, y, states, option_texts)."""
    lab = Counter(o[i] for o, i in zip(train["option_texts"], train["y"]))
    top = lab.most_common(1)[0][0]                                   # majority label, by option text
    majority = np.mean([(o.index(top) if top in o else 0) == i for o, i in zip(test["option_texts"], test["y"])])
    bm25 = np.mean([bm25_pick(s, o) == i for s, o, i in zip(test["states"], test["option_texts"], test["y"])])
    best = max(majority, bm25)
    rep = {"n_train": len(train["y"]), "n_test": len(test["y"]), "trained_acc": trained["acc"], "trained_top3": trained["top3"],
           "majority_class_acc": float(majority), "majority_label": top, "bm25_acc": float(bm25),
           "released_heads_zero_shot_acc": None if zero_shot is None else zero_shot["acc"]}
    rep["beats_baselines"] = bool(trained["acc"] > best)
    rep["verdict"] = (f"Trained heads beat the best baseline: {trained['acc']:.1%} vs {best:.1%}." if rep["beats_baselines"]
                      else f"No gain over baselines: trained {trained['acc']:.1%} vs best baseline {best:.1%}.")
    return rep


def _slice(d: dict, idx) -> dict:
    o = d["options"]
    return {"x": d["x"][idx], "y": d["y"][idx], "states": [d["states"][i] for i in idx],
            "option_texts": [d["option_texts"][i] for i in idx], "options": o if isinstance(o, np.ndarray) else [o[i] for i in idx]}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train CLM heads in pure MLX from a JSONL file of {state, options, label}.")
    ap.add_argument("data")
    ap.add_argument("--out", required=True, help="heads .safetensors to write (a .json sidecar is written next to it)")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--val-frac", type=float, default=0.1, help="fraction of the non-test data used for early stopping")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--schedule", choices=["constant", "warmup_cosine"], default="constant")
    ap.add_argument("--min-epochs-before-stop", type=int, default=0)
    ap.add_argument("--init-mode", choices=["released", "fresh"], default="released", help="fresh = random heads, same architecture")
    ap.add_argument("--center", action=argparse.BooleanOptionalAction, default=None,
                    help="center inputs on the train mean (default: auto = on for fresh heads, off for released)")
    ap.add_argument("--init", default=None, help="heads to warm-start from (default: the released head)")
    ap.add_argument("--encoder", default=None)
    ap.add_argument("--cache", default=None, help="embedding cache .npz (default: <data>.npz)")
    ap.add_argument("--report", default=None, help="report JSON path (default: <out>.report.json)")
    a = ap.parse_args(argv)
    if not a.out.endswith(".safetensors"):
        ap.error("--out must end in .safetensors")
    d = encode_jsonl(a.data, a.cache, encoder=a.encoder)
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(d["y"]))
    nt = max(1, int(len(perm) * a.test_frac + 1e-9))
    test, train = _slice(d, perm[:nt]), _slice(d, perm[nt:])
    heads, m = train_heads(train["x"], train["options"], train["y"], val_frac=a.val_frac, init=a.init, epochs=a.epochs,
                           lr=a.lr, batch=a.batch, patience=a.patience, seed=a.seed, rng=rng, out=a.out,
                           schedule=a.schedule, min_epochs_before_stop=a.min_epochs_before_stop, init_mode=a.init_mode,
                           center=a.center, test=(test["x"], test["options"], test["y"]))
    zero = evaluate_heads(Heads(a.init or default_head()), test["x"], test["options"], test["y"])
    rep = report_card(train, test, evaluate_heads(heads, test["x"], test["options"], test["y"]), zero)
    rep["training"] = {k: v for k, v in m.items() if k not in ("loss_history", "val_acc_history")}
    path = a.report or a.out[: -len(".safetensors")] + ".report.json"
    json.dump(rep, open(path, "w"), indent=1)
    print(f"test n={rep['n_test']}  trained {rep['trained_acc']:.1%}  majority {rep['majority_class_acc']:.1%}  "
          f"bm25 {rep['bm25_acc']:.1%}  released heads {zero['acc']:.1%}\n{rep['verdict']}\nheads: {a.out}  report: {path}")


if __name__ == "__main__":
    main()
