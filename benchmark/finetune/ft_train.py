"""Fine-tune CLM's projection heads on Banking77 routes, on a Mac, from cached MLX embeddings.
Recipe follows upstream finetune.py 'choice': warm start from the released head, softmax over each
question's own candidates (the 77 routes), hard targets, AdamW lr 5e-4, batch 256, <=20 epochs,
early stop on a 10% validation split of TRAIN (the test set is never seen during training)."""
import json, os, sys, time, copy
import numpy as np, torch
from clm.heads import HeadPair
torch.manual_seed(0); rng = np.random.default_rng(0)
Z = np.load("emb.npz"); opts, tr, te, ytr, yte = (torch.tensor(Z[k]) for k in ("opts", "tr", "te", "ytr", "yte"))
CK = os.environ.get("CLM_CKPT")
def fresh():
    hp = HeadPair("ft", CK, "cpu"); hp.ensure(); return hp
def evaluate(hp, X, y):
    with torch.no_grad():
        zs = torch.nn.functional.normalize(hp.state_head(X), dim=-1); za = torch.nn.functional.normalize(hp.action_head(opts), dim=-1)
        s = zs @ za.T; top = s.topk(3, dim=-1).indices
    return {"acc": (top[:, 0] == y).float().mean().item(), "top3": (top == y[:, None]).any(-1).float().mean().item()}
def train(idx, epochs=60, lr=5e-4, batch=256, patience=5):
    hp = fresh(); params = list(hp.state_head.parameters()) + list(hp.action_head.parameters())
    for p in params: p.requires_grad_(True)
    hp.state_head.train(); hp.action_head.train()
    scale = torch.tensor(float(np.log(hp.scale)), requires_grad=True)
    opt = torch.optim.AdamW(params + [scale], lr=lr)
    idx = rng.permutation(idx); nv = max(1, len(idx) // 10); val, trn = idx[:nv], idx[nv:]
    best, best_state, bad, t0 = -1, None, 0, time.time()
    for ep in range(epochs):
        for b in range(0, len(trn), batch):
            j = torch.tensor(trn[b:b + batch])
            zs = torch.nn.functional.normalize(hp.state_head(tr[j]), dim=-1); za = torch.nn.functional.normalize(hp.action_head(opts), dim=-1)
            loss = torch.nn.functional.cross_entropy(scale.exp().clamp(max=100) * zs @ za.T, ytr[j])
            opt.zero_grad(); loss.backward(); opt.step()
        hp.state_head.eval(); hp.action_head.eval()
        v = evaluate(hp, tr[torch.tensor(val)], ytr[torch.tensor(val)])["acc"]
        hp.state_head.train(); hp.action_head.train()
        if v > best: best, best_state, bad = v, (copy.deepcopy(hp.state_head.state_dict()), copy.deepcopy(hp.action_head.state_dict())), 0
        else:
            bad += 1
            if bad >= patience: break
    hp.state_head.load_state_dict(best_state[0]); hp.action_head.load_state_dict(best_state[1]); hp.state_head.eval(); hp.action_head.eval()
    return hp, {"train_examples": len(trn), "val_acc": best, "epochs": ep + 1, "train_seconds": time.time() - t0}
res = {"zero_shot": evaluate(fresh(), te, yte)}
for per in (5, 20, None):
    idx = np.arange(len(ytr)) if per is None else np.concatenate([rng.permutation(np.where(ytr.numpy() == c)[0])[:per] for c in range(77)])
    hp, info = train(idx)
    res[f"{'all' if per is None else per}_per_route"] = {**info, **evaluate(hp, te, yte)}
    if per is None:
        torch.save({"state_head": hp.state_head.state_dict(), "action_head": hp.action_head.state_dict(), "logit_scale": torch.tensor(np.log(hp.scale)), "cfg": hp.cfg}, "clm-banking77-heads.pt")
json.dump(res, open("ft-results.json", "w"), indent=1)
print(json.dumps(res, indent=1))
