"""Fairness checks: (1) linear probe on the same frozen embeddings; (2) unseen routes: train CLM heads
on 60 routes, test on messages from the 17 routes never seen in training."""
import json, os, copy, time
import numpy as np, torch
from clm.heads import HeadPair
torch.manual_seed(0); rng = np.random.default_rng(0)
Z = np.load("emb.npz"); opts, tr, te, ytr, yte = (torch.tensor(Z[k]) for k in ("opts", "tr", "te", "ytr", "yte"))
CK = os.environ["CLM_CKPT"]; res = {}
# (1) linear probe: 77-way logistic regression on frozen 4096-d encodings
W = torch.nn.Linear(4096, 77); opt = torch.optim.AdamW(W.parameters(), lr=1e-3, weight_decay=1e-4); t0 = time.time()
for ep in range(60):
    for b in torch.randperm(len(tr)).split(256):
        loss = torch.nn.functional.cross_entropy(W(tr[b] * 30), ytr[b]); opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad(): res["linear_probe_all"] = {"acc": (W(te * 30).argmax(-1) == yte).float().mean().item(), "train_seconds": time.time() - t0}
# (2) unseen routes
unseen = set(rng.permutation(77)[:17].tolist()); seen = [c for c in range(77) if c not in unseen]
def heads():
    hp = HeadPair("x", CK, "cpu"); hp.ensure(); return hp
def score(hp, X, y, allowed):
    with torch.no_grad():
        s = torch.nn.functional.normalize(hp.state_head(X), dim=-1) @ torch.nn.functional.normalize(hp.action_head(opts), dim=-1).T
        mask = torch.full((77,), -1e9); mask[list(allowed)] = 0
        return ((s + mask).argmax(-1) == y).float().mean().item()
tm = torch.tensor([int(v) in unseen for v in yte]); Xu, yu = te[tm], yte[tm]
res["unseen_routes"] = {"n_test": int(tm.sum()), "chance_among_77": 1 / 77,
                        "released_head_among_77": score(heads(), Xu, yu, range(77)), "released_head_among_17": score(heads(), Xu, yu, unseen)}
hp = heads(); ps = list(hp.state_head.parameters()) + list(hp.action_head.parameters()); o = torch.optim.AdamW(ps, lr=5e-4)
sel = torch.tensor([i for i in range(len(ytr)) if int(ytr[i]) not in unseen]); remap = {c: i for i, c in enumerate(seen)}
ys = torch.tensor([remap[int(v)] for v in ytr[sel]]); seen_opts = opts[torch.tensor(seen)]
for ep in range(20):
    for b in torch.randperm(len(sel)).split(256):
        zs = torch.nn.functional.normalize(hp.state_head(tr[sel][b]), dim=-1); za = torch.nn.functional.normalize(hp.action_head(seen_opts), dim=-1)
        loss = torch.nn.functional.cross_entropy(hp.scale * zs @ za.T, ys[b]); o.zero_grad(); loss.backward(); o.step()
hp.state_head.eval(); hp.action_head.eval()
res["unseen_routes"].update({"tuned_on_60_among_77": score(hp, Xu, yu, range(77)), "tuned_on_60_among_17_unseen": score(hp, Xu, yu, unseen),
                             "tuned_on_60_seen_route_msgs_among_77": score(hp, te[~tm], yte[~tm], range(77))})
json.dump(res, open("ft-fair.json", "w"), indent=1); print(json.dumps(res, indent=1))
