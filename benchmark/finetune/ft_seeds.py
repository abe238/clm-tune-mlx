"""Headline number: CLM heads trained on all of Banking77 train, 3 seeds, 60-epoch cap with early
stopping on 1,000 held-out train messages; scored on the full test and on a fixed 1,000-message sample."""
import json, random, copy, os, numpy as np, torch
from clm.heads import HeadPair
Z = np.load("emb.npz"); opts, tr, te, ytr, yte = (torch.tensor(Z[k]) for k in ("opts", "tr", "te", "ytr", "yte"))
d = json.load(open("banking77.json")); idx = torch.tensor([d["test"].index(r) for r in random.Random(2).sample(d["test"], 1000)])
CK = os.environ["CLM_CKPT"]
def run(seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    hp = HeadPair("x", CK, "cpu"); hp.ensure(); ps = list(hp.state_head.parameters()) + list(hp.action_head.parameters())
    sc = torch.tensor(float(np.log(hp.scale)), requires_grad=True); o = torch.optim.AdamW(ps + [sc], lr=5e-4)
    perm = rng.permutation(len(ytr)); val, trn = torch.tensor(perm[:1000]), perm[1000:]; best, bs, bad = -1, None, 0
    def ev(X, y):
        with torch.no_grad():
            s = torch.nn.functional.normalize(hp.state_head(X), dim=-1) @ torch.nn.functional.normalize(hp.action_head(opts), dim=-1).T
        return (s.argmax(-1) == y).float().mean().item()
    for ep in range(60):
        for b in torch.tensor(rng.permutation(trn)).split(256):
            zs = torch.nn.functional.normalize(hp.state_head(tr[b]), dim=-1); za = torch.nn.functional.normalize(hp.action_head(opts), dim=-1)
            loss = torch.nn.functional.cross_entropy(sc.exp().clamp(max=100) * zs @ za.T, ytr[b]); o.zero_grad(); loss.backward(); o.step()
        v = ev(tr[val], ytr[val])
        if v > best: best, bs, bad = v, (copy.deepcopy(hp.state_head.state_dict()), copy.deepcopy(hp.action_head.state_dict())), 0
        else:
            bad += 1
            if bad >= 5: break
    hp.state_head.load_state_dict(bs[0]); hp.action_head.load_state_dict(bs[1])
    return {"full_test": ev(te, yte), "same_1000": ev(te[idx], yte[idx]), "epochs": ep + 1}
res = {f"seed{s}": run(s) for s in (0, 1, 2)}
json.dump(res, open("ft-seeds.json", "w"), indent=1); print(json.dumps(res, indent=1))
