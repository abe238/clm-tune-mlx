"""Train CLM heads on Mind2Web next-action steps (per-step candidate sets), test on the frozen 150 web steps.
Train = train_7, train_9, and train_8 tasks with no step in the test set (task-level split).
Usage: /tmp/clmv/bin/python web_ft.py <encoder>"""
import json, os, re, sys, time, copy, random
from html.parser import HTMLParser
import numpy as np, torch
from clm.schema import build_pairs
from clm.heads import HeadPair
from clm_tune_mlx import MLXEmbedder
enc = sys.argv[1]; CK = os.environ["CLM_CKPT"]; random.seed(0); torch.manual_seed(0)
test = json.loads(open("web_cases.json").read()); test_ids = {c["id"] for c in test}
Q = {"type": "choice", "instructions": test[0]["question"]["instructions"], "criteria": {"x": "y"}}
class P(HTMLParser):
    def __init__(s): super().__init__(); s.stack, s.el = [], {}
    def handle_starttag(s, tag, attrs):
        a = dict(attrs); b = a.get("backend_node_id"); s.stack.append(b)
        if b: s.el[b] = {"tag": tag, "attrs": a, "text": []}
    def handle_endtag(s, tag):
        if s.stack: s.stack.pop()
    def handle_data(s, data):
        for b in s.stack:
            if b: s.el[b]["text"].append(data)
def describe(e):
    t = " ".join(" ".join(e["text"]).split())[:100]
    x = " ".join(v for v in (e["attrs"].get(k) for k in ("aria_label", "title", "placeholder", "alt", "value", "type", "name")) if v)[:80]
    return f"[{e['tag']}] {t} {x}".strip()
GENERIC = {"button", "link", "submit", "text", "search", "div", "span", "input", "svg", "img", "a", "true", "false"}
informative = lambda s: any(w not in GENERIC for w in re.findall(r"[a-zA-Z]{3,}", s.split("]", 1)[1].lower()))
steps = []
for f in ("m2w/data/train/train_7.json", "m2w/data/train/train_9.json", "m2w/data/train/train_8.json"):
    d = json.load(open(f))
    bad = {t["annotation_id"] for t in d for a in t["actions"] if a["action_uid"] in test_ids}
    for t in d:
        if t["annotation_id"] in bad: continue
        for k, a in enumerate(t["actions"]):
            if not a["pos_candidates"]: continue
            p = P(); p.feed(a["cleaned_html"]); g = a["pos_candidates"][0]["backend_node_id"]
            if g not in p.el or not informative(describe(p.el[g])): continue
            gt = describe(p.el[g]); negs = list(dict.fromkeys(describe(p.el[c["backend_node_id"]]) for c in a["neg_candidates"] if c["backend_node_id"] in p.el))
            negs = [n for n in negs if n != gt and informative(n)]; random.shuffle(negs)
            if len(negs) < 14: continue
            op = a["operation"]["op"] + (f" (type: {a['operation']['value']})" if a["operation"]["value"] else "")
            state = {"task": t["confirmed_task"], "website": t["website"], "previous_actions": t["action_reprs"][:k][-5:] or ["none"], "next_operation": op}
            steps.append({"task": t["annotation_id"], "state": build_pairs(state, {"q": Q})["q"][0], "cands": [gt] + negs[:40]})
tasks = sorted({s["task"] for s in steps}); random.shuffle(tasks); val_tasks = set(tasks[: max(1, len(tasks) // 10)])
print(f"{len(steps)} train steps from {len(tasks)} tasks", flush=True)
emb = MLXEmbedder(enc); t0 = time.time()
texts = list(dict.fromkeys([s["state"] for s in steps] + [c for s in steps for c in s["cands"]] +
                           [build_pairs(c["state"], {"q": Q})["q"][0] for c in test] + [v for c in test for v in c["question"]["criteria"].values()]))
V, _ = emb.embed(texts); ix = {t: i for i, t in enumerate(texts)}; V = torch.tensor(V); enc_s = time.time() - t0
print(f"encoded {len(texts)} texts in {enc_s/60:.1f} min", flush=True)
def group(items):   # padded [B, C] candidate indices with the gold at column 0
    C = max(len(s["cands"]) for s in items); idx = torch.zeros(len(items), C, dtype=torch.long); m = torch.zeros(len(items), C, dtype=torch.bool)
    for i, s in enumerate(items):
        idx[i, :len(s["cands"])] = torch.tensor([ix[c] for c in s["cands"]]); m[i, :len(s["cands"])] = True
    return torch.tensor([ix[s["state"]] for s in items]), idx, m
def test_acc(hp):
    ok = 0
    with torch.no_grad():
        for c in test:
            z = torch.nn.functional.normalize(hp.state_head(V[ix[build_pairs(c["state"], {"q": Q})["q"][0]]][None]), dim=-1)
            keys = list(c["question"]["criteria"]); A = torch.nn.functional.normalize(hp.action_head(V[[ix[c["question"]["criteria"][k]] for k in keys]]), dim=-1)
            ok += keys[int((A @ z[0]).argmax())] == c["gold"]
    return ok / len(test)
def step_acc(hp, items):
    s_i, c_i, m = group(items)
    with torch.no_grad():
        z = torch.nn.functional.normalize(hp.state_head(V[s_i]), dim=-1); a = torch.nn.functional.normalize(hp.action_head(V[c_i]), dim=-1)
        sc = (a @ z[:, :, None]).squeeze(-1).masked_fill(~m, -1e9)
    return (sc.argmax(-1) == 0).float().mean().item()
res = {"train_steps": sum(s["task"] not in val_tasks for s in steps), "encode_minutes": enc_s / 60}
base = HeadPair("x", CK, "cpu"); base.ensure(); res["released_head_test_acc"] = test_acc(base)
trn = [s for s in steps if s["task"] not in val_tasks]; val = [s for s in steps if s["task"] in val_tasks]
runs = []
for seed in (0, 1, 2):
    torch.manual_seed(seed); random.seed(seed)
    hp = HeadPair("x", CK, "cpu"); hp.ensure(); ps = list(hp.state_head.parameters()) + list(hp.action_head.parameters())
    sc = torch.tensor(float(np.log(hp.scale)), requires_grad=True); o = torch.optim.AdamW(ps + [sc], lr=5e-4); best, bs, bad = -1, None, 0; t0 = time.time()
    for ep in range(60):
        random.shuffle(trn)
        for b in range(0, len(trn), 64):
            s_i, c_i, m = group(trn[b:b + 64])
            z = torch.nn.functional.normalize(hp.state_head(V[s_i]), dim=-1); a = torch.nn.functional.normalize(hp.action_head(V[c_i]), dim=-1)
            logits = (sc.exp().clamp(max=100) * (a @ z[:, :, None]).squeeze(-1)).masked_fill(~m, -1e9)
            loss = torch.nn.functional.cross_entropy(logits, torch.zeros(len(s_i), dtype=torch.long)); o.zero_grad(); loss.backward(); o.step()
        v = step_acc(hp, val)
        if v > best: best, bs, bad = v, (copy.deepcopy(hp.state_head.state_dict()), copy.deepcopy(hp.action_head.state_dict())), 0
        else:
            bad += 1
            if bad >= 5: break
    hp.state_head.load_state_dict(bs[0]); hp.action_head.load_state_dict(bs[1])
    runs.append({"seed": seed, "val_acc": best, "epochs": ep + 1, "train_seconds": time.time() - t0, "test_acc": test_acc(hp)})
    if seed == 0: torch.save({"state_head": hp.state_head.state_dict(), "action_head": hp.action_head.state_dict(), "logit_scale": torch.tensor(np.log(hp.scale)), "cfg": hp.cfg}, "clm-mind2web-heads.pt")
res["finetuned"] = runs; res["finetuned_test_mean"] = float(np.mean([r["test_acc"] for r in runs]))
json.dump(res, open("results/web-ft.json", "w"), indent=1); print(json.dumps(res, indent=1))
