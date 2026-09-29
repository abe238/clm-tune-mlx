"""Mind2Web, full pages: rank EVERY candidate element on the page for the 150 frozen steps.
Saves recall@K and top-10 shortlists (CLM and BM25) for the two-stage test.
Usage: /tmp/clmv/bin/python web_full.py <encoder> <tag>"""
import json, math, os, re, sys, time, statistics as st
from collections import Counter
from html.parser import HTMLParser
import numpy as np
from clm.schema import build_pairs
from clm_tune_mlx import load_engine
enc, tag = sys.argv[1], sys.argv[2]
frozen = {c["id"]: c for c in json.loads(open("web_cases.json").read())}
d = json.load(open("m2w/data/train/train_8.json"))
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
tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())
eng = load_engine(enc, os.environ.get("CLM_CKPT")); head = eng.heads["clm-latest"].ensure()
Q = "Which page element should be used for `next_operation` to make progress on `task`?"
K = (1, 5, 10, 20); hits = {m: {k: 0 for k in K} for m in ("clm", "bm25")}; sizes, lat, short = [], [], {"clm": [], "bm25": []}
for task in d:
    for k, a in enumerate(task["actions"]):
        if a["action_uid"] not in frozen: continue
        c = frozen[a["action_uid"]]; p = P(); p.feed(a["cleaned_html"])
        cand = {}
        for x in a["pos_candidates"] + a["neg_candidates"]:
            b = x["backend_node_id"]
            if b in p.el:
                t = describe(p.el[b])
                if t not in cand.values(): cand[b] = t
        gold_text = c["question"]["criteria"][c["gold"]]
        keys = list(cand); texts = [cand[b] for b in keys]
        if gold_text not in texts: continue
        g = texts.index(gold_text); sizes.append(len(texts))
        E, _ = eng.embedder.embed(texts); A = head.project_actions(E).cpu().numpy()
        s = build_pairs(c["state"], {"q": {"type": "choice", "instructions": Q, "criteria": {"x": "y"}}})["q"][0]
        t = time.time(); e, _ = eng.embedder.embed([s]); z = head.project_states(e).cpu().numpy()[0]; rc = np.argsort(-(A @ z)); lat.append(time.time() - t)
        docs = [tok(x) for x in texts]; df = Counter(w for dd in docs for w in set(dd)); N = len(docs); avg = sum(map(len, docs)) / N
        qt = tok(c["state"]["task"] + " " + c["state"]["next_operation"])
        bm = np.argsort(-np.array([sum(math.log(1 + (N - df[w] + .5) / (df[w] + .5)) * Counter(dd)[w] * 2.5 / (Counter(dd)[w] + 1.5 * (.25 + .75 * len(dd) / avg)) for w in qt if w in dd) for dd in docs]))
        for m, r in (("clm", rc), ("bm25", bm)):
            pos = int(np.where(r == g)[0][0])
            for kk in K: hits[m][kk] += pos < kk
            short[m].append({"id": c["id"], "state": c["state"], "gold_text": gold_text, "options": [texts[i] for i in r[:10]]})
n = len(sizes)
out = {"tag": tag, "steps": n, "candidates_per_page": {"median": st.median(sizes), "max": max(sizes)},
       "recall": {m: {f"@{k}": v / n for k, v in h.items()} for m, h in hits.items()}, "rank_ms_p50_state_only": 1000 * st.median(lat)}
json.dump(out, open(f"results/webfull-{tag}.json", "w"), indent=1)
json.dump(short, open(f"results/web-shortlists-{tag}.json", "w"))
print(json.dumps(out, indent=1))
