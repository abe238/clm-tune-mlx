"""Choice-only pairs via clm_tune_mlx.schema.build_pairs -> one MLX encode job -> emb.npz.
NOTE: build_pairs returns (state_text, option_keys, option_texts). Gold = gold[q]["label"] (the `all` config has no
`<q>__label` column; label is the gold argmax key, same thing)."""
import json, time
import numpy as np
from clm_tune_mlx.schema import build_pairs
from clm_tune_mlx import MLXEmbedder
rows = [json.loads(l) for l in open("raw.jsonl")]
items = []
for r in rows:
    state, qs, gold = json.loads(r["state"]), json.loads(r["questions"]), json.loads(r["gold"])
    for q, spec in qs.items():
        if spec["type"] != "choice": continue
        st, keys, texts = build_pairs(state, {q: spec})[q]
        lab = gold[q]["label"]
        assert lab in keys, (r["id"], q, lab, keys)
        items.append(dict(id=r["id"], split=r["split"], q=q, workflow=r["workflow"], state=st, keys=keys, texts=texts, y=keys.index(lab)))
print(len(items), "choice questions", {s: sum(i["split"] == s for i in items) for s in ("train", "test")})
uniq = list(dict.fromkeys([t for i in items for t in i["texts"]]))
states = [i["state"] for i in items]
emb = MLXEmbedder(); t = time.time()
uv, _ = emb.embed(uniq); x, _ = emb.embed(states)
secs = time.time() - t
at = {t: n for n, t in enumerate(uniq)}
kmax = max(len(i["texts"]) for i in items)
idx = np.full((len(items), kmax), -1, np.int64)
for n, i in enumerate(items): idx[n, :len(i["texts"])] = [at[t] for t in i["texts"]]
np.savez("emb.npz", x=x, uvec=uv, idx=idx, y=np.array([i["y"] for i in items]), split=np.array([i["split"] for i in items]),
         case=np.array([i["id"] for i in items]), q=np.array([i["q"] for i in items]), workflow=np.array([i["workflow"] for i in items]))
json.dump({"encoder": emb.model.__class__.__name__, "embed_seconds": secs, "n": len(items)}, open("embed-meta.json", "w"))
print(f"encoded {len(items)} states + {len(uniq)} options in {secs/60:.1f} min")
