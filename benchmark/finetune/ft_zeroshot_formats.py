"""Released head with three route wordings (is it just the labels?)."""
import json, os, numpy as np, torch
from clm_tune_mlx import MLXEmbedder
from clm.heads import HeadPair
labels = json.load(open("embed-meta.json"))["labels"]; Z = np.load("emb.npz"); te, yte = torch.tensor(Z["te"]), torch.tensor(Z["yte"])
emb = MLXEmbedder(); hp = HeadPair("x", os.environ["CLM_CKPT"], "cpu"); hp.ensure()
clean = lambda l: l.replace("_", " ").replace("?", "").strip()
out = {}
for name, texts in {"two-word route name": [clean(l).capitalize() for l in labels], "raw label": labels,
                    "one-sentence description": [f"The customer's message is about {clean(l)}." for l in labels]}.items():
    O = torch.tensor(emb.embed(texts)[0])
    with torch.no_grad():
        s = torch.nn.functional.normalize(hp.state_head(te), dim=-1) @ torch.nn.functional.normalize(hp.action_head(O), dim=-1).T
    out[name] = (s.argmax(-1) == yte).float().mean().item()
json.dump(out, open("zeroshot-formats.json", "w"), indent=1); print(out)
