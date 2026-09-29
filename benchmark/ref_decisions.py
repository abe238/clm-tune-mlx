"""Decision-level fidelity: the upstream CLM engine with a PyTorch (transformers, CPU, bf16 weights)
Qwen3-8B encoder over all 120 cases, compared with the MLX runs. Slow on purpose: it is the reference."""
import json, os, time
import numpy as np, torch
from transformers import AutoModel, AutoTokenizer
from clm.engine import Engine
from common import load_cases, Writer, HERE
class TorchEmbedder:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
        self.model = AutoModel.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16).eval()
        self.cache = {}
    def embed(self, texts):
        for t in dict.fromkeys(texts):
            if t not in self.cache:
                ids = self.tok.encode(t)[-2048:]
                with torch.no_grad():
                    h = self.model(input_ids=torch.tensor([ids])).last_hidden_state[0, -1].float().numpy()
                self.cache[t] = h / (np.linalg.norm(h) + 1e-12)
        return np.stack([self.cache[t] for t in texts]), 0
    def healthy(self): return True
eng = Engine(embedder=TorchEmbedder(), checkpoint=(os.environ.get("CLM_CKPT") or __import__("clm.heads", fromlist=["download"]).download()), device="cpu", action_cache="0")
w = Writer("clm-torch-ref")
for c in load_cases():
    t = time.time(); w.write(c, eng.answer(c["state"], {"q": c["question"]})["answers"]["q"], time.time() - t)
w.close()
ref = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(HERE, "results", "clm-torch-ref.jsonl"))}
out = {}
for tag in ("bf16", "q8", "q4"):
    m = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(HERE, "results", f"clm-mlx-{tag}.jsonl"))}
    diffs = [max(abs(m[i]["probs"][k] - ref[i]["probs"][k]) for k in ref[i]["probs"]) for i in ref]
    out[tag] = {"same_label": sum(m[i]["label"] == ref[i]["label"] for i in ref), "n": len(ref), "max_prob_diff": max(diffs), "mean_prob_diff": float(np.mean(diffs))}
out["torch_ref_accuracy"] = sum(r["correct"] for r in ref.values()) / len(ref)
json.dump(out, open(os.path.join(HERE, "results", "clm-decision-parity.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
