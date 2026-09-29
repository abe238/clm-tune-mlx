"""CLM-8B with the Qwen3-8B encoder in MLX (clm_tune_mlx) + the released head, over the frozen cases.
Usage: CLM_CKPT=/path/CLM_v0.1-8B.pt python run_clm_mlx.py [encoder] [tag]   (head downloads if CLM_CKPT unset)"""
import json, os, sys, time
import numpy as np
from clm.engine import Engine, RAW_MODEL, DEFAULT_MODEL
from clm.heads import download
from clm_tune_mlx import MLXEmbedder
from common import load_cases, Writer, HERE

encoder = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen3-8B-8bit"
tag = sys.argv[2] if len(sys.argv) > 2 else "q8"
head = os.environ.get("CLM_CKPT") or download()
emb = MLXEmbedder(encoder)
eng = Engine(embedder=emb, checkpoint=head, device="cpu", action_cache="0")
writers = {DEFAULT_MODEL: Writer(f"clm-mlx-{tag}"), RAW_MODEL: Writer(f"clm-raw-{tag}")}
for c in load_cases():
    for m, w in writers.items():
        t = time.time()
        try:
            w.write(c, eng.answer(c["state"], {"q": c["question"]}, model=m)["answers"]["q"], time.time() - t)
        except Exception as e:
            w.write(c, None, time.time() - t, repr(e))
for w in writers.values():
    w.close()
texts = list(emb.cache)  # every distinct text, for parity.py (not committed: large)
np.savez_compressed(os.path.join(HERE, "results", f"emb-{tag}.npz"), texts=np.array(texts, dtype=object), vecs=np.stack([emb.cache[t] for t in texts]))
import mlx.core as mx
json.dump({"tag": tag, "path": encoder, "load_seconds": emb.load_seconds, "peak_gib": mx.get_peak_memory() / 2**30,
           "head": os.path.basename(head), "n_texts": len(texts)}, open(os.path.join(HERE, "results", f"meta-clm-{tag}.json"), "w"))
