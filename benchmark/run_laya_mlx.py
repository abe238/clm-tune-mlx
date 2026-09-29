"""mizorewww/laya-mlx 0.2.0 (verified wheel) on the local Laya checkpoints, over the frozen cases.
Usage: <laya-mlx venv python> run_laya_mlx.py <english|typed-decisions> <float16|float32>"""
import json, os, sys, time, warnings
import laya_mlx
from common import load_cases, Writer, HERE
name, dtype = sys.argv[1], sys.argv[2]
root = os.environ["LAYA_PATH"]  # local Laya checkpoint dir (typed-decisions/ inside)
t = time.time()
with warnings.catch_warnings(record=True) as warns:
    warnings.simplefilter("always")
    agent = laya_mlx.load(root if name == "english" else os.path.join(root, "typed-decisions"), dtype=dtype)
load_s = time.time() - t
cases = load_cases()
agent.predict(cases[0]["state"], {"q": cases[0]["question"]})  # warm-up (graph compile), not timed
w = Writer(f"laya-mlx-{name}-{'fp16' if dtype == 'float16' else 'fp32'}")
for c in cases:
    t = time.time()
    try:
        w.write(c, agent.predict(c["state"], {"q": c["question"]})["answers"]["q"], time.time() - t)
    except Exception as e:
        w.write(c, None, time.time() - t, repr(e))
w.close()
import mlx.core as mx
json.dump({"package": "laya-mlx 0.2.0", "checkpoint": name, "dtype": dtype, "load_seconds": load_s,
           "peak_gib": mx.get_peak_memory() / 2**30, "warnings": [str(x.message)[:300] for x in warns]},
          open(os.path.join(HERE, "results", f"meta-laya-mlx-{name}-{dtype}.json"), "w"))
print("done", name, dtype)
