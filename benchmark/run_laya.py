"""Laya over the frozen cases. Usage: python run_laya.py <english|typed-decisions> [device]
LAYA_PATH points at a local checkpoint dir; default downloads convaiinnovations/laya from Hugging Face."""
import json, os, sys, time
import laya
from common import load_cases, Writer, HERE
name = sys.argv[1]; device = sys.argv[2] if len(sys.argv) > 2 else "mps"
root = os.environ.get("LAYA_PATH")
t = time.time()
if root:
    agent = laya.load(root if name == "english" else os.path.join(root, "typed-decisions"), device=device)
else:
    agent = laya.load("convaiinnovations/laya", device=device, subfolder=None if name == "english" else "typed-decisions")
load_s = time.time() - t
w = Writer(f"laya-{name}")
for c in load_cases():
    t = time.time()
    try:
        w.write(c, agent.predict(c["state"], {"q": c["question"]})["answers"]["q"], time.time() - t)
    except Exception as e:
        w.write(c, None, time.time() - t, repr(e))
w.close()
json.dump({"checkpoint": name, "device": str(agent.device), "load_seconds": load_s, "laya": laya.__version__},
          open(os.path.join(HERE, "results", f"meta-laya-{name}.json"), "w"))
