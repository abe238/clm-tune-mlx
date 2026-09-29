"""Run a frozen agentic question set against one backend. Usage:
  python run_suite.py <cases.json> clm <encoder> <tag> | laya-mlx <ckpt dir> <tag>"""
import hashlib, json, os, sys, time
cases_path, kind, target, tag = sys.argv[1:5]
raw = open(cases_path, "rb").read()
want = open(cases_path.replace(".json", ".sha256")).read().split()[0]
if hashlib.sha256(raw).hexdigest() != want:
    sys.exit("cases changed since frozen; refusing to run")
cases = json.loads(raw)
if kind == "clm":
    from clm_tune_mlx import load_engine
    eng = load_engine(target, os.environ.get("CLM_CKPT")); ask = lambda c: eng.answer(c["state"], {"q": c["question"]})["answers"]["q"]
elif kind == "laya-mlx":
    import laya_mlx
    agent = laya_mlx.load(target, dtype="float16"); ask = lambda c: agent.predict(c["state"], {"q": c["question"]})["answers"]["q"]
else:
    sys.exit(f"unknown backend {kind}; use clm or laya-mlx")
ask(cases[0])  # warm-up, untimed
suite = os.path.basename(cases_path).split("_cases")[0]
os.makedirs(f"results/{suite}", exist_ok=True)
with open(f"results/{suite}/{tag}.jsonl", "w") as f:
    for c in cases:
        t = time.time()
        try:
            a = ask(c); p = a.get("probabilities") or {}
            row = {"id": c["id"], "label": a.get("choice"), "gold_prob": p.get(c["gold"]), "seconds": time.time() - t, "error": None}
        except Exception as e:
            row = {"id": c["id"], "label": None, "gold_prob": None, "seconds": time.time() - t, "error": repr(e)[:200]}
        row["correct"] = row["label"] == c["gold"]
        f.write(json.dumps(row) + "\n"); f.flush()
print("done", suite, tag)
