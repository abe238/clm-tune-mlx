"""Shared case loading, answer normalisation and result writing for every model runner."""
import hashlib, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN = "6ff2b519d17b61f3d125d2500b782a617ca462df180507bc85640f6a5e618445"

def load_cases():
    raw = open(os.path.join(HERE, "cases.json"), "rb").read()
    if hashlib.sha256(raw).hexdigest() != FROZEN:  # fixtures are frozen: refuse to run on an edited set
        sys.exit("cases.json changed since it was frozen; refusing to run")
    cases = json.loads(raw)
    if os.environ.get("SMOKE"):  # one case per type, for a wiring check
        cases = [next(c for c in cases if c["type"] == t) for t in ("noul", "choice", "score")]
    return cases

def normalise(case, ans):
    """-> (label, probs) from any /v1/systemone-style answer (CLM, Laya)."""
    t = case["type"]
    if t == "noul":
        p = float(ans["noul"])
        return ("true" if p >= 0.5 else "false"), {"false": 1 - p, "true": p}
    probs = {str(k): float(v) for k, v in (ans.get("probabilities") or {}).items()}
    if t == "choice":
        return ans["choice"], probs
    if probs:
        return max(probs, key=probs.get), probs
    n = len(case["question"]["criteria"])  # score with no distribution: nearest level
    return str(min(n - 1, max(0, round(float(ans["score"]))))), {}

class Writer:
    def __init__(self, model):
        self.path = os.path.join(HERE, "results", f"{model}.jsonl")
        self.model = model
        self.f = open(self.path, "w")
    def write(self, case, ans=None, seconds=None, error=None, extra=None):
        row = {"id": case["id"], "model": self.model, "seconds": seconds, "error": error, "raw": ans}
        if ans is not None and error is None:
            row["label"], row["probs"] = normalise(case, ans)
            row["correct"] = row["label"] == case["gold"]
            if case["type"] == "score":
                row["expected_level"] = float(ans["score"])
        row.update(extra or {})
        self.f.write(json.dumps(row, ensure_ascii=False) + "\n"); self.f.flush()
    def close(self):
        self.f.close()
