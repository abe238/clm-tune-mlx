"""Summaries for the agentic suites (tools, web) and T-Rex."""
import json, glob, os, statistics
out = {}
for suite in ("tools", "web"):
    out[suite] = {}
    for f in sorted(glob.glob(f"results/{suite}/*.jsonl")):
        tag = os.path.basename(f)[:-6]; rows = [json.loads(l) for l in open(f)]
        by = {}
        for r in rows:
            by.setdefault(r["id"].split("|")[1] if "|" in r["id"] else "web-15", []).append(r)
        out[suite][tag] = {k: {"n": len(v), "accuracy": sum(r["correct"] for r in v) / len(v), "errors": sum(1 for r in v if r["error"]),
                               "p50_ms": 1000 * statistics.median(r["seconds"] for r in v),
                               "mean_gold_prob": statistics.mean(r["gold_prob"] or 0 for r in v)} for k, v in by.items()}
out["trex"] = {}
for f in sorted(glob.glob("../results/trex/*.json")):
    s = json.load(open(f)).get("summary", {})
    out["trex"][os.path.basename(f)[:-5]] = {k: s.get(k) for k in ("survived", "deaths", "mean_agreement_with_planner", "shield_interventions", "mean_decisions", "latency_ms_p50_median", "answers_discarded")}
for d in ("../results/trex/upstream-published",):
    for f in sorted(glob.glob(f"{d}/*.json")):
        s = json.load(open(f))["summary"]; out["trex"]["upstream-" + os.path.basename(f)[:-5]] = {k: s.get(k) for k in ("survived", "deaths", "mean_agreement_with_planner", "shield_interventions", "mean_decisions", "latency_ms_p50_median", "answers_discarded")}
json.dump(out, open("results/summary.json", "w"), indent=1)
for suite in ("tools", "web"):
    for tag, v in out[suite].items():
        print(suite, f"{tag:22s}", "  ".join(f"{k}={100*x['accuracy']:.1f}% err={x['errors']} p50={x['p50_ms']:.0f}ms" for k, x in v.items()))
for tag, v in out["trex"].items(): print("trex", tag, v)
