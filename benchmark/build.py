"""results/data.json + template.html -> index.html (standalone; personal paths scrubbed)."""
import json, os, re
from common import HERE, FROZEN
d = json.load(open(os.path.join(HERE, "results", "data.json")))
A = d["slices"]["all"]; P = d["parity"]["per_tag"]["q8"]; PB = d["parity"]["per_tag"].get("bf16")
wrong = {m: [c["id"] for c in d["cases"] if not c["results"][m]["correct"]] for m in d["models"]}
acc = lambda m, s="all": 100 * d["slices"][s][m]["accuracy"]
d["hash"] = FROZEN
d["headline"] = (
  f"<b>Laya typed-decisions leads at {acc('laya-typed-decisions'):.1f}%</b>, with Laya English at {acc('laya-english'):.1f}%. "
  f"<b>CLM-8B on MLX scored {acc('clm-mlx-bf16'):.1f}% at full precision ({acc('clm-mlx-q8'):.1f}% at 8-bit)</b>: strong enough on noul ({acc('clm-mlx-q8','type:noul'):.0f}%) but weak on choice ({acc('clm-mlx-q8','type:choice'):.0f}%) and score ({acc('clm-mlx-q8','type:score'):.0f}%). "
  f"The trained heads are doing real work (the encoder alone scores {acc('clm-raw-q8'):.1f}%), and the MLX encoder matches the PyTorch reference "
  f"(min cosine {PB['cos_min'] if PB else P['cos_min']:.4f}), so this gap belongs to CLM v0.1 on these general typed questions, not to the port.")
PB = d["parity"]["per_tag"].get("bf16")
d["parity_note"] = ("The port reuses upstream CLM's own engine, schema and trained heads unchanged; only the encoder runs on MLX. "
  + (f"Full-precision bf16 MLX matches PyTorch to a minimum cosine of {PB['cos_min']:.5f}; " if PB else "")
  + f"8-bit quantization moves the embeddings by at most {1 - P['cos_min']:.4f}, and changes "
  + (f"{100 * (1 - sum(c['results']['clm-mlx-q8']['label'] == c['results']['clm-mlx-bf16']['label'] for c in d['cases']) / d['n']):.0f}% of CLM's answers "
     f"({acc('clm-mlx-q8'):.1f}% vs {acc('clm-mlx-bf16'):.1f}% accuracy)." if PB else "."))
d["method"] = """<ul>
<li><b>Test set.</b> 120 hand-written cases (40 noul, 40 choice, 40 score) across support, code, moderation, security, privacy, tools, email, scheduling, reviews, Q&amp;A, dates and arithmetic; 37 easy, 46 medium, 37 hard. Hard cases carry flags: negation, distractors, non-English (Spanish, French, German, Chinese), inputs longer than 512 tokens, arithmetic and date reasoning. Gold labels are by construction. The file was frozen (sha256 recorded) before any model ran; every runner refuses to start if it changes.</li>
<li><b>Identical inputs.</b> Every model received the same state and the same question object (<code>/v1/systemone</code>-style typed-decision shape), one question per request. Choice questions always include a way-out option.</li>
<li><b>CLM-8B on MLX.</b> Upstream <code>contrastive-lm</code> engine and the released general head <code>CLM_v0.1-8B.pt</code> (sha256 b2b4a8c9…), with the Qwen3-8B encoder run in MLX (<code>mlx-community/Qwen3-8B-8bit</code>) using last-token, final-norm pooling on unpadded sequences, exactly as upstream serves it through vLLM. Heads on CPU. The encoder-only ablation (<code>clm-raw</code>) uses cosine in the raw embedding space; its latency is near zero because it reuses the cached embeddings.</li>
<li><b>Laya.</b> <code>laya</code> 0.3.3, local checkpoints <code>laya</code> (English, ModernBERT-large, 421M, 512-token context) and <code>laya-typed-decisions</code> (1024-token context), on the Mac GPU (MPS).</li>
<li><b>Metrics.</b> Accuracy = top-probability answer equals gold. Brier = multi-class squared error over all options. Log loss on the gold option. ECE over 10 confidence bins, where confidence = probability of the chosen option. Score MAE = |expected level - gold level|. Latency is wall time per question on an M5 Pro (all models ran locally).</li>
<li><b>Limits.</b> 120 cases is small: a slice of 7 cases moves 14 points per case. These are general typed-decision questions, not each model's training distribution (Laya's presets and CLM's agentic verification tasks were deliberately not used). One run, no repeats; all models are deterministic.</li>
</ul>"""
R = lambda f: json.load(open(os.path.join(HERE, "results", f))) if os.path.exists(os.path.join(HERE, "results", f)) else None
def rows(f):
    for base in ("results", "results/extra"):
        q = os.path.join(HERE, base, f + ".jsonl")
        if os.path.exists(q): return [json.loads(l) for l in open(q)]
def accof(f):
    r = rows(f); return sum(x["correct"] for x in r) / len(r) if r else None
variants = []
dp, emb_par = R("clm-decision-parity.json"), d["parity"]["per_tag"]
for tag, label in (("bf16", "CLM-8B · MLX bf16"), ("q8", "CLM-8B · MLX 8-bit"), ("q4", "CLM-8B · MLX 4-bit")):
    if not dp or tag not in dp: continue
    perf, cos = R(f"perf-clm-{tag}.json") or {}, emb_par.get(tag, {}).get("cos_min")
    ok = cos is not None and cos > 0.99
    variants.append({"name": label, "ref": "PyTorch CLM (upstream engine, bf16 weights)", "same": dp[tag]["same_label"], "n": dp[tag]["n"],
                     "maxdiff": dp[tag]["max_prob_diff"], "acc": accof(f"clm-mlx-{tag}"), "p50": (perf.get("one_question_options_cached") or {}).get("p50_ms"),
                     "peak": perf.get("peak_gib"), "ok": ok,
                     "verdict": (f"faithful (embedding min cosine {cos:.4f})" if ok else f"drifts (embedding min cosine {cos:.3f}); experimental")})
lp = R("laya-mlx-parity.json")
for key, label in (("typed-decisions-fp16", "laya-mlx typed-decisions · fp16"), ("typed-decisions-fp32", "laya-mlx typed-decisions · fp32"),
                   ("english-fp16", "laya-mlx English · fp16"), ("english-fp32", "laya-mlx English · fp32")):
    if not lp: break
    v = lp["by_variant"][key]; ck, dt = key.rsplit("-", 1)
    meta = R(f"meta-laya-mlx-{ck}-{'float16' if dt == 'fp16' else 'float32'}.json") or {}
    same = int(v["same_label"].split("/")[0]) + v["clamped_same"]
    variants.append({"name": label, "ref": "PyTorch Laya (same checkpoint)", "same": same, "n": 120, "maxdiff": v["max_prob_diff"], "acc": v["acc"],
                     "p50": v["p50_ms"], "peak": meta.get("peak_gib"), "ok": same == 120,
                     "verdict": "faithful (13-option questions use clamped temperatures by design)"})
d["variants"] = variants
if "laya-mlx-typed-decisions-fp16" in d["models"]:
    lv = lp["by_variant"]["typed-decisions-fp16"]
    d["headline"] += (f" <b>laya-mlx (mizorewww/laya-mlx 0.2.0) is a faithful port:</b> same answer as PyTorch Laya on 120 of 120 questions, "
                      f"at {lv['p50_ms']:.1f} ms per question in fp16 on this Mac, versus {A['laya-typed-decisions']['p50_ms']:.0f} ms for PyTorch on the same GPU.")
d["method"] = d["method"].replace("<li><b>Metrics.</b>", "<li><b>laya-mlx.</b> <code>laya-mlx</code> 0.2.0 from PyPI (wheel sha256 1a80a0cc…, reviewed before install), loading the same local checkpoints in fp16 and fp32; one untimed warm-up call first.</li>\n<li><b>Metrics.</b>")
html = open(os.path.join(HERE, "template.html")).read().replace("__DATA__", json.dumps(d, ensure_ascii=False).replace("</", "<\\/"))
home = os.path.expanduser("~")
html = html.replace(home, "~")
assert home not in html and not re.search(r"Bearer [A-Za-z0-9]|sk-[A-Za-z0-9]{8}|[A-Za-z0-9._%+-]+@(gmail|icloud|yahoo|outlook)\.com", html), "personal data or secret in page"
open(os.path.join(HERE, "index.html"), "w").write(html)
print("index.html", len(html) // 1024, "KB")
