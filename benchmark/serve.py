"""Local live-test server: the benchmark page plus POST /api/ask for CLM-MLX and Laya.
Binds 127.0.0.1 only.
Usage: python serve.py, then open http://127.0.0.1:8765"""
import json, os, sys, threading, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import normalise
LOCK = threading.Lock()  # one model call at a time: MLX and torch are not shared across threads
MODELS = {}

def clm():
    if "clm" not in MODELS:
        from clm_tune_mlx import load_engine
        MODELS["clm"] = load_engine(os.environ.get("CLM_ENCODER", "mlx-community/Qwen3-8B-8bit"), os.environ.get("CLM_CKPT"))
    return MODELS["clm"]

def laya_agent(name):
    if name not in MODELS:
        import laya
        root = os.environ.get("LAYA_PATH")
        sub = None if name == "laya-english" else "typed-decisions"
        MODELS[name] = (laya.load(root if not sub else os.path.join(root, sub), device="cpu") if root
                        else laya.load("convaiinnovations/laya", device="cpu", subfolder=sub))
    return MODELS[name]

def ask(model, state, q):
    if model == "clm":
        return clm().answer(state, {"q": q})["answers"]["q"]
    if model.startswith("laya-"):
        return laya_agent(model).predict(state, {"q": q})["answers"]["q"]
    raise ValueError(f"unknown model {model}")

class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, open(os.path.join(HERE, "index.html"), "rb").read(), "text/html; charset=utf-8")
        if self.path == "/api/health":
            return self._send(200, {"loaded": sorted(MODELS)})
        self._send(404, {"error": "not found"})
    def do_POST(self):
        if self.path != "/api/ask":
            return self._send(404, {"error": "not found"})
        if self.headers.get("Origin") not in (None, "http://127.0.0.1:8765", "http://localhost:8765"):
            return self._send(403, {"error": "cross-origin requests refused"})
        try:
            req = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length", 0)), 200_000)))
            state, q, models = req["state"], req["question"], req.get("models") or ["clm", "laya-typed-decisions", "laya-english"]
            case = {"type": q["type"], "question": q}
        except Exception as e:
            return self._send(400, {"error": f"bad request: {e}"})
        out = {}
        for m in models:
            t = time.time()
            try:
                with LOCK:
                    a = ask(m, state, q)
                label, probs = normalise(case, a)
                out[m] = {"label": label, "probs": probs, "seconds": time.time() - t,
                          "expected_level": float(a["score"]) if q["type"] == "score" else None}
            except Exception as e:
                out[m] = {"error": f"{type(e).__name__}: {e}"[:300], "seconds": time.time() - t}
        self._send(200, {"results": out})
    def log_message(self, *a):
        pass

if __name__ == "__main__":
    print("http://127.0.0.1:8765")
    ThreadingHTTPServer(("127.0.0.1", 8765), H).serve_forever()
