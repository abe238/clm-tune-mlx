"""A `POST /v1/systemone`-style typed-decision API for CLM on MLX, with automatic micro-batching.

One worker owns the model. Requests that arrive while it is busy wait in a queue; when it frees
up it takes everything waiting, sends all their state and option texts through the encoder in
one batched pass, then answers each request from the embedding cache. A lone request is served
at once (no batching timer), and a burst costs about one forward pass instead of one per request.

    clm-tune-mlx-serve --port 8700                      # 8-bit encoder
    clm-tune-mlx-serve --encoder Qwen/Qwen3-8B          # bf16
"""
from __future__ import annotations

import argparse
import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Batcher:
    """Serialises model access and coalesces whatever is queued into one encoder pass."""

    def __init__(self, engine=None, max_batch: int = 32, loader=None):
        # MLX >= 0.32 keeps streams per thread: arrays made on one thread can't be evaluated on another.
        # With `loader`, the engine is built on the worker thread that also runs every model call.
        self.engine, self.max_batch, self.loader = engine, max_batch, loader
        self.q: queue.Queue = queue.Queue()
        self.batches: list[int] = []            # sizes of recent batches, for /v1/stats
        self.ready = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()
        self.ready.wait()
        if self.load_error:
            raise self.load_error

    def answer(self, state, questions: dict, model: str | None = None) -> dict:
        job = {"state": state, "questions": questions, "model": model, "done": threading.Event()}
        self.q.put(job)
        job["done"].wait()
        if "error" in job:
            raise job["error"]
        return job["result"]

    def _run(self):
        from .schema import build_pairs
        self.load_error = None
        try:
            if self.loader is not None:
                self.engine = self.loader()
        except Exception as e:  # noqa: BLE001 - surfaced to the constructor
            self.load_error = e
        finally:
            self.ready.set()
        if self.load_error:
            return
        while True:
            jobs = [self.q.get()]
            while len(jobs) < self.max_batch:
                try:
                    jobs.append(self.q.get_nowait())
                except queue.Empty:
                    break
            self.batches = (self.batches + [len(jobs)])[-1000:]
            try:  # one batched encoder pass for every text in the batch (prefix sharing included)
                texts = [t for j in jobs for s, _, opts in build_pairs(j["state"], j["questions"]).values() for t in (s, *opts)]
                self.engine.embedder.embed(texts)
            except Exception:
                pass  # a malformed request fails on its own below, not the whole batch
            for j in jobs:
                try:
                    kw = {"model": j["model"]} if j["model"] else {}
                    j["result"] = self.engine.answer(j["state"], j["questions"], **kw)
                except Exception as e:  # noqa: BLE001 - reported to that caller only
                    j["error"] = e
                j["done"].set()


def make_handler(batcher: Batcher):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path.startswith("/v1/stats"):
                b = batcher.batches
                return self._send(200, {"batches": len(b), "mean_batch": sum(b) / len(b) if b else 0, "max_batch": max(b, default=0)})
            self._send(200, {"data": [{"id": m["name"]} for m in batcher.engine.models()]})

        def do_POST(self):
            if not self.path.startswith("/v1/systemone"):
                return self._send(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                self._send(200, batcher.answer(req["state"], req["questions"], req.get("model")))
            except (KeyError, ValueError) as e:
                self._send(400, {"error": str(e)})
            except Exception as e:  # noqa: BLE001
                self._send(500, {"error": repr(e)[:300]})

        def log_message(self, *a):
            pass

    return H


def main(argv=None):
    ap = argparse.ArgumentParser(description="CLM on MLX behind a /v1/systemone-style typed-decision API, with micro-batching.")
    ap.add_argument("--encoder", default=None, help="MLX encoder (default mlx-community/Qwen3-8B-8bit; Qwen/Qwen3-8B for bf16)")
    ap.add_argument("--checkpoint", "--heads", dest="checkpoint", default=None,
                    help="CLM heads: .safetensors (+ .json sidecar, as written by clm-tune-mlx-train) or upstream .pt (default: download the released head)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8700)
    ap.add_argument("--max-batch", type=int, default=32)
    a = ap.parse_args(argv)
    from .embedder import DEFAULT_ENCODER, load_engine
    batcher = Batcher(max_batch=a.max_batch, loader=lambda: load_engine(a.encoder or DEFAULT_ENCODER, a.checkpoint))
    print(f"clm-mlx serving on http://{a.host}:{a.port}", flush=True)
    ThreadingHTTPServer((a.host, a.port), make_handler(batcher)).serve_forever()


if __name__ == "__main__":
    main()
