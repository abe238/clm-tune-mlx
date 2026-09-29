"""System python3 (needs `datasets`): typed-decisions `all` config -> raw.jsonl (one row per split row, JSON fields kept as strings)."""
import json
from datasets import load_dataset
d = load_dataset("LocalLLaMA/typed-decisions", "all")
with open("raw.jsonl", "w") as f:
    for sp in ("train", "test"):
        for r in d[sp]:
            f.write(json.dumps({**r, "split": sp}) + "\n")
