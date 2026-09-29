"""Download Banking77 (PolyAI, CC-BY-4.0) from legacy-datasets/banking77 and write banking77.json.
Label 53 is named `reverted_card_payment?` (with the question mark) in the dataset card."""
import json, re
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
R = "legacy-datasets/banking77"
card = open(hf_hub_download(R, "README.md", repo_type="dataset")).read()
labels = {int(i): n for i, n in re.findall(r"'?(\d+)'?: ([A-Za-z_]+\??)\n", card)}
assert len(labels) == 77, len(labels)
data = {s: pq.read_table(hf_hub_download(R, f"data/{s}-00000-of-00001.parquet", repo_type="dataset")).to_pylist() for s in ("train", "test")}
json.dump({"labels": {str(k): v for k, v in sorted(labels.items())}, **data}, open("banking77.json", "w"))
print({k: len(v) for k, v in data.items()})
