# Agentic benchmarks

- `tools_cases.json`: 150 requests from the Berkeley Function Calling Leaderboard (BFCL v3 `simple`, Apache-2.0, gorilla-llm), each with the correct function among 10 random, 40 random or 10 look-alike candidates from BFCL's pool of 589 functions.
- `web_cases.json`: 150 steps from Mind2Web (CC-BY-4.0, OSU NLP Group, `train_8.json`): pick the page element for the next action among 15 real elements from that page.


```bash
python run_suite.py tools_cases.json clm Qwen/Qwen3-8B clm-mlx-bf16
python run_suite.py web_cases.json laya-mlx /path/to/laya/typed-decisions laya-mlx-typed-fp16
python summarize.py
```
