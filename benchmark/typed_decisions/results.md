# typed-decisions (choice-only) with A0, A4, A1, A2

| cfg | test acc % mean (min-max) | last-epoch test % | val acc % | kept epoch | train s |
|---|---|---|---|---|---|
| A0 {"epochs": 60, "patience": 5} | 55.8 (53.5-58.5) | 56.4 (55.8-57.2) | 61.5 (52.6-66.1) | 21/8/10 | 2.1 |
| A4 {"epochs": 80, "patience": 20, "schedule": "warmup_cosine", "init_mode": "fresh"} | 41.7 (41.3-41.8) | 41.8 (41.8-41.8) | 47.4 (40.6-51.2) | 5/4/4 | 2.9 |
| A1 {"epochs": 80, "patience": 20} | 62.5 (55.5-67.2) | 64.0 (58.8-67.2) | 68.8 (65.7-73.9) | 67/76/10 | 7.4 |
| A2 {"epochs": 80, "patience": 20, "schedule": "warmup_cosine"} | 59.7 (54.8-65.3) | 64.4 (62.2-66.8) | 65.3 (62.3-67.8) | 23/45/11 | 5.5 |

Choice-only: 1800 train / 600 test questions. Baselines on our choice-only test: global majority label `investigate` 12.3%; per-question-name majority 28.3%; uniform chance 23.3%.

Reference points (from the study/card, NOT directly comparable): study CLM heads Qwen3-8B 75.3% (PyTorch, 80 epochs); majority class 48.4%; teacher self-agreement ceiling 73.5%.

Caveat: the study scored ALL question types (noul, choice, score); this run scores choice questions only. Gold = `gold[q].label`. Evaluation only: weights trained on this data must not be shipped (Apache-2.0 data, eval use only per spec).
