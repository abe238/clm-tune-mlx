# Banking77 trainer experiments (3 seeds, all train data)

| cfg | config | test acc % mean (min-max) | last-epoch test % | val acc % | kept epoch | epochs run | train s (mean) |
|---|---|---|---|---|---|---|---|
| A0 | {"epochs": 60, "patience": 5} | 85.04 (83.96-86.27) | 84.74 (83.15-85.88) | 85.17 (84.60-86.20) | 16/21/30 | 21/26/35 | 11.1 |
| A1 | {"epochs": 80, "patience": 20} | 86.47 (85.68-86.92) | 86.89 (86.56-87.08) | 86.97 (85.40-88.10) | 67/64/38 | 80/80/58 | 31.1 |
| A2 | {"epochs": 80, "patience": 20, "schedule": "warmup_cosine"} | 86.60 (86.36-86.85) | 86.76 (86.40-86.95) | 86.90 (86.00-87.40) | 48/67/61 | 68/80/80 | 31.8 |
| A3 | {"epochs": 80, "patience": 20, "init_mode": "fresh"} | 82.20 (78.90-84.16) | 79.92 (76.79-82.95) | 82.57 (78.30-84.90) | 39/79/71 | 59/80/80 | 30.3 |
| A4 | {"epochs": 80, "patience": 20, "schedule": "warmup_cosine", "init_mode": "fresh"} | 88.04 (87.63-88.34) | 88.08 (87.73-88.28) | 87.73 (86.80-88.70) | 74/76/71 | 80/80/80 | 34.9 |
