# Fine-tuning CLM's heads on a Mac (Banking77 routing)

```bash
python prepare_banking77.py              # 10,003 train / 3,080 test messages, 77 routes (CC-BY-4.0)
python ft_embed.py                       # one-time MLX encoding (3.5 min on M5 Pro, 8-bit)
CLM_CKPT=/path/CLM_v0.1-8B.pt python ft_train.py     # train heads, evaluate on the held-out test
CLM_CKPT=... python ft_fair.py           # linear-probe baseline + unseen-route test
python ft_baselines.py bm25 | laya-mlx <ckpt dir>
```

The test split is never used for training or model selection (early stopping uses 10% of train).
