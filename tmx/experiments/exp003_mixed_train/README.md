# exp003: 混合训练实验

- **阶段**: 5
- **状态**: infrastructure ready — 待 MER-Caption+ 媒体 + GPU 训练
- **目标**: val EW-F1 ≥ 62%（相对 exp001 +2~3%）

## 策略矩阵

| ID | 数据 | 初始化 | 命令 |
|----|------|--------|------|
| M0 | human only | scratch | exp001 引用 |
| M1 | mercaptionplus | scratch | `bash scripts/run_exp003.sh --variant M1` |
| M2 | mercaptionplus filtered | scratch | `bash scripts/run_exp003.sh --variant M2` |
| M3 | human + mercaptionplus | scratch | `bash scripts/run_exp003.sh --variant M3` |
| M4 | human + mercaptionplus | Human best ckpt | `bash scripts/run_exp003.sh --variant M4 --init-ckpt PATH` |
| M5 | human + mercaptionplus filtered | Human best ckpt | `bash scripts/run_exp003.sh --variant M5 --init-ckpt PATH` |

## 前置步骤

```bash
cd /root/autodl-tmp/MER2026/tmx
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# 1. 过滤 MER-Caption+ CSV（无需媒体）
bash scripts/filter_mercaptionplus.sh

# 2. 同步 mixed yaml 到 MERTools
bash scripts/sync_mertools_config.sh

# 3. Human epoch 选择（训练完成后）
bash scripts/sweep_epochs.sh --train-run human --epochs 10-60 --skip 5 --dry-run

# 4. 下载 MER-Caption+ 媒体（训练 M1/M2/M3+ 前）
# bash scripts/download_mercaptionplus.sh
```

## 结果

| Variant | EW-F1 | 备注 |
|---------|-------|------|
| M0 | — | exp001 baseline |
| M1 | — | mercaptionplus only |
| M2 | — | filtered |
| M3 | — | mixed scratch |
| M4 | — | mixed finetune |
| M5 | — | mixed filtered finetune |

汇总见 [summary.csv](./summary.csv)。
