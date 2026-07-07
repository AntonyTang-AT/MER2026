# exp001: 官方 AffectGPT Baseline 复现

- **阶段**: 2
- **状态**: infrastructure ready — 训练待 GPU/CUDA 兼容
- **目标**: val EW-F1 ≥ 59%

## 基础设施（已完成）

| 组件 | 路径 |
|------|------|
| 配置同步 | `scripts/sync_mertools_config.sh` |
| Human 数据解压 | `scripts/extract_human_data.sh`（1532 样本 × 3 模态） |
| 训练封装 | `src/training/train_affectgpt.py` |
| 推理封装 | `src/inference/affectgpt_runner.py` |
| ovlabel | `src/inference/openset_extractor.py` |
| 评估报告 | `src/evaluation/baseline_report.py` |

## 命令

```bash
# 1. 配置 + 数据（已完成一次）
bash scripts/sync_mertools_config.sh
bash scripts/extract_human_data.sh

# 2. 训练 Human-OV（~26h，60 epoch）
bash scripts/train_affectgpt.sh human

# 3. 推理 + ovlabel + 评估
bash scripts/infer_baseline.sh
bash scripts/eval_baseline.sh path/to/epoch-openset.npz --split val
```

## 冒烟记录（2026-07-07）

- `sync_mertools_config`: PASS（config.py 已指向项目 data/models）
- `extract_human_data`: PASS（audio/video/openface 各 1532 文件）
- `train` 1 iter 冒烟: 模型加载成功，RTX 5090 上前向报 `CUDA error: no kernel image is available`（需升级 PyTorch 或换 GPU）

## 结果

| 模型 | 数据 | EW-F1 | 备注 |
|------|------|-------|------|
| affectgpt_human | Human-OV | — | 待训练完成 |

详见 [RESULTS.md](./RESULTS.md)（评估后自动生成）。
