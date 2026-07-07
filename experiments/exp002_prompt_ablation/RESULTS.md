# exp002 Prompt 消融结果

> Human val（306 条），hold-out 模型 `20260707185`，best epoch **50**，2026-07-08

## 消融矩阵

| ID | Description Prompt | Openset Prompt | 后处理 | EW-F1 | Precision | Recall |
|----|-------------------|----------------|--------|------:|----------:|-------:|
| **A** | official | official | official | **60.71%** | 60.77% | 60.67% |
| **B** | routing | official | official | 51.30% | 57.13% | 46.60% |
| **C** | official | ew_aware | ew | 55.79% | 59.95% | 52.18% |
| **D** | routing | ew_aware | ew | 51.02% | 56.81% | 46.35% |

## 解读

- **A（可信 baseline）**：修复 train/val 泄漏后，official 全链路 EW-F1 ≈ **60.7%**（epoch sweep 最优）。
- **B vs A（−9.4pt）**：routing description prompt 显著低于 official reason，当前 routing 模板/策略未带来增益。
- **C vs A（−4.9pt）**：同 official reason 下，ew_aware ovlabel + ew 后处理反而下降。
- **B vs D（+0.3pt）**：routing reason 路径上 ew_aware 几乎无收益。

## 对比 subset

| 对比 | Δ EW-F1 | 结论 |
|------|--------:|------|
| B vs A | −9.4pt | routing description 显著负向 |
| C vs A | −4.9pt | ew_aware 后处理负向 |
| B vs D | +0.3pt | ew_aware 在 routing 路径上可忽略 |

## 产物路径

| Variant | openset npz |
|---------|-------------|
| A | `results-mer2026ov-human/...20260707185/checkpoint_000050_loss_0.002-openset.npz` |
| B | `results-mer2026ov-human-prompts/...20260707185/checkpoint_000050_loss_0.002-openset-B.npz` |
| C | `results-mer2026ov-human/...20260707185/checkpoint_000050_loss_0.002-openset-C.npz` |
| D | `results-mer2026ov-human-prompts/...20260707185/checkpoint_000050_loss_0.002-openset-D.npz` |

日志：`logs/rerun_leakage_fixed_eval.log`（A/C）、`logs/rerun_exp002_BD_holdout.log`（B/D）
