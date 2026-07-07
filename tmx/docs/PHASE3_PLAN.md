# 阶段 3：L4 矛盾检测与动态路由（tmx 工作区）

> 对应 [PLAN.md](./PLAN.md) 阶段 3 | SIT 参考：`SIT_mds/planning/阶段4.md`

## 目标

在不依赖 GPU / L2 VA 模型的前提下，用**轻量代理分数 + 专家规则 + 权重查表**，为每个样本输出四模态融合权重，供后续 Prompt / AffectGPT 使用。

## 模态映射（MER2026 ↔ SIT）

| MER2026 | SIT L4 | 代理信号来源 |
|---------|--------|--------------|
| text | text | 英文字幕 lexicon 情感 |
| audio | speech | librosa 能量 / 谱质心 |
| face | micro | OpenFace 帧序列细粒度变化 |
| frame | macro | OpenFace 帧序列粗粒度均值 |

## 任务分解

| ID | 模块 | 文件 | 验收 |
|----|------|------|------|
| 3.1 | 模态代理分数 | `modality_scorer.py` | 四模态输出 (v,a,conf) ∈ [-1,1] |
| 3.2 | VA 距离 | `va_distance.py` | 4×4 距离矩阵 + max pair |
| 3.3 | 专家规则 | `expert_rules.py` | 5 类矛盾 + involved_modalities |
| 3.4 | 权重查表 | `weight_selector.py` | 权重和为 1 + routing_confidence |
| 3.5 | 批量路由 | `run_routing.py` + `scripts/run_routing.sh` | CSV/JSON 输出 |
| 3.6 | 分析报告 | `docs/reports/routing_analysis.md` | 100 样本统计（脚本生成） |
| 3.7 | 单元测试 | `tests/test_routing/` | pytest 无 GPU |

## 执行顺序

```
3.1 → 3.2 → 3.3 → 3.4 → 3.5 → 3.7 → 3.6
```

## 配置

- `config/routing/contradiction_rules.yaml` — 规则阈值与优先级
- `config/routing/weight_table.yaml` — 矛盾类型 → 融合权重

## 一键命令

```bash
cd /root/autodl-tmp/MER2026/tmx
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
pytest tests/test_routing/ -q
bash scripts/run_routing.sh --split human --limit 100
```
