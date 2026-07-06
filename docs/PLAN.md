# MER2026 Track2（MER-FG）参赛方案总规划

> 文档版本：v1.0  
> 最后更新：2026-07-06  
> 工作目录：`/root/autodl-tmp/MER2026`

---

## 一、项目概述

### 1.1 任务定义

**MER-FG（Fine-grained Emotion）** 要求对多模态视频 clip 预测**任意数量、任意类别**的开放词汇情感标签，评估指标为 **Emotion Wheel 平均 F1（EW-F1）**。

| 维度 | 内容 |
|------|------|
| 输入 | 视频、音频、字幕（`subtitle_chieng.csv`）、OpenFace 人脸特征 |
| 训练集 | Human-OV（1,532 人工标注）+ MER-Caption+（31,327 自动标注） |
| 提交集 | `track1_track2_candidate.csv` 中 20,000 个候选样本 |
| 评估 | 5 个情感轮平均 F1，越高越好 |
| 提交平台 | [CodaBench Track 2](https://www.codabench.org/competitions/17196) |
| 结果截止 | **2026-07-13**（AoE） |

### 1.2 核心思路

官方 baseline 验证了两条路线：

| 路线 | 测试 EW-F1 |
|------|------------|
| Zero-shot MLLM（描述 → 抽标签） | ~30%–47% |
| AffectGPT 微调（四模态 → ov 标签） | ~59%–60% |

**本项目策略：** 以 **AffectGPT 为主模型**，迁移 SIT 项目的 **模态矛盾感知 + 动态路由 + Prompt 工程**，在 ~60% 基线上争取 **63%–68%**。

### 1.3 技术路线（三阶段流水线）

```
Stage A：模态感知（SIT-L4 迁移）
  轻量矛盾检测 → fusion_weights (text, audio, face, frame)
        ↓
Stage B：情感理解（MERTools AffectGPT）
  multiface + audio + frame + subtitle + 权重注入 Prompt
        ↓
Stage C：标签规范化（对齐 EW 评估）
  Qwen2.5 开放词汇抽取 + 同义词归一
        ↓
  CodaBench 提交
```

### 1.4 设计原则

- **不推翻 MERTools**：官方代码放 `third_party/MERTools`，自研增强放 `src/`
- **先规则后模型**：L4 路由先用 SIT 规则版，验证后再考虑小网络
- **评估驱动**：每阶段在 Human-OV val 集上量化 EW-F1
- **可复现**：固定 seed、保存 config + ckpt + 推理命令（符合比赛审查要求）

---

## 二、项目目录结构

```
MER2026/
├── SIT_mds/                    # SIT 参考文档（只读）
├── third_party/
│   └── MERTools/               # git clone 官方 baseline
├── config/                     # 统一配置
├── data/                       # MER2026 数据集（HF 下载）
├── models/                     # 预训练权重
├── outputs/                    # 实验输出、提交文件
├── logs/
├── scripts/                    # 一键脚本
├── src/                        # 自研增强代码
├── tests/
├── experiments/                # 实验记录
└── docs/                       # 项目文档（本文档）
```

详见 [PROJECT_STRUCTURE.md](./PROJECT_STRUCTURE.md)。

---

## 三、分阶段任务计划

### 阶段 0：环境与工程底座（~1 天）

| ID | 任务 | 交付物 |
|----|------|--------|
| 0.1 | 克隆 MERTools | `third_party/MERTools` |
| 0.2 | Conda 环境 vllm3 | 可 import torch/vllm |
| 0.3 | HF 数据申请与挂载 | `data/mer2026-dataset/` |
| 0.4 | 预训练模型下载 | `models/` |
| 0.5 | 统一配置框架 | `config/*.yaml` + `config_loader.py` |
| 0.6 | 日志与实验目录 | `logs/`、`outputs/`、`experiments/` |
| 0.7 | 基础类型定义 | `src/core/types.py` |

**验收：** 能读取 `track2_train_human.csv` 前 10 条。

---

### 阶段 1：数据与评估基础设施（~1.5 天）

| ID | 任务 | 交付物 |
|----|------|--------|
| 1.1 | 数据集索引模块 | `src/data/dataset_index.py` |
| 1.2 | Human-OV train/val 划分 | `data/splits/human_ov_val.txt` |
| 1.3 | 标签统计分析 | `docs/reports/data_stats.md` |
| 1.4 | EW 评估封装 | `src/evaluation/eval_runner.py` |
| 1.5 | CodaBench 提交格式 | `src/data/submission_formatter.py` |
| 1.6 | 错例分析框架 | `src/evaluation/error_analysis.py` |

**验收：** 对任意 npz 输出 EW-F1 + 错例统计。

---

### 阶段 2：官方 Baseline 复现（~2–3 天）

| ID | 任务 | 交付物 |
|----|------|--------|
| 2.1 | 对齐官方 config 路径 | 可运行官方脚本 |
| 2.2 | Zero-shot 抽测 | SALMONN/Chat-UniVi 100 样本 |
| 2.3 | ovlabel 两阶段跑通 | `-openset.npz` |
| 2.4 | AffectGPT 训练 Human-OV | checkpoint |
| 2.5 | AffectGPT 推理 | `results-mer2026ov/*.npz` |
| 2.6 | Baseline 评估报告 | `experiments/exp001_baseline/` |
| 2.7 | 记录 baseline 数字 | 对比表 |

**验收：** val EW-F1 ≥ 55%（完整训练后 ~59%）。

---

### 阶段 3：SIT-L4 矛盾检测与动态路由（~2 天）

| ID | 任务 | 交付物 |
|----|------|--------|
| 3.1 | 模态代理分数器 | `src/routing/modality_scorer.py` |
| 3.2 | VA 距离计算 | `src/routing/va_distance.py` |
| 3.3 | 专家规则分类 | `src/routing/expert_rules.py` |
| 3.4 | 权重查表 | `src/routing/weight_selector.py` |
| 3.5 | 批量路由 | `src/routing/run_routing.py` |
| 3.6 | 路由结果分析 | 分析报告 |
| 3.7 | 单元测试 | `tests/test_routing/` |

**验收：** 100 样本人工 spot-check 合理率 ≥ 70%。

---

### 阶段 4：Prompt 工程与标签抽取优化（~1.5 天）

| ID | 任务 | 交付物 |
|----|------|--------|
| 4.1 | Prompt 模板库 | `config/prompts/*.yaml` |
| 4.2 | 权重注入描述 Prompt | `src/prompts/description_builder.py` |
| 4.3 | EW-aware 抽标签 Prompt | `src/prompts/openset_builder.py` |
| 4.4 | openset 后处理 | `src/inference/openset_extractor.py` |
| 4.5 | Prompt 消融实验 | `experiments/exp002_prompt_ablation/` |
| 4.6 | 单元测试 | `tests/test_prompts/` |

**验收：** 同一 ckpt 下 Prompt 优化带来 val +1~3%。

---

### 阶段 5：AffectGPT 训练优化（~3–5 天）

| ID | 任务 | 交付物 |
|----|------|--------|
| 5.1 | MER-Caption+ 质量过滤 | `src/training/data_filter.py` |
| 5.2 | 混合训练策略 | 新 train yaml |
| 5.3 | Human-OV 最优训练 | best ckpt |
| 5.4 | MER-Caption+ 对比训练 | 对比 ckpt |
| 5.5 | 混合训练实验 | `experiments/exp003_mixed_train/` |
| 5.6 | 推理 epoch 选择 | epoch 报告 |

**验收：** val EW-F1 ≥ 61%，目标 62%+。

---

### 阶段 6：完整推理流水线集成（~1.5 天）

| ID | 任务 | 交付物 |
|----|------|--------|
| 6.1 | PipelineRunner | `src/inference/pipeline.py` |
| 6.2 | AffectGPT 封装 | `src/inference/affectgpt_runner.py` |
| 6.3 | 批量推理脚本 | `scripts/infer_full.sh` |
| 6.4 | 多 ckpt 集成（可选） | `src/inference/ensemble.py` |
| 6.5 | 集成测试 | `tests/test_pipeline/` |
| 6.6 | 性能 profiling | benchmark 报告 |

**验收：** 一键产出 `outputs/submissions/track2_*.csv`。

---

### 阶段 7：SIT-L1 特征增强（可选，~3 天）

| ID | 任务 | 交付物 |
|----|------|--------|
| 7.1 | OpenFace AU 解析 | `src/features/openface_au.py` |
| 7.2 | AU 弱信号放大 | 增强特征 |
| 7.3 | 音频韵律特征 | `src/features/speech_prosody.py` |
| 7.4 | 路由分数器升级 | 更准确 routing |
| 7.5 | 纯视觉旁路 Prompt | Prompt 分支 |
| 7.6 | 特征消融 | `experiments/exp004_feature_ablation/` |

**验收：** 至少一项带来 val +0.5~1.5%，否则不并入主 pipeline。

---

### 阶段 8：实验、提交与复现打包（~2 天+）

| ID | 任务 | 交付物 |
|----|------|--------|
| 8.1 | 消融实验矩阵 | `outputs/ablation/summary.csv` |
| 8.2 | 最终模型全量推理 | final npz |
| 8.3 | CodaBench 提交 | leaderboard 分数 |
| 8.4 | 复现材料打包 | zip 包 |
| 8.5 | 错例复盘 | 改进 backlog |
| 8.6 | 技术报告草稿 | MRAC workshop outline |

---

## 四、阶段依赖关系

```
阶段0 → 阶段1 → 阶段2 ─┬→ 阶段3 ─┐
                        ├→ 阶段4 ─┼→ 阶段6 → 阶段8
                        └→ 阶段5 ─┘
阶段3 → 阶段7（可选）→ 阶段6
```

**并行建议：** 阶段 2 训练挂起时并行阶段 3、4；阶段 5 长训时并行阶段 7 原型。

---

## 五、里程碑与预期指标

| 里程碑 | 完成阶段 | 目标 val EW-F1 |
|--------|----------|----------------|
| M1：Baseline 可信 | 0–2 | ≥ 59% |
| M2：Prompt+路由初效 | 3–4 | ≥ 61% |
| M3：训练优化 | 5 | ≥ 62% |
| M4：完整系统 | 6 | ≥ 63% |
| M5：冲刺提交 | 7–8 | ≥ 65%（stretch） |

---

## 六、关键资源链接

| 资源 | 链接 |
|------|------|
| 官网 | https://zeroqiaoba.github.io/MER-Challenge/ |
| Baseline 论文 | https://arxiv.org/abs/2604.19417 |
| 数据集 | https://huggingface.co/datasets/MERChallenge/MER2026 |
| Baseline 代码 | https://github.com/zeroQiaoba/MERTools |
| CodaBench | https://www.codabench.org/competitions/17196 |

---

## 七、风险与应对

| 风险 | 应对 |
|------|------|
| 数据未获批 | 先用子集开发 routing/prompt |
| GPU 不足 | 减小 batch、LoRA only、先 Human-OV |
| L4 规则不准 | 降级为 consistent 等权 |
| EW-F1 不升 | error_analysis 定位同义词/漏标 |
| 复现审查 | 固定 seed、保存完整 config + 命令日志 |

---

## 八、变更记录

| 版本 | 日期 | 变更说明 |
|------|------|----------|
| v1.0 | 2026-07-06 | 初版：目录骨架 + 八阶段任务计划 |
