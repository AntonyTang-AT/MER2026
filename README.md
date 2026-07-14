# MER2026 Track2 — 开放词汇情感识别

ACM Multimedia 2026 **MER-FG / Track2**：多模态开放词汇（open-vocabulary）情感识别。  
输入短视频（人脸 / 音频 / 文本），输出英文情绪词列表；官方指标为 Emotion Wheel 对齐后的 **EW-F1**。

仓库：[AntonyTang-AT/MER2026](https://github.com/AntonyTang-AT/MER2026)

---

## 当前最优结果（生产锁定）

| 项 | 值 |
|------|-----|
| 变体 | `R3_RL_triple_ser_lr_dtrb_dtrb_reason_cap8` |
| **Test EW-F1** | **65.7297%** |
| 说明文档 | [`experiments/exp015_selective_fusion/BEST_MODEL_PIPELINE.md`](experiments/exp015_selective_fusion/BEST_MODEL_PIPELINE.md) |

**一句话**：三专家（RL gap1 + SFT e14 + e15）经 **SER-lr** 选择性路由后，用 **reason 引导的 DTRB** 在分歧样本上补 recall，再按 **EW synonym** 清洗提交。

```text
RL official openset
      │
      ▼
 RRB-gap1  (reason↔openset 最多补 1 标)
      │
      ├──────────────┐
      ▼              ▼
   SER-lr  ◄── e14 / e15
      │
      ▼
 DTRB (reason_guided, cap8)
      │
      ▼
 EW sanitize → answer.csv / zip
```

### 关键超参

| 模块 | 配置 |
|------|------|
| RRB | `gap_add`, `max_add=1`, `require_gap=True`, `allow_noise_swap=True` |
| SER | `ser_lr`, `confidence_threshold=0.65`, `max_switch_rate=0.10` |
| DTRB | `reason_guided=True`, `max_labels=8` |
| 提交 | `sanitize_mode=ew` |

### 复现（本地已有三路 openset + RL reason 时）

```bash
conda activate vllm3
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

python scripts/rebuild_best_model_candidate20k.py
# 产物：
#   outputs/exp021/R3_RL_triple_ser_lr_dtrb_dtrb_reason_cap8_candidate20k.npz
#   outputs/submissions/R3_RL_triple_ser_lr_dtrb_dtrb_reason_cap8_candidate20k.zip
```

核心代码：

| 路径 | 作用 |
|------|------|
| `src/inference/rl_openset_bridge.py` | RRB-gap1 |
| `src/inference/expert_router.py` | SER-lr |
| `src/inference/dtrb_boost.py` | DTRB reason_guided |
| `src/inference/triple_union.py` | 三路融合 |
| `src/data/submission_formatter.py` | EW sanitize 提交 |
| `outputs/exp015/ser_router_model.json` | SER 路由器权重 |
| `scripts/rebuild_best_model_candidate20k.py` | 一键重建生产栈 |

---

## 项目结构（简要）

```text
MER2026/
├── src/                 # 训练 / 推理 / 路由 / 评测代码
├── scripts/             # 实验与重建脚本
├── config/              # 配置
├── experiments/         # 实验记录与最优流水线文档
├── data/                # 数据集（大文件，默认不入库）
├── models/              # Qwen / CLIP / HuBERT（大文件，默认不入库）
├── third_party/         # MERTools 等
├── outputs/             # 推理产物；生产栈相关小资产可入库
└── main|tmx|yxp|zzj|cyx # 队员工作区（软链共享大数据）
```

协作说明：[`docs/TEAM_WORKFLOW.md`](docs/TEAM_WORKFLOW.md)

---

## 快速开始

```bash
cd MER2026
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
conda activate vllm3

# 同步 MERTools 配置（如需要）
bash scripts/sync_mertools_config.sh

# 本地验证 openset npz
bash scripts/eval_local.sh /path/to/openset.npz --split val
```

队员工作区初始化：

```bash
bash scripts/init_team_workspaces.sh
```

---

## 设计原则（踩坑摘要）

1. **融合优于单模硬刷** — 成绩主要来自门控融合，而非再堆一个万能 checkpoint。  
2. **默认保守、局部激进** — SER 限切换率；DTRB 只打分歧样本；RRB 每次最多 +1。  
3. **以官方提交面为准** — EW synonym 折叠后不变的「细标优化」对 test 近似无效。  
4. **外部 LLM 重写 reason 已证伪** — DeepSeek / Gemini 全量改写分歧子集均低于生产（约 65.5–65.6%）。

详细流程图、消融与「明确不做」列表见  
[`BEST_MODEL_PIPELINE.md`](experiments/exp015_selective_fusion/BEST_MODEL_PIPELINE.md)。

---

## 资源链接

- [MER2026 Challenge](https://zeroqiaoba.github.io/MER-Challenge/)
- [MERTools](https://github.com/zeroQiaoba/MERTools)
- [数据集 HuggingFace](https://huggingface.co/datasets/MERChallenge/MER2026)
- [CodaBench Track2](https://www.codabench.org/competitions/17196)
