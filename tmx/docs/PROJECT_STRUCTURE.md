# 项目目录结构说明

> 与 [PLAN.md](./PLAN.md) 配套使用。自研代码在 `src/`，官方 baseline 在 `third_party/MERTools`。

## 目录树

```
MER2026/
│
├── SIT_mds/                          # SIT 参考文档（只读，不参与构建）
│   ├── guides/
│   └── planning/
│
├── third_party/
│   └── MERTools/                     # git clone: zeroQiaoba/MERTools
│       └── MER2026/MER2026_Track2/     # 官方 AffectGPT、evaluation 等
│
├── config/
│   ├── global.yaml                   # 路径、GPU、随机种子
│   ├── dataset.yaml                  # 数据 csv 与字段
│   ├── models.yaml                   # 预训练模型路径
│   ├── pipeline.yaml                 # 推理流水线开关
│   ├── routing/
│   │   ├── weight_table.yaml         # 矛盾类型 → 模态权重
│   │   └── contradiction_rules.yaml  # 矛盾判定阈值
│   └── prompts/
│       ├── description.yaml          # Stage B 推理 Prompt
│       └── openset_extract.yaml      # Stage C 标签抽取 Prompt
│
├── data/
│   ├── mer2026-dataset/              # HF 数据集（需申请下载）
│   └── splits/                       # train/val 划分文件
│
├── models/                           # 预训练权重（HF / 网盘）
│
├── outputs/
│   ├── checkpoints/
│   ├── results-mer2026ov/
│   ├── submissions/
│   ├── eval_logs/
│   └── ablation/
│
├── logs/
│
├── scripts/
│   ├── setup_env.sh
│   ├── clone_mertools.sh
│   ├── train_affectgpt.sh
│   ├── infer_full.sh
│   ├── eval_local.sh
│   └── submit_codabench.sh
│
├── src/
│   ├── core/                         # 配置、类型、上下文
│   ├── data/                         # 数据集索引、提交格式
│   ├── routing/                      # SIT-L4：矛盾检测与动态路由
│   ├── features/                     # SIT-L1：可选特征增强
│   ├── prompts/                      # Prompt 模板与构建
│   ├── inference/                    # 推理流水线
│   ├── evaluation/                   # EW-F1 与错例分析
│   └── training/                     # 训练 wrapper 与数据过滤
│
├── tests/
│   ├── test_routing/
│   ├── test_prompts/
│   ├── test_evaluation/
│   └── test_pipeline/
│
├── experiments/                      # 各次实验 config + 结果摘要
│
├── docs/
│   ├── PLAN.md                       # 总规划（主文档）
│   └── PROJECT_STRUCTURE.md          # 本文件
│
├── requirements.txt
└── README.md
```

## 模块职责

| 目录 | 职责 |
|------|------|
| `src/routing/` | 每个样本输出 `contradiction_type` + `fusion_weights` |
| `src/prompts/` | 将路由结果注入 AffectGPT / Qwen Prompt |
| `src/inference/` | 编排 Stage A→B→C，调用官方 `inference_hybird.py` |
| `src/evaluation/` | 封装官方 `wheel.py`，本地 EW-F1 |
| `third_party/MERTools` | 尽量不修改；路径由 `config/global.yaml` 指向 |

## 与官方代码的调用关系

```
scripts/infer_full.sh
    → src/inference/pipeline.py
        → src/routing/run_routing.py          # Stage A
        → src/inference/affectgpt_runner.py   # Stage B → third_party/.../inference_hybird.py
        → src/inference/openset_extractor.py  # Stage C → ovlabel_extraction 逻辑
    → src/evaluation/eval_runner.py           # 本地评估
    → src/data/submission_formatter.py        # CodaBench 格式
```
