# Agent 提示文档 — 工作区 `tmx`

> **用法：** 在 Cursor 中打开 `/root/autodl-tmp/MER2026/tmx` 作为工作区根目录，复制下方「一键 Prompt」作为 Agent **第一条消息**。
>
> 除「工作区身份」中的路径与目录名外，**各工作区本文档内容一致**。

---

## 当前进度快照（2026-07-07）

| 类别 | 状态 |
|------|------|
| **阶段 0–1** 环境 / 数据 / 评估 | ✅ 完成 |
| **阶段 2** Human-OV 训练 | ✅ **60 epoch 已完成**（checkpoint 61 个，本地） |
| **阶段 2** 推理 + exp001 评估 | ⏳ **下一步**（2.5 → 2.6） |
| **阶段 3** L4 路由 | ✅ 代码 + 测试（Human routing 1532 样本已有） |
| **阶段 4** Prompt 工程 | ✅ 代码 + exp002 脚手架（数字待 2.5） |
| **阶段 5** 训练优化 | ✅ 过滤/混合/epoch 扫描脚本；⏳ GPU 实验 |
| **阶段 6** 流水线集成 | ✅ `infer_full.sh` + 11 集成测试 |
| **阶段 7–8** 特征 / 提交 | ⏳ 未开始 |

**里程碑目标：** M1 ≥59% → M2 ≥61% → M3 ≥62% → M4 ≥63% → M5 ≥65%（val EW-F1）

**环境：** conda `vllm3` · torch **2.8+cu128** · **5× RTX 5090** · 单脚本默认 `--cuda 0`

**数据：** Human + Candidate 媒体 ✅ · MER-Caption+ **媒体未下载**（CSV + 过滤 CSV 已有）

**详细清单：** [docs/TASK_CHECKLIST.md](./TASK_CHECKLIST.md) · [docs/PLAN.md](./PLAN.md)

---

## 架构速览

```
routing.json → AffectGPT(Stage B) → reason.npz
    → Qwen ovlabel(Stage C) → openset.npz → EW-F1 / 提交 CSV
```

| 阶段 | 模块 | 脚本 |
|------|------|------|
| 路由 | `src/routing/` | `bash scripts/run_routing.sh` |
| 训练 | `src/training/` | `bash scripts/train_affectgpt.sh human` |
| 推理 | `src/inference/` | `bash scripts/infer_affectgpt.sh` |
| 抽标签 | `openset_extractor.py` | `bash scripts/run_ovlabel.sh --all` |
| 评估 | `src/evaluation/` | `bash scripts/eval_baseline.sh <npz> --split val` |
| 一键 | `pipeline.py` | `bash scripts/infer_full.sh` |

**Human checkpoint 路径（本地）：**
`third_party/MERTools/MER2026/MER2026_Track2/output/human_outputhybird_.../checkpoint_*.pth`

---

## 推荐任务队列（GPU 串行）

1. **Epoch 扫描选 best ckpt**（~4–8h，单卡）
   `bash scripts/sweep_epochs.sh --train-run human --epochs 10-60 --skip 5`
2. **exp001 基线评估** — 扫描内含 EW-F1，或 `bash scripts/infer_baseline.sh`
3. **exp002 Prompt 消融** — `bash scripts/run_exp002.sh --variant D --split val`
4. **20k 提交推理** — `bash scripts/infer_full.sh`（~10–20h）
5. **（可选）MER-Caption+ 下载 + exp003 混合训练**

---

## 文档索引

| 文档 | 用途 |
|------|------|
| [MEMBER_RULES.md](./MEMBER_RULES.md) | 工作区边界与 Git 规则 |
| [TASK_CHECKLIST.md](./TASK_CHECKLIST.md) | 分阶段任务勾选 |
| [PLAN.md](./PLAN.md) | 总体规划 |
| [PHASE2_PLAN.md](./PHASE2_PLAN.md) | Baseline 复现 |
| [PHASE4_PLAN.md](./PHASE4_PLAN.md) | Prompt 消融 |
| [PHASE5_PLAN.md](./PHASE5_PLAN.md) | 训练优化 |
| [PHASE6_PLAN.md](./PHASE6_PLAN.md) | 流水线集成 |
| [TEAM_WORKFLOW.md](../docs/TEAM_WORKFLOW.md) | 全队协作 |

---

## Agent 协作提示

- **不要**在训练占用 GPU 时并行启动第二个 GPU 任务（5090 单任务 ~30GB）。
- **不要**直接改 `third_party/MERTools` 官方源码；用 `src/training/mertools_entry.py` 补丁。
- 实验输出写入本目录 `outputs/`、`logs/`，避免覆盖他人 checkpoint。
- 队长更新公共代码后：在仓库根执行 `bash scripts/init_team_workspaces.sh` 同步各工作区。
- 提交 PR 前在本工作区跑 `pytest tests/ -q`。


---

## 开发前必做

```bash
export PROJECT_ROOT="/root/autodl-tmp/MER2026/tmx"
cd "$PROJECT_ROOT"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
conda activate vllm3
bash scripts/sync_mertools_config.sh
pytest tests/ -q   # 期望 ~59 passed
```

---

## 一键 Prompt

```
你是 MER2026 Track2（MER-FG）参赛项目的接续开发 Agent。

## 工作区身份（必须遵守）
- 当前工作区：tmx
- 根目录：/root/autodl-tmp/MER2026/tmx
- 只允许在 `tmx/` 内创建、修改、删除代码与配置
- 禁止修改 MER2026/main、MER2026/yxp、MER2026/zzj、MER2026/cyx 等其他工作区
- 共享 data/、models/、third_party/ 为软链接（指向仓库根），勿删除共享数据
- Git 只 commit 本目录 `tmx/` 下的变更

## 项目状态（2026-07-07）
- 阶段 0–1、3–6 代码已完成；Human-OV 训练 60 epoch 已完成
- 下一步：epoch 扫描 → exp001/exp002 评估 → 20k 提交推理
- MER-Caption+ 媒体未下载；5×5090；脚本默认单卡 GPU 0

## 开发前
1. cd /root/autodl-tmp/MER2026/tmx && export PYTHONPATH=$(pwd):${PYTHONPATH:-}
2. 阅读 docs/MEMBER_RULES.md、docs/TASK_CHECKLIST.md
3. 改动限制在 tmx/ 下

## 关键命令
bash scripts/sync_mertools_config.sh
bash scripts/sweep_epochs.sh --train-run human --epochs 10-60 --skip 5
bash scripts/eval_baseline.sh path/to/openset.npz --split val
pytest tests/ -q

## 汇报要求
- 改动了 tmx/ 下哪些文件
- 是否影响其他工作区（应为：否）
- 若跑实验：EW-F1 数字、使用的 epoch/ckpt、日志路径
```
