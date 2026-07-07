# MER2026 Track2 — MER-FG 参赛项目（团队协作）

ACM Multimedia 2026 多模态细粒度情感识别（开放词汇）赛道。

## 工作区

| 目录 | 说明 | Agent | 规则 |
|------|------|-------|------|
| **main** | 主集成 / 公共基线 | [`main/docs/AGENT.md`](main/docs/AGENT.md) | [`main/docs/MEMBER_RULES.md`](main/docs/MEMBER_RULES.md) |
| tmx | 队员 | [`tmx/docs/AGENT.md`](tmx/docs/AGENT.md) | [`tmx/docs/MEMBER_RULES.md`](tmx/docs/MEMBER_RULES.md) |
| yxp | 队员 | [`yxp/docs/AGENT.md`](yxp/docs/AGENT.md) | [`yxp/docs/MEMBER_RULES.md`](yxp/docs/MEMBER_RULES.md) |
| zzj | 队员 | [`zzj/docs/AGENT.md`](zzj/docs/AGENT.md) | [`zzj/docs/MEMBER_RULES.md`](zzj/docs/MEMBER_RULES.md) |
| cyx | 队员 | [`cyx/docs/AGENT.md`](cyx/docs/AGENT.md) | [`cyx/docs/MEMBER_RULES.md`](cyx/docs/MEMBER_RULES.md) |

**协作说明：** [`docs/TEAM_WORKFLOW.md`](docs/TEAM_WORKFLOW.md)

## 共享资源（仓库根目录，全队一份）

| 路径 | 说明 |
|------|------|
| [`data/`](data/) | MER2026 数据集 |
| [`models/`](models/) | Qwen / CLIP / HuBERT |
| [`third_party/MERTools/`](third_party/MERTools/) | 官方 baseline |

各队员目录通过软链接引用上述路径，**请勿复制四份大文件**。

## 快速开始（以 `main` 或队员 `tmx` 为例）

```bash
cd MER2026/main    # 或 MER2026/tmx
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
conda activate vllm3

bash scripts/sync_mertools_config.sh
bash scripts/eval_local.sh /path/to/openset.npz --split val
```

## 规划文档

各队员目录内均有副本：

- [`docs/PLAN.md`](tmx/docs/PLAN.md) — 技术路线与分阶段任务
- [`docs/TASK_CHECKLIST.md`](tmx/docs/TASK_CHECKLIST.md) — 进度清单

## 初始化工作区

```bash
bash scripts/init_team_workspaces.sh   # 创建/同步 main + 四名队员目录
```

## 资源链接

- [MER2026 Challenge](https://zeroqiaoba.github.io/MER-Challenge/)
- [MERTools](https://github.com/zeroQiaoba/MERTools)
- [数据集 HuggingFace](https://huggingface.co/datasets/MERChallenge/MER2026)
- [CodaBench Track2](https://www.codabench.org/competitions/17196)
