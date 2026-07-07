# MER2026 — 主工作区 main（集成 / 公共基线）

本目录为 **main** 的独立开发空间，内含完整项目代码副本（与队员目录结构相同）。

## 必读

- [docs/MEMBER_RULES.md](docs/MEMBER_RULES.md) — **只能在 `main/` 内改动代码**
- [docs/AGENT.md](docs/AGENT.md) — Cursor Agent 接续开发提示
- [../docs/TEAM_WORKFLOW.md](../docs/TEAM_WORKFLOW.md) — 协作与目录说明

## 快速开始

```bash
export PROJECT_ROOT="/root/autodl-tmp/MER2026/main"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
conda activate vllm3

bash scripts/sync_mertools_config.sh
python -c "from src.core.config_loader import load_global_config; print(load_global_config())"
```

## 共享资源（勿复制、勿删除）

| 链接 | 实际路径 |
|------|----------|
| `data/` | `/root/autodl-tmp/MER2026/data/` |
| `models/` | `/root/autodl-tmp/MER2026/models/` |
| `third_party/` | `/root/autodl-tmp/MER2026/third_party/` |

实验输出请写入本目录下的 `outputs/`、`logs/`。
