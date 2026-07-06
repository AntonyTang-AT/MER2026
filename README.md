# MER2026 Track2 — MER-FG 参赛项目

ACM Multimedia 2026 多模态细粒度情感识别（开放词汇）赛道。

## 文档

- **[docs/PLAN.md](docs/PLAN.md)** — 技术路线、分阶段任务、里程碑（主规划文档）
- **[docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)** — 目录与模块说明
- **[docs/TASK_CHECKLIST.md](docs/TASK_CHECKLIST.md)** — 任务进度勾选清单

## 快速开始（待阶段 0 完成后）

```bash
# 1. 克隆官方 baseline
bash scripts/clone_mertools.sh

# 2. 创建环境（见 third_party/MERTools/MER2026/environment_vllm3.yml）
bash scripts/setup_env.sh

# 3. 配置路径：编辑 config/global.yaml

# 4. 本地评估
bash scripts/eval_local.sh
```

## 资源

- [MER2026 Challenge](https://zeroqiaoba.github.io/MER-Challenge/)
- [MERTools](https://github.com/zeroQiaoba/MERTools)
- [数据集 HuggingFace](https://huggingface.co/datasets/MERChallenge/MER2026)
- [CodaBench Track2](https://www.codabench.org/competitions/17196)

## 目录概览

| 路径 | 说明 |
|------|------|
| `src/` | 自研增强（路由、Prompt、流水线） |
| `third_party/MERTools/` | 官方 baseline（git clone） |
| `config/` | 统一 YAML 配置 |
| `SIT_mds/` | 前序 SIT 项目参考文档 |
