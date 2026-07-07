# 阶段 4：Prompt 工程与标签抽取优化（tmx 工作区）

> 对应 [PLAN.md](./PLAN.md) 阶段 4 | 前置：[PHASE3_PLAN.md](./PHASE3_PLAN.md) 已完成

## 实现状态（2026-07-07）

| ID | 模块 | 状态 |
|----|------|------|
| 4.1 | `src/prompts/templates.py` + `config/prompts/pipeline.yaml` | ✅ |
| 4.2 | `description_builder.py` + `infer_with_prompts.py` + `scripts/infer_with_prompts.sh` | ✅ |
| 4.3 | `openset_builder.py` | ✅ |
| 4.4 | `openset_postprocess.py` + 扩展 `openset_extractor.py` | ✅ |
| 4.5 | `experiments/exp002_prompt_ablation/` + `scripts/run_exp002.sh` | ✅ 基础设施（GPU 数字待 2.5） |
| 4.6 | `tests/test_prompts/`（20 passed） | ✅ |

## 目标与验收

- **目标**：不改 ckpt，通过 Stage B 权重注入 Prompt + Stage C EW-aware 抽标签 + 后处理，提升 val EW-F1 +1~3%。
- **验收**：
  - `pytest tests/test_prompts/ -q` 全绿
  - exp002 产出 A/B/C/D 对比（训练完成后）
  - 最优 variant val EW-F1 相对 official +≥1%

## 数据流

```
routing.json → description_builder → infer_with_prompts → reason.npz
reason.npz → openset_builder → Qwen → openset_postprocess → openset.npz → ew_metric
```

## 任务分解

### 4.1 模板库 — `src/prompts/templates.py`

- `load_description_config` / `load_openset_config` / `load_pipeline_config`
- `format_template`、`build_few_shot_block`
- 配置：`config/prompts/pipeline.yaml`（synonym_map、ew_level1_hints、default_variant）

### 4.2 权重注入描述 Prompt — `src/prompts/description_builder.py`

- `build_description_prompt(subtitle, routing, variant=official|default|routing)`
- `load_routing_map(path)` 读取阶段 3 JSON
- 推理：`src/inference/infer_with_prompts.py`（per-sample user_message，输出至 `results-*-prompts/`）

### 4.3 EW-aware 抽标签 — `src/prompts/openset_builder.py`

- `build_openset_prompt` / `build_openset_prompt_batch`
- variant：`official`（复刻 qwen）| `ew_aware`（YAML few-shot + EW hints）

### 4.4 openset 后处理 — `src/inference/openset_postprocess.py`

- `strip_qwen_prefix` → `parse_openset_string` → 同义词/去重 → `format_openset_list`
- `openset_extractor.py` 新增 `--prompt-variant`、`--postprocess ew`

### 4.5 消融 exp002

| ID | 描述 | Openset | 后处理 |
|----|------|---------|--------|
| A | official | official | official |
| B | routing | official | official |
| C | official | ew_aware | ew |
| D | routing | ew_aware | ew |

### 4.6 单元测试

`tests/test_prompts/` — 5 个测试文件，20 cases。

## 一键命令

```bash
cd /root/autodl-tmp/MER2026/tmx
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

pytest tests/test_prompts/ -q

bash scripts/infer_with_prompts.sh \
  --routing-json outputs/routing/human_routing.json \
  --prompt-variant routing

bash scripts/run_ovlabel.sh --reason-npz path/to/reason.npz \
  --prompt-variant ew_aware --postprocess ew

bash scripts/run_exp002.sh --variant D --split val
```

## 风险与对策

| 风险 | 对策 |
|------|------|
| 官方 inference 无 per-sample message | tmx `infer_with_prompts.py` 独立循环 |
| routing 仅 100 样本 | 全量前用 `run_routing.sh` 无 `--limit` 补跑 |
| GPU 与训练争用 | exp002 安排在训练结束后 |
