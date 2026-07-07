# 任务进度清单

> 详细说明见 [PLAN.md](./PLAN.md)。完成一项将 `[ ]` 改为 `[x]`。

## 阶段 0：环境与工程底座

- [x] 0.1 克隆 MERTools（commit `89120b7`）
- [x] 0.2 Conda 环境 vllm3（2026-07-07：torch 2.8+cu128、vllm 0.10.2、5×5090）
- [x] 0.3 HF 数据（9/9 zip ✅；Human+candidate 21531×3 媒体齐全；verify PASS）
- [x] 0.4 预训练模型（CLIP ✅ HuBERT ✅ Qwen 4/4 ✅）
- [x] 0.5 统一配置框架（config + config_loader）
- [x] 0.6 日志与实验目录
- [x] 0.7 基础类型定义（types.py）

## 阶段 1：数据与评估

- [x] 1.1 dataset_index.py
- [x] 1.2 Human-OV 划分
- [x] 1.3 标签统计
- [x] 1.4 eval_runner / ew_metric
- [x] 1.5 submission_formatter
- [x] 1.6 error_analysis

## 阶段 2：Baseline 复现

- [x] 2.1 对齐官方 config（sync_mertools_config + models symlink）
- [x] 2.2 Zero-shot 抽测脚本（`scripts/zeroshot_smoke.sh`，需额外 MLLM 权重）
- [x] 2.3 ovlabel 流程（`src/inference/openset_extractor.py`）
- [ ] 2.4 AffectGPT 训练（代码就绪；Human 数据已解压；RTX 5090 需 PyTorch sm_120 支持）
- [ ] 2.5 AffectGPT 推理
- [ ] 2.6 评估报告 exp001
- [ ] 2.7 记录 baseline 数字

## 阶段 3：L4 路由

- [x] 3.1 modality_scorer（规则版 VA 代理）
- [x] 3.2 va_distance
- [x] 3.3 expert_rules
- [x] 3.4 weight_selector
- [x] 3.5 run_routing + `scripts/run_routing.sh`
- [x] 3.6 路由分析（100 样本 → `docs/reports/routing_analysis.md`）
- [x] 3.7 单元测试（10 passed）

## 阶段 4：Prompt 优化

- [x] 4.1 模板库（templates.py + pipeline.yaml）
- [x] 4.2 description_builder + infer_with_prompts
- [x] 4.3 openset_builder
- [x] 4.4 openset_postprocess + openset_extractor 扩展
- [x] 4.5 消融 exp002 基础设施（GPU 数字待 2.5）
- [x] 4.6 单元测试（20 passed）

## 阶段 5：训练优化

- [x] 5.1 data_filter
- [x] 5.2 混合训练策略
- [ ] 5.3 Human-OV 最优
- [ ] 5.4 MER-Caption+ 对比
- [x] 5.5 混合实验 exp003（脚手架）
- [x] 5.6 epoch 选择（脚本）

## 阶段 6：流水线集成

- [x] 6.1 pipeline.py + stage_manager.py
- [x] 6.2 affectgpt_runner（run_stage_b + find_latest_npz）
- [x] 6.3 infer_full.sh + config/pipeline.yaml 扩展
- [x] 6.4 ensemble.py（label_union / majority_vote）
- [x] 6.5 集成测试（11 passed）
- [x] 6.6 benchmark_pipeline.sh + pipeline_benchmark.md

## 阶段 7：特征增强（可选）

- [ ] 7.1–7.6 见 PLAN.md

## 阶段 8：提交与复现

- [ ] 8.1 消融汇总
- [ ] 8.2 全量 20k 推理
- [ ] 8.3 CodaBench 提交
- [ ] 8.4 复现材料
- [ ] 8.5 错例复盘
- [ ] 8.6 技术报告

## 里程碑

| 里程碑 | 目标 EW-F1 | 达成 |
|--------|------------|------|
| M1 Baseline | ≥ 59% | |
| M2 Prompt+路由 | ≥ 61% | |
| M3 训练优化 | ≥ 62% | |
| M4 完整系统 | ≥ 63% | |
| M5 冲刺 | ≥ 65% | |
