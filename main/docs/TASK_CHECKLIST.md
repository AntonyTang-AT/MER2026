# 任务进度清单

> 详细说明见 [PLAN.md](./PLAN.md)。完成一项将 `[ ]` 改为 `[x]`。

## 阶段 0：环境与工程底座

- [x] 0.1 克隆 MERTools（commit `89120b7`）
- [x] 0.2 Conda 环境 vllm3（2026-07-07 验证通过：torch 2.6+cu124、transformers 4.52.1、vllm 0.8.5，250 包）
- [ ] 0.3 HF 数据（CSV ✅；Human 三包已解压 1532×3；candidate zip 4/4 仍缺）
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

- [ ] 3.1 modality_scorer
- [ ] 3.2 va_distance
- [ ] 3.3 expert_rules
- [ ] 3.4 weight_selector
- [ ] 3.5 run_routing
- [ ] 3.6 路由分析
- [ ] 3.7 单元测试

## 阶段 4：Prompt 优化

- [ ] 4.1 模板库
- [ ] 4.2 description_builder
- [ ] 4.3 openset_builder
- [ ] 4.4 openset_extractor
- [ ] 4.5 消融 exp002
- [ ] 4.6 单元测试

## 阶段 5：训练优化

- [ ] 5.1 data_filter
- [ ] 5.2 混合训练策略
- [ ] 5.3 Human-OV 最优
- [ ] 5.4 MER-Caption+ 对比
- [ ] 5.5 混合实验 exp003
- [ ] 5.6 epoch 选择

## 阶段 6：流水线集成

- [ ] 6.1 pipeline.py
- [ ] 6.2 affectgpt_runner
- [ ] 6.3 infer_full.sh
- [ ] 6.4 ensemble（可选）
- [ ] 6.5 集成测试
- [ ] 6.6 profiling

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
