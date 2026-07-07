# 阶段 6：完整推理流水线集成（tmx 工作区）

> 对应 [PLAN.md](./PLAN.md) 阶段 6 | 前置：阶段 3/4 已完成，阶段 2.5 提供 ckpt

## 实现状态（2026-07-07）

| ID | 模块 | 状态 |
|----|------|------|
| 6.1 | `pipeline.py` + `stage_manager.py` | ✅ |
| 6.2 | `affectgpt_runner` 增强 | ✅ |
| 6.3 | `infer_full.sh` + `config/pipeline.yaml` | ✅ |
| 6.4 | `ensemble.py` | ✅ |
| 6.5 | `tests/test_pipeline/`（11 passed） | ✅ |
| 6.6 | `benchmark_pipeline.sh` + 报告模板 | ✅ |

## 目标

将分散脚本（`run_routing.sh`、`infer_with_prompts.sh`、`run_ovlabel.sh`、`submission_formatter`）整合为 **可配置、可恢复的一键 PipelineRunner**，最终产出 CodaBench 提交文件。

## 目标与验收

- **目标**：Stage A（路由）→ Stage B（AffectGPT）→ Stage C（ovlabel）→ 提交 CSV 全链路编排。
- **验收**：
  - `bash scripts/infer_full.sh` 一键产出 `outputs/submissions/track2_*.csv`（20k 行）
  - `pytest tests/test_pipeline/ -q` 全绿（mock/干跑，无 GPU 必需）
  - `docs/reports/pipeline_benchmark.md` 记录各阶段耗时

## 现状与缺口

| 组件 | 现状 |
|------|------|
| 配置 | `config/pipeline.yaml` 已有 stages 开关 |
| 编排核心 | `src/inference/pipeline.py` **TODO 空壳** |
| Stage B | `affectgpt_runner.py` 有两条路径，缺 pipeline 统一入口与产物解析 |
| Stage A/C | `run_routing.py`、`openset_extractor.py` 已可用 |
| 提交 | `submission_formatter.py` 已实现 20k CSV |
| 一键脚本 | `scripts/infer_full.sh` **仅占位 TODO** |
| 集成 | `ensemble.py` TODO；`tests/test_pipeline/` 不存在 |

## 目标架构

```
config/pipeline.yaml
        ↓
PipelineRunner (pipeline.py)
        ↓
┌───────────────┬────────────────────┬──────────────────┐
│ Stage A       │ Stage B            │ Stage C          │
│ run_routing   │ affectgpt_runner   │ openset_extractor│
│ routing.json  │ reason.npz         │ openset.npz      │
└───────────────┴────────────────────┴──────────────────┘
        ↓ (可选 ensemble)
submission_formatter → outputs/submissions/track2_*.csv
```

## 运行模式

| 模式 | split | Stage B 数据集 | 终端动作 |
|------|-------|----------------|----------|
| `submit` | candidate (20k) | MER2026OV | 写 submission CSV |
| `eval` | human (1532) | 本地验证 | 可选 EW-F1 |

## 任务分解

| ID | 模块 | 文件 | 验收 |
|----|------|------|------|
| 6.1 | PipelineRunner | `src/inference/pipeline.py` | CLI 编排 A→B→C→提交；`--skip-existing` |
| 6.2 | AffectGPT 封装增强 | `src/inference/affectgpt_runner.py` | `run_stage_b()` + `find_latest_*_npz()` |
| 6.3 | 一键脚本 | `scripts/infer_full.sh` | 替换 TODO，透传 `--limit/--cuda` |
| 6.4 | 多 ckpt 集成（可选） | `src/inference/ensemble.py` | `label_union` 融合多个 openset npz |
| 6.5 | 集成测试 | `tests/test_pipeline/` | dry-run、submission 20k 形状 |
| 6.6 | 性能 profiling | `scripts/benchmark_pipeline.sh` + 报告 | 各阶段 timing JSON/MD |

---

### 6.1 PipelineRunner — `src/inference/pipeline.py`

**核心 API**：

```python
@dataclass
class PipelineArtifacts:
    routing_json: Path | None
    reason_npz: Path | None
    openset_npz: Path | None
    submission_csv: Path | None
    timing: dict[str, float]

class PipelineRunner:
    def run(self, *, mode: str = "submit", limit: int | None = None) -> PipelineArtifacts: ...
```

**阶段编排**（读 `config/pipeline.yaml`）：

1. **Stage A — routing**（`stages.routing.enabled`）
   - 调用 `run_batch()` from `run_routing.py`
   - split：`candidate`（submit）或 `human`（eval）
   - 输出：`outputs/routing/{split}_routing.json`

2. **Stage B — affectgpt**（`stages.affectgpt.enabled`）
   - `use_routing_in_prompt=true` → `infer_with_prompts.py`
   - 否则 → `affectgpt_runner.run_inference`
   - 记录 `reason_npz` 路径

3. **Stage C — openset**（`stages.openset_extract.enabled`）
   - `openset_extractor.extract_openset_custom`
   - prompt/postprocess 读 `config/prompts/pipeline.yaml`

4. **Stage D — ensemble**（可选，`stages.ensemble.enabled`）

5. **Stage E — submission**（submit 模式）
   - `format_submission()` → `outputs/submissions/track2_{ts}.csv`

**辅助**：
- `StageManager`：enabled 阶段、依赖校验
- `resolve_latest_npz()`：从 MERTools output 取最新 npz
- `@timed_stage`：记录耗时

**CLI**：
```bash
python -m src.inference.pipeline \
  --mode submit|eval \
  --limit N \
  --skip-existing \
  --profile \
  --openset-npz PATH   # 跳过 B/C，直接提交
```

---

### 6.2 AffectGPT 封装增强 — `affectgpt_runner.py`

**补齐 API**（供 6.1 调用）：

```python
def run_stage_b(*, use_routing, routing_json, prompt_variant, ...) -> Path: ...
def find_latest_reason_npz(*, prompts: bool = False) -> Path | None: ...
def find_latest_openset_npz(reason_npz: Path | None = None) -> Path | None: ...
```

- 统一读 `baseline.yaml` inference 段
- `prompts=True` 时搜索 `results-*-prompts` 目录

---

### 6.3 批量推理脚本 — `scripts/infer_full.sh`

```bash
bash scripts/sync_mertools_config.sh
python -m src.inference.pipeline --mode submit "$@"
```

- `infer_baseline.sh` 保留为 exp001 官方对照
- `infer_full.sh` 为主系统默认路径（routing + ew openset）

---

### 6.4 多 ckpt 集成（可选）— `ensemble.py`

```python
def merge_openset_npz(npz_paths, *, strategy="label_union") -> dict[str, str]: ...
```

- `label_union`：多 npz openset 并集 → 去重 → 格式化
- `majority_vote`：保留 ≥2 ckpt 出现的标签（次优先）
- 默认 `stages.ensemble.enabled: false`

---

### 6.5 集成测试 — `tests/test_pipeline/`

| 文件 | 覆盖 |
|------|------|
| `test_stage_manager.py` | enabled 阶段、依赖校验 |
| `test_artifact_resolver.py` | find_latest_npz |
| `test_pipeline_dry_run.py` | 编排顺序、skip-existing |
| `test_ensemble.py` | label_union |
| `test_pipeline_submission.py` | mock npz → 20k CSV |

---

### 6.6 性能 Profiling

- pipeline `--profile` → `artifacts.timing` JSON
- `scripts/benchmark_pipeline.sh`：`--limit 100` 跑 3 次取均值
- `docs/reports/pipeline_benchmark.md` 模板

记录：`routing_sec`、`affectgpt_sec_per_sample`、`ovlabel_sec_per_sample`、`total_sec`

---

## 执行顺序

```
6.1 stage_manager + artifact 解析（CPU）
    → 6.2 affectgpt_runner 增强
        → 6.1 PipelineRunner 主逻辑
            → 6.3 infer_full.sh
            → 6.4 ensemble（并行）
                → 6.5 集成测试
                    → 6.6 profiling
```

**并行策略**：
- 训练进行中：6.1/6.2/6.4/6.5 CPU 开发
- 2.5 ckpt 就绪后：`--limit 100` 冒烟 → 20k submit

## 配置扩展（pipeline.yaml）

```yaml
stages:
  affectgpt:
    use_routing_in_prompt: true
    prompt_variant: routing
  openset_extract:
    prompt_variant: ew_aware
    postprocess: ew

artifacts:
  routing_dir: outputs/routing
  submission_dir: outputs/submissions

runtime:
  skip_existing: true
```

## 风险与对策

| 风险 | 对策 |
|------|------|
| 20k 三阶段耗时数天 | `--skip-existing`、分阶段 CLI |
| routing 20k CPU 慢 | 缓存 JSON；预计算 |
| MERTools output 路径漂移 | 集中 `resolve_latest_npz()` |
| B/C GPU 争用 | 顺序执行；`CUDA_VISIBLE_DEVICES` |

## 一键命令（目标态）

```bash
cd /root/autodl-tmp/MER2026/tmx
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

pytest tests/test_pipeline/ -q

python -m src.inference.pipeline --mode eval --limit 100 --profile

bash scripts/infer_full.sh --cuda 0

bash scripts/benchmark_pipeline.sh --limit 100
```

## 与阶段 8 衔接

- 阶段 6 产出 `track2_*.csv` → `scripts/submit_codabench.sh`
- 阶段 8.2 全量 20k 推理即 `infer_full.sh` 生产运行
