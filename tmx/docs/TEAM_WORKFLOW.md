# 四人协作 — 目录与使用说明

MER2026 仓库根目录为 **`MER2026/`**，内含 **main 主工作区**、四名队员的**平行工作区**与**共享资源**。

## 目录结构

```
MER2026/                          # 仓库根（Git 根）
├── README.md
├── docs/TEAM_WORKFLOW.md
├── data/                         # 【共享】
├── models/                       # 【共享】
├── third_party/                  # 【共享】
├── logs/
│
├── main/                         # 【主工作区】完整代码副本（与队员目录同级）
├── tmx/
├── yxp/
├── zzj/
└── cyx/
```

每个工作区目录（以 `main/` 为例，队员目录结构相同）：

```
main/   # 或 tmx/ yxp/ zzj/ cyx/
├── README.md
├── src/ config/ scripts/ tests/ docs/ experiments/ SIT_mds/
├── logs/ outputs/
├── data -> ../data
├── models -> ../models
└── third_party -> ../third_party
```

**不会**把 66GB 数据、26GB 模型复制四份；仅复制代码，大资源通过软链接共享。

## 如何使用不同目录

### 1. 在 Cursor 中打开工作区

| 工作区 | 应打开的根目录 |
|--------|----------------|
| **main** | `MER2026/main/` |
| tmx | `MER2026/tmx/` |
| yxp | `MER2026/yxp/` |
| zzj | `MER2026/zzj/` |
| cyx | `MER2026/cyx/` |

每人只打开**自己的**子目录作为 workspace root（main 用于集成/公共基线）。

### 2. 环境变量

在各自目录下：

```bash
export PROJECT_ROOT=/path/to/MER2026/tmx   # 换成你的队员 id
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
conda activate vllm3
```

### 3. 运行脚本

所有 `scripts/*.sh` 在队员目录内执行即可，例如：

```bash
cd MER2026/tmx
bash scripts/sync_mertools_config.sh
bash scripts/train_affectgpt.sh human
```

`data/`、`models/` 会通过软链接读写**仓库根目录**下的同一份数据。

### 4. Agent 接续开发

各工作区内有独立 Agent 文档：

| 工作区 | Agent 文档 |
|--------|------------|
| main | `main/docs/AGENT.md` |
| tmx | `tmx/docs/AGENT.md` |
| yxp | `yxp/docs/AGENT.md` |
| zzj | `zzj/docs/AGENT.md` |
| cyx | `cyx/docs/AGENT.md` |

打开对应 workspace 后，复制该文件中的「一键 Prompt」给 Cursor Agent。

## 改动边界（基本要求）

1. **只能改自己目录**：变更限定在 `MER2026/{main|tmx|yxp|zzj|cyx}/` 内。
2. **禁止改他人目录**：不得修改其他工作区文件夹。
3. **共享资源只读**：默认不直接改 `data/`、`models/`、`third_party/`；数据下载用脚本追加，勿删他人依赖文件。
4. **输出隔离**：checkpoint、npz、提交 csv 写入各自 `outputs/`，命名带队员 id 或实验 id。
5. **Git 提交**：只 `git add` 自己队员目录下的变更（共享根目录 `data/models` 已在 `.gitignore`）。

细则见各目录 `docs/MEMBER_RULES.md`。

## 初始化与同步

首次创建或队长更新模板后，在仓库根执行：

```bash
bash scripts/init_team_workspaces.sh
```

该脚本会：复制代码至 **main + 四名队员** 目录，并重建软链接。

**注意：** `--delete` 会同步删除队员目录中已被根目录移除的文件；若队员本地有未合并的独有文件，请先备份。

## 推荐分工（参考）

| 队员 | 模块方向 |
|------|----------|
| tmx | 数据索引、评估、baseline、环境脚本 |
| yxp | SIT-L4 路由、矛盾检测 |
| zzj | 训练策略、AffectGPT 训练封装 |
| cyx | 推理流水线、CodaBench 提交 |

以 `docs/TASK_CHECKLIST.md`（各队员目录内各有一份副本）为准。

## 常见问题

**Q: 为什么根目录还有 `src/`？**  
A: 初始化后根目录代码可作为模板；日常开发请只用队员子目录。可选：仅保留 `scripts/init_team_workspaces.sh` 在根目录。

**Q: 两人同时训练会冲突吗？**  
A: 共用 GPU 时需协调；checkpoint 请写到各自 `outputs/checkpoints/`，不要覆盖 `third_party/.../output/` 下他人实验。

**Q: 如何同步队长更新的公共代码？**  
A: 队长在根目录或 `tmx/` 改完后运行 `bash scripts/init_team_workspaces.sh`，队员再 merge 自己目录内的差异。
