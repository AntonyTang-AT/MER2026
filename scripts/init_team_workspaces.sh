#!/usr/bin/env bash
# 初始化工作区：main + tmx / yxp / zzj / cyx
# 共享 data、models、third_party 在仓库根目录，各工作区通过软链接引用
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACES=(main tmx yxp zzj cyx)

COPY_DIRS=(src config scripts tests docs experiments SIT_mds)
COPY_FILES=(requirements.txt)

other_workspaces() {
  local self="$1"
  local o=()
  for w in "${WORKSPACES[@]}"; do
    [[ "$w" != "$self" ]] && o+=("$w")
  done
  local IFS="、"
  echo "${o[*]/#/MER2026/}"
}

workspace_label() {
  if [[ "$1" == "main" ]]; then
    echo "主工作区 main（集成 / 公共基线）"
  else
    echo "队员工作区 \`$1\`"
  fi
}

echo "MER2026 workspace init: ${ROOT}"

for m in "${WORKSPACES[@]}"; do
  WS="${ROOT}/${m}"
  FORBIDDEN=$(other_workspaces "$m")
  LABEL=$(workspace_label "$m")

  mkdir -p "${WS}/logs" "${WS}/outputs/eval_logs" "${WS}/outputs/submissions"
  touch "${WS}/logs/.gitkeep" "${WS}/outputs/.gitkeep"

  echo "========== ${m} =========="
  for d in "${COPY_DIRS[@]}"; do
    if [[ -d "${ROOT}/${d}" ]]; then
      rsync -a --delete \
        --exclude '.ipynb_checkpoints' \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        "${ROOT}/${d}/" "${WS}/${d}/"
    fi
  done
  for f in "${COPY_FILES[@]}"; do
    [[ -f "${ROOT}/${f}" ]] && cp -f "${ROOT}/${f}" "${WS}/${f}"
  done

  for link in data models third_party; do
    target="${WS}/${link}"
    rm -rf "${target}"
    ln -sfn "../${link}" "${target}"
  done

  cat > "${WS}/README.md" << EOF
# MER2026 — ${LABEL}

本目录为 **${m}** 的独立开发空间，内含完整项目代码副本（与队员目录结构相同）。

## 必读

- [docs/MEMBER_RULES.md](docs/MEMBER_RULES.md) — **只能在 \`${m}/\` 内改动代码**
- [docs/AGENT.md](docs/AGENT.md) — Cursor Agent 接续开发提示
- [../docs/TEAM_WORKFLOW.md](../docs/TEAM_WORKFLOW.md) — 协作与目录说明

## 快速开始

\`\`\`bash
export PROJECT_ROOT="${ROOT}/${m}"
cd "\$PROJECT_ROOT"
export PYTHONPATH="\$PROJECT_ROOT:\${PYTHONPATH:-}"
conda activate vllm3

bash scripts/sync_mertools_config.sh
python -c "from src.core.config_loader import load_global_config; print(load_global_config())"
\`\`\`

## 共享资源（勿复制、勿删除）

| 链接 | 实际路径 |
|------|----------|
| \`data/\` | \`${ROOT}/data/\` |
| \`models/\` | \`${ROOT}/models/\` |
| \`third_party/\` | \`${ROOT}/third_party/\` |

实验输出请写入本目录下的 \`outputs/\`、\`logs/\`。
EOF

  cat > "${WS}/docs/MEMBER_RULES.md" << EOF
# 开发规则 — \`${m}\`

## 1. 工作区边界（必须遵守）

| 允许 | 禁止 |
|------|------|
| 修改 \`MER2026/${m}/\` 下所有文件 | 修改 ${FORBIDDEN} 等其他工作区目录 |
| 在本目录 \`outputs/\`、\`logs/\`、\`experiments/\` 做实验 | 未经协商删除或移动根目录 \`data/\`、\`models/\` |
| 阅读根目录 \`docs/TEAM_WORKFLOW.md\` 与共享 \`data/models\` | 直接改 \`third_party/MERTools\` 官方代码（需全队同意） |

**Cursor 工作区根目录应设为：**

\`\`\`
${ROOT}/${m}
\`\`\`

在 Agent 对话中请明确：「当前工作区 \`${m}\`，只在 \`${m}/\` 下编辑。」

## 2. 共享资源

- \`data/\`、\`models/\`、\`third_party/\` 为**全队共用**（软链接）。
- 下载脚本可在任一工作区执行，写入根目录 \`data/\`。
- checkpoint、npz、提交 csv 放在各自 \`outputs/\`，避免互相覆盖。

## 3. Git 与合并

- 只 commit 本目录 \`${m}/\` 下的变更。
- 禁止 \`git add\` 其他工作区目录下的未授权修改。

## 4. 工作区说明

| 目录 | 角色 |
|------|------|
| \`main/\` | 主集成线 / 公共基线代码 |
| \`tmx/\` \`yxp/\` \`zzj/\` \`cyx/\` | 四名队员平行开发 |

## 5. 自检

\`\`\`bash
cd ${ROOT}/${m}
export PYTHONPATH="\$(pwd):\${PYTHONPATH:-}"
pytest tests/ -q
\`\`\`
EOF

  cat > "${WS}/docs/AGENT.md" << EOF
# Agent 提示文档 — 工作区 \`${m}\`

> **用法：** 在 Cursor 中打开 \`${ROOT}/${m}\`，复制下方「一键 Prompt」作为 Agent 第一条消息。

---

## 一键 Prompt

\`\`\`
你是 MER2026 Track2（MER-FG）参赛项目的接续开发 Agent。

## 工作区身份（必须遵守）
- 当前工作区：${m}
- 根目录：${ROOT}/${m}
- **只允许**在 \`${m}/\` 内创建、修改、删除代码与配置
- **禁止**修改 ${FORBIDDEN} 等其他工作区
- 共享 \`data/\`、\`models/\`、\`third_party/\` 为软链接，勿删除共享数据

## 项目背景
- GitHub：https://github.com/AntonyTang-AT/MER2026
- 协作：${ROOT}/docs/TEAM_WORKFLOW.md
- 规则：docs/MEMBER_RULES.md
- 规划：docs/PLAN.md、docs/TASK_CHECKLIST.md

## 开发前
1. cd ${ROOT}/${m} && export PYTHONPATH=\$(pwd):\${PYTHONPATH:-}
2. 阅读 docs/MEMBER_RULES.md
3. 改动限制在 ${m}/ 下

## 关键命令
\`\`\`bash
cd ${ROOT}/${m}
bash scripts/sync_mertools_config.sh
bash scripts/eval_local.sh /path/to/openset.npz --split val
\`\`\`

## 汇报
- 改动了 \`${m}/\` 下哪些文件
- 是否影响其他工作区（应为：否）
\`\`\`
EOF

  # 写入 / 更新 team 标识（先去掉旧 block 再追加，避免 rsync 后 id 错误）
  if [[ -f "${WS}/config/global.yaml" ]]; then
    python3 - << PY
from pathlib import Path
p = Path("${WS}/config/global.yaml")
text = p.read_text(encoding="utf-8")
if "\nteam:" in text:
    text = text[: text.index("\nteam:")].rstrip() + "\n"
p.write_text(text, encoding="utf-8")
PY
    cat >> "${WS}/config/global.yaml" << EOF

team:
  member_id: "${m}"
  workspace: "${m}"
  repo_root: "${ROOT}"
EOF
  fi

  echo "  -> ${WS} OK"
done

echo ""
echo "Done. Workspaces: ${WORKSPACES[*]}"
echo "Shared: data/ models/ third_party/ (repo root)"
echo "See docs/TEAM_WORKFLOW.md"
