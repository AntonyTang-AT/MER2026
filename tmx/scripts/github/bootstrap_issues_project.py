#!/usr/bin/env python3
"""Create GitHub Issues and Project board from docs/PLAN.md phases."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

OWNER = "AntonyTang-AT"
REPO = "MER2026"
PROJECT_TITLE = "MER2026 Track2 开发看板"

PHASES = [
    {
        "title": "[阶段0] 环境与工程底座",
        "labels": ["phase-0", "priority-critical"],
        "milestone": "M0-基础设施",
        "body": """## 目标
目录就绪、配置可加载、能调用官方 MERTools。

## 任务清单
- [ ] 0.1 克隆 MERTools → `third_party/MERTools`
- [ ] 0.2 Conda 环境 vllm3
- [ ] 0.3 HF 数据申请与挂载 → `data/mer2026-dataset/`
- [ ] 0.4 预训练模型下载 → `models/`
- [ ] 0.5 统一配置框架（config + config_loader）
- [ ] 0.6 日志与实验目录
- [ ] 0.7 基础类型定义（types.py）

## 验收标准
能读取 `track2_train_human.csv` 前 10 条。

## 参考
- docs/PLAN.md 阶段 0
- docs/TASK_CHECKLIST.md
""",
    },
    {
        "title": "[阶段1] 数据与评估基础设施",
        "labels": ["phase-1", "priority-critical"],
        "milestone": "M0-基础设施",
        "body": """## 目标
本地可算 EW-F1，有 train/val 划分。

## 任务清单
- [ ] 1.1 `src/data/dataset_index.py`
- [ ] 1.2 Human-OV train/val 划分 → `data/splits/human_ov_val.txt`
- [ ] 1.3 标签统计分析 → `docs/reports/data_stats.md`
- [ ] 1.4 EW 评估封装 → `src/evaluation/eval_runner.py`
- [ ] 1.5 CodaBench 提交格式 → `src/data/submission_formatter.py`
- [ ] 1.6 错例分析 → `src/evaluation/error_analysis.py`

## 验收标准
对任意 npz 预测文件输出 EW-F1 + 错例统计。

## 参考
docs/PLAN.md 阶段 1
""",
    },
    {
        "title": "[阶段2] 官方 Baseline 复现",
        "labels": ["phase-2", "priority-critical"],
        "milestone": "M1-Baseline",
        "body": """## 目标
建立可信 baseline（AffectGPT ~59% EW-F1）。

## 任务清单
- [ ] 2.1 对齐官方 config 路径
- [ ] 2.2 Zero-shot 抽测（SALMONN / Chat-UniVi）
- [ ] 2.3 ovlabel 两阶段跑通
- [ ] 2.4 AffectGPT 训练 Human-OV
- [ ] 2.5 AffectGPT 推理 → `results-mer2026ov/*.npz`
- [ ] 2.6 评估报告 → `experiments/exp001_baseline/`
- [ ] 2.7 记录 baseline 数字

## 验收标准
val EW-F1 ≥ 55%（完整训练后 ~59%）。

## 里程碑
M1：Baseline 可信 ≥ 59%
""",
    },
    {
        "title": "[阶段3] SIT-L4 矛盾检测与动态路由",
        "labels": ["phase-3", "enhancement"],
        "milestone": "M2-路由Prompt",
        "body": """## 目标
每样本输出 `contradiction_type` + `fusion_weights`。

## 任务清单
- [ ] 3.1 `src/routing/modality_scorer.py`
- [ ] 3.2 `src/routing/va_distance.py`
- [ ] 3.3 `src/routing/expert_rules.py`
- [ ] 3.4 `src/routing/weight_selector.py`
- [ ] 3.5 `src/routing/run_routing.py`
- [ ] 3.6 路由结果分析
- [ ] 3.7 单元测试 `tests/test_routing/`

## 验收标准
100 样本人工 spot-check 合理率 ≥ 70%。

## 配置
- `config/routing/weight_table.yaml`
- `config/routing/contradiction_rules.yaml`
""",
    },
    {
        "title": "[阶段4] Prompt 工程与标签抽取优化",
        "labels": ["phase-4", "enhancement"],
        "milestone": "M2-路由Prompt",
        "body": """## 目标
不改模型提升 val EW-F1 +1~3%。

## 任务清单
- [ ] 4.1 Prompt 模板库 `config/prompts/`
- [ ] 4.2 `src/prompts/description_builder.py`
- [ ] 4.3 `src/prompts/openset_builder.py`
- [ ] 4.4 `src/inference/openset_extractor.py`
- [ ] 4.5 Prompt 消融 → `experiments/exp002_prompt_ablation/`
- [ ] 4.6 单元测试 `tests/test_prompts/`

## 验收标准
同一 ckpt 下 Prompt 优化带来明显提升。

## 里程碑
M2：Prompt+路由 ≥ 61%
""",
    },
    {
        "title": "[阶段5] AffectGPT 训练优化",
        "labels": ["phase-5", "priority-high"],
        "milestone": "M3-训练优化",
        "body": """## 目标
超越官方 60% 训练策略，val ≥ 62%。

## 任务清单
- [ ] 5.1 MER-Caption+ 质量过滤 → `src/training/data_filter.py`
- [ ] 5.2 混合训练策略（Human-OV + Caption+）
- [ ] 5.3 Human-OV 最优 ckpt
- [ ] 5.4 MER-Caption+ 对比训练
- [ ] 5.5 混合实验 → `experiments/exp003_mixed_train/`
- [ ] 5.6 推理 epoch 选择

## 里程碑
M3：训练优化 ≥ 62%
""",
    },
    {
        "title": "[阶段6] 完整推理流水线集成",
        "labels": ["phase-6", "priority-high"],
        "milestone": "M4-完整系统",
        "body": """## 目标
一条命令跑完 Stage A→B→C，产出可提交文件。

## 任务清单
- [ ] 6.1 `src/inference/pipeline.py`
- [ ] 6.2 `src/inference/affectgpt_runner.py`
- [ ] 6.3 `scripts/infer_full.sh`
- [ ] 6.4 多 ckpt 集成（可选）→ `ensemble.py`
- [ ] 6.5 集成测试 `tests/test_pipeline/`
- [ ] 6.6 性能 profiling

## 验收标准
`scripts/infer_full.sh` 产出 `outputs/submissions/track2_*.csv`

## 里程碑
M4：完整系统 ≥ 63%
""",
    },
    {
        "title": "[阶段7] SIT-L1 特征增强（可选）",
        "labels": ["phase-7", "optional"],
        "milestone": "M5-冲刺",
        "body": """## 目标
强化 face/audio 细粒度信号（可选进阶）。

## 任务清单
- [ ] 7.1 OpenFace AU 解析
- [ ] 7.2 AU 弱信号放大
- [ ] 7.3 音频韵律特征
- [ ] 7.4 路由分数器升级
- [ ] 7.5 纯视觉旁路 Prompt
- [ ] 7.6 特征消融 → `experiments/exp004_feature_ablation/`

## 验收标准
至少一项带来 val +0.5~1.5%，否则不并入主 pipeline。
""",
    },
    {
        "title": "[阶段8] 实验、提交与复现打包",
        "labels": ["phase-8", "priority-critical"],
        "milestone": "M5-冲刺",
        "body": """## 目标
CodaBench 正式提交 + 复现材料（截止 2026-07-13 AoE）。

## 任务清单
- [ ] 8.1 消融实验矩阵 → `outputs/ablation/summary.csv`
- [ ] 8.2 全量 20k 推理
- [ ] 8.3 CodaBench 提交
- [ ] 8.4 复现材料打包（权重、config、seed、脚本）
- [ ] 8.5 错例复盘
- [ ] 8.6 技术报告草稿（MRAC workshop，截止 07-22）

## 里程碑
M5：冲刺 ≥ 65%（stretch goal）

## 链接
- CodaBench: https://www.codabench.org/competitions/17196
""",
    },
]

MILESTONES = [
    ("M0-基础设施", "阶段 0–1：环境与评估"),
    ("M1-Baseline", "阶段 2：官方 baseline 复现，EW-F1 ≥ 59%"),
    ("M2-路由Prompt", "阶段 3–4：SIT 路由 + Prompt，≥ 61%"),
    ("M3-训练优化", "阶段 5：训练策略，≥ 62%"),
    ("M4-完整系统", "阶段 6：端到端 pipeline，≥ 63%"),
    ("M5-冲刺", "阶段 7–8：提交与冲刺，≥ 65%"),
]

LABELS = {
    "phase-0": "9B59B6",
    "phase-1": "1D76DB",
    "phase-2": "5319E7",
    "phase-3": "B60205",
    "phase-4": "D93F0B",
    "phase-5": "FBCA04",
    "phase-6": "0E8A16",
    "phase-7": "006B75",
    "phase-8": "6F42C1",
    "priority-critical": "B60205",
    "priority-high": "E99695",
    "enhancement": "C5DEF5",
    "optional": "D4C5F9",
}


def api(method: str, url: str, data: dict | None = None) -> dict:
    token = os.environ["GH_TOKEN"]
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    body = None
    if data is not None:
        body = json.dumps(data).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read().decode()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        err = e.read().decode()
        raise RuntimeError(f"{method} {url} -> {e.code}: {err}") from e


def graphql(query: str, variables: dict | None = None) -> dict:
    token = os.environ["GH_TOKEN"]
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    req = urllib.request.Request(
        "https://api.github.com/graphql",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        out = json.loads(resp.read().decode())
    if out.get("errors"):
        raise RuntimeError(json.dumps(out["errors"], indent=2))
    return out["data"]


def ensure_labels() -> None:
    base = f"https://api.github.com/repos/{OWNER}/{REPO}/labels"
    existing = {x["name"] for x in api("GET", f"{base}?per_page=100")}
    for name, color in LABELS.items():
        if name in existing:
            continue
        api("POST", base, {"name": name, "color": color, "description": name})
        print(f"label: {name}")


def ensure_milestones() -> dict[str, int]:
    base = f"https://api.github.com/repos/{OWNER}/{REPO}/milestones"
    existing = {m["title"]: m["number"] for m in api("GET", f"{base}?state=all&per_page=100")}
    ids = {}
    for title, desc in MILESTONES:
        if title in existing:
            ids[title] = existing[title]
            continue
        m = api("POST", base, {"title": title, "description": desc, "state": "open"})
        ids[title] = m["number"]
        print(f"milestone: {title} #{m['number']}")
    return ids


def create_issues(milestone_ids: dict[str, int]) -> list[dict]:
    base = f"https://api.github.com/repos/{OWNER}/{REPO}/issues"
    created = []
    for phase in PHASES:
        issue = api(
            "POST",
            base,
            {
                "title": phase["title"],
                "body": phase["body"],
                "labels": phase["labels"],
                "milestone": milestone_ids[phase["milestone"]],
            },
        )
        created.append(issue)
        print(f"issue #{issue['number']}: {issue['title']}")
    return created


def get_project_v2(number: int) -> dict:
    data = graphql(
        """
        query($login: String!, $number: Int!) {
          user(login: $login) {
            projectV2(number: $number) {
              id
              url
              fields(first: 30) {
                nodes {
                  ... on ProjectV2SingleSelectField {
                    id
                    name
                    options { id name }
                  }
                }
              }
            }
          }
        }
        """,
        {"login": OWNER, "number": number},
    )
    return data["user"]["projectV2"]


def create_project_and_add_issues(issues: list[dict], project_number: int | None = None) -> str:
    if project_number is not None:
        proj = get_project_v2(project_number)
        project_id = proj["id"]
        project_url = proj["url"]
        print(f"project (existing): {project_url}")
    else:
        owner = graphql(
            """
            query($login: String!) {
              user(login: $login) { id }
            }
            """,
            {"login": OWNER},
        )
        owner_id = owner["user"]["id"]

        repo = graphql(
            """
            query($owner: String!, $name: String!) {
              repository(owner: $owner, name: $name) { id }
            }
            """,
            {"owner": OWNER, "name": REPO},
        )
        repo_id = repo["repository"]["id"]

        project = graphql(
            """
            mutation($ownerId: ID!, $repoId: ID!, $title: String!) {
              createProjectV2(input: {ownerId: $ownerId, repositoryId: $repoId, title: $title}) {
                projectV2 { id url number }
              }
            }
            """,
            {"ownerId": owner_id, "repoId": repo_id, "title": PROJECT_TITLE},
        )
        project_id = project["createProjectV2"]["projectV2"]["id"]
        project_url = project["createProjectV2"]["projectV2"]["url"]
        proj = get_project_v2(project["createProjectV2"]["projectV2"]["number"])
        print(f"project: {project_url}")

    status_field = None
    todo_option = None
    for node in proj["fields"]["nodes"]:
        if node and node.get("name") == "Status":
            status_field = node
            todo_option = next(
                (o for o in node["options"] if o["name"] in ("Todo", "未开始", "To do")),
                node["options"][0] if node["options"] else None,
            )
            break

    for issue in issues:
        item = graphql(
            """
            mutation($projectId: ID!, $contentId: ID!) {
              addProjectV2ItemById(input: {projectId: $projectId, contentId: $contentId}) {
                item { id }
              }
            }
            """,
            {"projectId": project_id, "contentId": issue["node_id"]},
        )
        item_id = item["addProjectV2ItemById"]["item"]["id"]
        if status_field and todo_option:
            graphql(
                """
                mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $optionId: String!) {
                  updateProjectV2ItemFieldValue(input: {
                    projectId: $projectId,
                    itemId: $itemId,
                    fieldId: $fieldId,
                    value: {singleSelectOptionId: $optionId}
                  }) { projectV2Item { id } }
                }
                """,
                {
                    "projectId": project_id,
                    "itemId": item_id,
                    "fieldId": status_field["id"],
                    "optionId": todo_option["id"],
                },
            )
        print(f"  added issue #{issue['number']} to project")

    return project_url


def fetch_open_phase_issues() -> list[dict]:
    issues = api(
        "GET",
        f"https://api.github.com/repos/{OWNER}/{REPO}/issues?state=open&per_page=100",
    )
    return [i for i in issues if "pull_request" not in i]


def main() -> None:
    if not os.environ.get("GH_TOKEN"):
        sys.exit("GH_TOKEN required")
    import sys as _sys

    if "--project-only" in _sys.argv:
        issues = fetch_open_phase_issues()
        project_num = 2
        for arg in _sys.argv[1:]:
            if arg.startswith("--project-number="):
                project_num = int(arg.split("=", 1)[1])
        project_url = create_project_and_add_issues(issues, project_number=project_num)
        print("\n=== Done (project only) ===")
        print(f"Project: {project_url}")
        print(f"Issues linked: {len(issues)}")
        return

    ensure_labels()
    milestone_ids = ensure_milestones()
    issues = create_issues(milestone_ids)
    project_url = create_project_and_add_issues(issues)
    print("\n=== Done ===")
    print(f"Repo: https://github.com/{OWNER}/{REPO}")
    print(f"Project: {project_url}")
    print(f"Issues: {len(issues)} created")


if __name__ == "__main__":
    main()
