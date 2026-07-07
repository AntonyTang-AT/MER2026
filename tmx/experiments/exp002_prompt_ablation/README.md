# exp002: Prompt 消融实验

- **阶段**: 4
- **状态**: infrastructure ready — GPU 评估待阶段 2.5 训练完成
- **目标**: 同一 ckpt 下 Prompt 优化 val EW-F1 +1~3%

## 消融矩阵

| ID | 描述 Prompt | Openset Prompt | 后处理 | 用途 |
|----|-------------|----------------|--------|------|
| A | official | official | official | exp001 对照 |
| B | routing | official | official | 隔离 Stage B |
| C | official | ew_aware | ew | 隔离 Stage C |
| D | routing | ew_aware | ew | 完整方案 |

## 命令

```bash
cd /root/autodl-tmp/MER2026/tmx
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# 单 variant（需已训练 ckpt）
bash scripts/run_exp002.sh --variant B --split val
bash scripts/run_exp002.sh --variant D --split val --cuda 0

# CPU 单测
pytest tests/test_prompts/ -q
```

## 结果

| Variant | EW-F1 | 备注 |
|---------|-------|------|
| A | — | 待 exp001 baseline |
| B | — | routing description |
| C | — | ew openset only |
| D | — | full prompt stack |

详见 [RESULTS.md](./RESULTS.md)。
