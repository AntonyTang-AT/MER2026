# Human-OV 路由分析报告（100 样本）

- 样本数: 100
- 平均 routing_confidence: 0.806

## 矛盾类型分布

| 类型 | 数量 | 占比 |
|------|------|------|
| intensity_mismatch | 89 | 89.0% |
| sarcasm | 9 | 9.0% |
| consistent | 1 | 1.0% |
| masking | 1 | 1.0% |

## 说明

- 当前为 **规则版 VA 代理**（字幕 lexicon + librosa + OpenFace 帧），非 L2 神经网络 VA。
- `intensity_mismatch` 占比较高属预期：代理分数噪声会拉大模态 valence 差距。
- 后续可结合 L2 VA 或人工 spot-check 调 `config/routing/contradiction_rules.yaml` 阈值。

原始结果: `outputs/routing/human_routing.json`
