# MER2026 Human-OV 标签统计

## 数据完整性（0.3，2026-07-07）

| 项 | 状态 |
|----|------|
| CSV | human 1533 行、candidate 20001 行、subtitle、mercaptionplus 31328 行 |
| ZIP | 9/9 通过 `7z t` |
| 解压媒体 | audio / video / openface_face 各 **21531** 文件 |
| 覆盖率 | Human 1532 + Candidate 20000 = 21532 名，去重后 **21531** 名，**0 缺失** |
| 校验 | `bash scripts/verify_downloads.sh` → **OVERALL: PASS** |

> MER-Caption+ **媒体包**（~31k 视频）未下载，属阶段 5 训练扩展，不影响 Track2 提交推理。

- 总样本数: 1532
- 训练集: 1226
- 验证集: 306
- 唯一标签数: 770
- 每样本标签数: min=1, max=14, mean=5.40
- 单标签样本: 17 (1.1%)
- 多标签样本: 1515 (98.9%)

## Top-30 高频标签

| 标签 | 出现次数 |
|------|----------|
| serious | 430 |
| worried | 409 |
| angry | 293 |
| confused | 266 |
| concerned | 217 |
| dissatisfied | 195 |
| happy | 177 |
| helpless | 173 |
| joyful | 158 |
| nervous | 151 |
| anxious | 149 |
| firm | 140 |
| surprised | 126 |
| contemplative | 107 |
| sad | 106 |
| determined | 100 |
| relaxed | 98 |
| excited | 93 |
| pleased | 93 |
| smile | 91 |
| curious | 87 |
| contemplate | 86 |
| doubtful | 84 |
| friendly | 80 |
| well-thought-out | 75 |
| uncertainty | 72 |
| positive | 67 |
| disappointed | 66 |
| urgent | 63 |
| afraid | 62 |
