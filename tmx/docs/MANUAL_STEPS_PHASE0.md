# Issue #1 阶段 0 — 需您手动完成的操作

> 自动任务完成后，请按下列步骤操作。完成后在 GitHub Issue #1 中勾选对应项。

## 1. HuggingFace 数据集申请（必须，任务 0.3）

1. 打开：https://huggingface.co/datasets/MERChallenge/MER2026  
2. 登录 HuggingFace，填写 **Dataset Access Form**（团队信息格式见页面说明）  
3. 等待组织者 **人工审核通过**（通常 1–2 个工作日）  
4. 审核通过后，在本机执行：

```bash
source /etc/network_turbo   # AutoDL 学术加速
pip install -U "huggingface_hub[cli]"
huggingface-cli login        # 粘贴您的 HF Token

# 下载到项目目录（全量约 729GB，请确认磁盘 ≥ 750GB 或按需下载）
cd /root/autodl-tmp/MER2026
huggingface-cli download MERChallenge/MER2026 \
  --local-dir data/mer2026-dataset \
  --local-dir-use-symlinks False
```

**磁盘提醒：** 当前数据盘约 650GB 可用，全量数据可能不够，请在 AutoDL 控制台 **扩容数据盘** 或联系平台。

---

## 2. HuggingFace Token（模型下载，任务 0.4）

1. https://huggingface.co/settings/tokens 创建 **Read** 权限 Token  
2. 登录并后台下载模型：

```bash
source /etc/network_turbo
export HF_TOKEN=hf_你的token
huggingface-cli login --token "$HF_TOKEN"

# 后台下载（约 15–20GB+）
cd /root/autodl-tmp/MER2026
nohup bash scripts/download_models.sh > logs/download_models.nohup.log 2>&1 &
tail -f logs/download_models.log
```

---

## 3. Zero-shot 额外权重（可选，百度网盘）

官方 README 提到部分 zero-shot 模型在百度网盘：  
https://pan.baidu.com/s/1KHL1oGCtvqr8IMNWDWxH3Q?pwd=djjw  

下载后解压到：  
`third_party/MERTools/MER2026/MER2026_Track2/models/`

---

## 4. GPU 实例（训练/推理前必须）

**当前开发机未检测到 GPU**，无法运行 AffectGPT 训练或 CUDA 推理。

请在 AutoDL 租用 **带 NVIDIA GPU** 的实例（建议 ≥ 24GB 显存；满配训练建议 80GB×1 或更多），将本项目目录同步过去后继续 **阶段 2**。

---

## 5. 完成后自检

```bash
cd /root/autodl-tmp/MER2026
conda activate vllm3
python -c "from src.core.config_loader import load_global_config; print(load_global_config())"
head -3 data/mer2026-dataset/track2_train_human.csv
ls models/Qwen2.5-7B-Instruct/config.json
```

全部通过后，在 Issue #1 勾选 0.1–0.4，并开始 Issue #2 阶段 1。
