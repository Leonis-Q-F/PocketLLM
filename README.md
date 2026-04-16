# PocketLLM

PocketLLM 是一个面向单卡实验的轻量级大语言模型训练仓库，目标是在尽量少的工程噪音下，把模型主干、数据适配层和训练入口组织成一条可读、可维护、可扩展的链路。

当前仓库的第一优先级是提供一条可理解、可裁剪的单卡训练链路。模型主干、LoRA 工具和多种数据集适配器已经在仓库中，正式训练入口覆盖预训练、全量 SFT、LoRA 和 GRPO。

## 当前状态

- 当前状态：活跃原型
- 已提供：模型定义、LoRA 工具、数据集适配器、单卡预训练入口、全量 SFT 入口、LoRA 入口、GRPO 入口
- 尚未提供：官方 `DPO` 训练入口、评测脚本、推理命令行工具、持续集成流程
- 适合场景：本地实验、代码阅读、最小可复现训练链路
- 不适合场景：生产训练平台、多人协作的大规模训练工程、公开基准提交

## 主要特性

- 仅解码器因果语言模型
- GQA 注意力、RoPE、RMSNorm、SwiGLU
- 可选 MoE 前馈层与路由辅助损失
- 本地 tokenizer 资源，位于 `model/`
- `Pretrain / SFT / DPO / RLAIF` 数据适配器
- LoRA 注入、保存、加载与合并工具
- 预训练、全量 SFT、LoRA、GRPO 阶段的检查点保存与续训能力
- 面向单卡的训练流程

## 仓库结构

```text
PocketLLM
├── README.md
├── CLAUDE.md
├── requirements.txt
├── dataset
│   ├── dataloader.py
│   └── data
│       ├── README.md
│       ├── pretrain_t2t_mini.jsonl
│       ├── pretrain_t2t.jsonl
│       ├── sft_t2t_mini.jsonl
│       ├── sft_t2t.jsonl
│       ├── dpo.jsonl
│       ├── rlaif.jsonl
│       └── ...
├── model
│   ├── model_pocketllm.py
│   ├── model_lora.py
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── __init__.py
└── trainer
    ├── train_grpo.py
    ├── train_full_sft.py
    ├── train_lora.py
    ├── train_pre.py
    ├── rollout_engine.py
    └── trainer_utils.py
```

## 环境要求

- Python 3.10 及以上
- 推荐使用支持 CUDA 的显卡，当前主要目标环境为单卡 `RTX 4070 Ti`
- PyTorch 2.2 及以上
- Windows 和 Linux 均可，但当前路径示例优先按照本仓库现状编写

## 安装方式

```bash
python -m venv .venv
```

```bash
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

```bash
# Linux / macOS
source .venv/bin/activate
pip install -r requirements.txt
```

## 数据目录

训练数据默认放在 `dataset/data/` 下，当前仓库已经包含本地数据文件。

- `pretrain_t2t_mini.jsonl`：预训练冒烟测试首选
- `pretrain_t2t.jsonl`：更大的预训练语料
- `sft_t2t_mini.jsonl` / `sft_t2t.jsonl`：供未来监督微调阶段使用
- `dpo.jsonl`：偏好学习数据
- `rlaif.jsonl`：强化学习阶段的 prompt 数据
- `agent_rl*.jsonl`：预留给后续 agentic RL 实验

更详细的数据字段说明见 [dataset/data/README.md](dataset/data/README.md)。

## 数据格式

### 预训练

```jsonl
{"text": "这是一个预训练样本。"}
```

### 监督微调

```jsonl
{
  "conversations": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好，我是 PocketLLM。"}
  ]
}
```

### 偏好学习

```jsonl
{
  "chosen": [
    {"role": "user", "content": "Q"},
    {"role": "assistant", "content": "good answer"}
  ],
  "rejected": [
    {"role": "user", "content": "Q"},
    {"role": "assistant", "content": "bad answer"}
  ]
}
```

### 强化学习

`RLAIFDataset` 当前按多轮对话提示词读取样本，通常要求最后一条 `assistant` 留空或可被视为采样位置。

## 快速开始

当前训练脚本对若干路径使用了相对当前工作目录的默认值，因此推荐从 `trainer/` 目录启动。

```bash
cd trainer
python train_pre.py \
  --data_path ../dataset/data/pretrain_t2t_mini.jsonl \
  --save_dir ../out \
  --save_weight pretrain_smoke \
  --hidden_size 512 \
  --num_hidden_layers 4 \
  --max_seq_len 256 \
  --batch_size 4 \
  --accumulation_steps 4
```

如果要继续训练同一组权重，可以在同一工作目录下启用恢复模式：

```bash
cd trainer
python train_pre.py \
  --data_path ../dataset/data/pretrain_t2t_mini.jsonl \
  --save_dir ../out \
  --save_weight pretrain_smoke \
  --from_resume 1
```

## 训练说明

- 预训练入口是 [trainer/train_pre.py](trainer/train_pre.py)。
- 全量 SFT 入口是 [trainer/train_full_sft.py](trainer/train_full_sft.py)。
- LoRA 微调入口是 [trainer/train_lora.py](trainer/train_lora.py)。
- GRPO 强化学习入口是 [trainer/train_grpo.py](trainer/train_grpo.py)。
- GRPO rollout 推理封装在 [trainer/rollout_engine.py](trainer/rollout_engine.py)，默认使用本地 PyTorch 生成，也保留 SGLang HTTP 推理入口。
- `trainer_utils.init_model()` 默认从 `../model` 读取 tokenizer，因此文档中的命令统一假设从 `trainer/` 目录执行。
- `--use_wandb` 实际接入的是 `swanlab`，相关依赖已写入 `requirements.txt`。
- GRPO 默认从 `full_sft` 权重继续训练，需要可用的奖励模型路径；若仅调试链路，需要同时传入 `--reward_model_path none --allow_rule_reward_only`。

## 设计原则

- 先保证链路真实存在，再写文档，不预支未来功能。
- 优先把单卡路径打通，再考虑更复杂的工程抽象。
- 文档、依赖和脚本应在同一个提交中保持一致。
- 对人可见的说明文本优先使用中文；变量名、函数名、模块名保持英文。

## 路线图

1. 补齐官方 `DPO` 训练入口。
2. 增加评测与推理脚本。
3. 增加最小冒烟测试与持续集成流程。
4. 继续拆分训练 checkpoint 与推理权重导出职责。

## 贡献方式

- 欢迎提交问题、文档修订、训练脚本修复和最小可验证的功能补丁。
- 新增运行依赖时，请同步更新 `requirements.txt`。
- 修改路径约定、训练入口或目录结构时，请同步更新 `README.md` 和 `CLAUDE.md`。
- 请避免在文档中承诺仓库尚未交付的功能。

## 致谢

- 设计与数据组织参考了 [minimind](https://github.com/jingyaogong/minimind)。
- `dataset/data/` 中的数据说明和元数据文件保留了上游数据分发痕迹，二次分发前请自行核查来源与许可。

## 许可说明

仓库当前没有单独的 `LICENSE` 文件。若要作为公开开源仓库长期维护，建议在对外分发前补齐明确许可协议。
