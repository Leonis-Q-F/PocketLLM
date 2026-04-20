# PocketLLM

> 一个面向单卡实验的轻量级 LLM 训练仓库，聚焦于清晰、直接、可裁剪的训练链路。

PocketLLM 用尽可能少的代码骨架，把 decoder-only 大语言模型从预训练一路串到 SFT、LoRA、DPO 和 GRPO。  
它适合作为本地实验、代码阅读和训练流程裁剪的起点。

## 项目内容

- 一个 decoder-only causal language model 主干
- GQA、RoPE、RMSNorm、SwiGLU，以及可选 MoE 前馈层
- 预训练、全量 SFT、LoRA、DPO、GRPO 五个训练入口
- 一套统一的数据适配器
- 一份本地 tokenizer
- 一个交互式推理脚本
- 一个权重转换与 LoRA 合并脚本

## 安装

环境要求：

- Python 3.10+
- PyTorch 2.2+
- 一张可用的 CUDA 显卡更合适

安装步骤：

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

## 数据约定

训练数据默认放在 `dataset/data/` 下，训练脚本按这个约定读取 JSONL 文件。

### 预训练

```jsonl
{"text": "这是一个预训练样本。"}
```

### SFT / LoRA

```jsonl
{
  "conversations": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好，欢迎使用 PocketLLM。"}
  ]
}
```

### DPO

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

### GRPO / RLAIF

`RLAIFDataset` 读取多轮对话，并通过 tokenizer 的 chat template 生成 rollout prompt。通常会将最后一轮留给待生成的 `assistant`。

## 最快跑通

如果只是先验证链路，可以先跑一遍最小预训练，再用推理脚本检查权重是否可用。

### 1. 准备一份最小预训练数据

```jsonl
{"text": "PocketLLM 是一个用于实验的小型语言模型训练仓库。"}
{"text": "这是一条用于冒烟测试的样本。"}
```

把它保存为：

```text
dataset/data/pretrain_t2t_mini.jsonl
```

### 2. 运行预训练

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

### 3. 回到仓库根目录做一次推理

```bash
cd ..
python eval_llm.py \
  --load_from model \
  --save_dir out \
  --weight pretrain_smoke \
  --hidden_size 512 \
  --num_hidden_layers 4
```

## 训练入口

每个阶段都拆成了单独脚本，便于阅读、替换和定向修改。

### 预训练

入口文件：[trainer/train_pre.py](trainer/train_pre.py)

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

### 全量 SFT

入口文件：[trainer/train_full_sft.py](trainer/train_full_sft.py)

```bash
cd trainer
python train_full_sft.py \
  --data_path ../dataset/data/sft_t2t_mini.jsonl \
  --save_dir ../out \
  --save_weight full_sft_smoke \
  --from_weight pretrain_smoke \
  --hidden_size 512 \
  --num_hidden_layers 4 \
  --max_seq_len 512 \
  --batch_size 4 \
  --accumulation_steps 4
```

### LoRA

入口文件：[trainer/train_lora.py](trainer/train_lora.py)

```bash
cd trainer
python train_lora.py \
  --data_path ../dataset/data/lora_medical.jsonl \
  --save_dir ../out \
  --lora_name lora_medical_smoke \
  --from_weight full_sft_smoke \
  --hidden_size 512 \
  --num_hidden_layers 4 \
  --max_seq_len 512 \
  --batch_size 4
```

### DPO

入口文件：[trainer/train_dpo.py](trainer/train_dpo.py)

```bash
cd trainer
python train_dpo.py \
  --data_path ../dataset/data/dpo.jsonl \
  --save_dir ../out \
  --save_weight dpo_smoke \
  --from_weight full_sft_smoke \
  --hidden_size 512 \
  --num_hidden_layers 4 \
  --max_seq_len 512 \
  --batch_size 2 \
  --accumulation_steps 4
```

### GRPO

入口文件：[trainer/train_grpo.py](trainer/train_grpo.py)

GRPO 这条链路会额外用到：

- 奖励模型目录，通过 `--reward_model_path` 指定
- SGLang 服务地址，通过 `--sglang_base_url` 指定

对应的 rollout 封装在 [trainer/rollout_engine.py](trainer/rollout_engine.py)。

```bash
cd trainer
python train_grpo.py \
  --data_path ../dataset/data/rlaif.jsonl \
  --save_dir ../out \
  --save_weight grpo_smoke \
  --from_weight full_sft_smoke \
  --reward_model_path /path/to/reward-model \
  --sglang_base_url http://localhost:8996 \
  --sglang_model_path ../model \
  --sglang_shared_path ./sglang_ckpt_grpo \
  --hidden_size 512 \
  --num_hidden_layers 4 \
  --batch_size 1 \
  --num_generations 4
```

## 推理

推理脚本是 [eval_llm.py](eval_llm.py)。

它支持两种加载方式：

- `--load_from model`：从仓库里的原生 torch 权重加载
- `--load_from <transformers目录>`：从导出的 Transformers 权重加载

如果你要叠加 LoRA，可以这样跑：

```bash
python eval_llm.py \
  --load_from model \
  --save_dir out \
  --weight full_sft_smoke \
  --lora_weight lora_medical_smoke \
  --hidden_size 512 \
  --num_hidden_layers 4
```

## 权重转换与 LoRA 合并

脚本位置：[scripts/convert_model.py](scripts/convert_model.py)

这个脚本包含几类常用转换：

- 原生 torch 权重转 Transformers 格式
- 原生 torch 权重转 PocketLLM Transformers 格式
- Transformers 权重转回 torch 权重
- 基座权重和 LoRA 权重合并
- chat template 的 `jinja <-> json` 转换

这个脚本当前是直接改文件底部参数再运行的风格：

```bash
python scripts/convert_model.py
```

## 使用说明

- 训练脚本默认从 `trainer/` 目录启动
- tokenizer 默认从 `../model` 读取
- `--use_swanlab` 和 `--swanlab_project` 在训练脚本里是统一可用的
- 续训时可以使用 `--from_resume 1`

## 致谢

- 数据组织和部分工程思路参考了 [minimind](https://github.com/jingyaogong/minimind)
