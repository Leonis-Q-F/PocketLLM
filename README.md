# PocketLLM

单卡 `RTX 4070Ti` 取向的轻量级大模型全链路训练与对齐框架。

项目目标不是“套壳调库”，而是沿着一条足够短、足够清晰的路径，把下面这条链路真正打通：

`Pretrain -> SFT / LoRA -> GRPO`

当前仓库以 [minimind](https://github.com/jingyaogong/minimind) 为原型做了重新裁剪，但策略不是照搬：

- 保留它“核心代码自己写”的精神。
- 缩小第一阶段实现面，先让主干正确，再逐步长出 tokenizer、数据清洗、评测与更强 reward。
- 面向单卡资源，把模型、脚本、数据接口都做成可裁剪的形态。

## 现在有什么

- `pocketllm/model.py`
  - Decoder-only 主干
  - GQA 风格 attention
  - RoPE
  - RMSNorm
  - SwiGLU FFN
  - 可选 MoE + load balance auxiliary loss
- `pocketllm/lora.py`
  - LoRA 线性层替换
  - 冻结非 LoRA 参数
  - 导出 LoRA adapter
- `pocketllm/data.py`
  - 预训练数据集
  - SFT 数据集
  - GRPO prompt 数据集
- `train_pretrain.py`
  - 最小预训练入口
- `train_sft.py`
  - 全参数 SFT 或 LoRA SFT
- `train_grpo.py`
  - 教学版 GRPO 闭环

## 推荐学习顺序

1. 先读 `pocketllm/model.py`
   - 理解一层 Block 里 attention / FFN / residual 的骨架。
2. 再跑 `train_pretrain.py`
   - 先用很小的数据和 `dense-44m` 预设做 smoke test。
3. 然后看 `train_sft.py`
   - 对比全参数 SFT 和 LoRA SFT 的训练参数量差异。
4. 最后看 `train_grpo.py`
   - 先理解 rollout、reward、reference model、advantage、KL penalty 各自的职责。

## 单卡 4070Ti 的建议路线

- 第一阶段：`dense-44m`
  - 目标是把训练链路跑通。
- 第二阶段：`dense-110m`
  - 目标是观察预训练和 SFT 后的能力变化。
- 第三阶段：`moe-200m`
  - 目标是体验“总参数更大，但单 token 激活更少”的 MoE 路线。

不要一上来就冲 0.2B。先跑通，再扩张。好系统不是堆参数堆出来的，是靠边界干净长出来的。

## 数据格式

### 1. 预训练

jsonl 每行至少包含：

```json
{"text": "这是一个预训练样本。"}
```

### 2. SFT

支持以下任一种：

```json
{"messages": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好，我是 PocketLLM。"}]}
```

```json
{"prompt": "解释一下 Transformer。", "response": "Transformer 是一种以注意力机制为核心的序列建模架构。"}
```

```json
{"instruction": "用一句话解释 LoRA", "output": "LoRA 用低秩增量矩阵在极少参数上完成微调。"}
```

### 3. GRPO

最简单格式：

```json
{"prompt": "2 + 2 = ?", "answer": "4"}
```

也支持消息格式，若最后一条是 `assistant`，它会被视为参考答案，其前文会被视为 prompt。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 预训练

```bash
python train_pretrain.py \
  --tokenizer-path Qwen/Qwen2.5-0.5B \
  --data-path data/pretrain.jsonl \
  --preset dense-44m \
  --output-dir out/pretrain_smoke
```

### 3. LoRA SFT

```bash
python train_sft.py \
  --tokenizer-path Qwen/Qwen2.5-0.5B \
  --data-path data/sft.jsonl \
  --preset dense-44m \
  --init-checkpoint out/pretrain_smoke/final.pt \
  --output-dir out/sft_lora \
  --lora-rank 16
```

### 4. GRPO

```bash
python train_grpo.py \
  --tokenizer-path Qwen/Qwen2.5-0.5B \
  --data-path data/grpo.jsonl \
  --preset dense-44m \
  --policy-checkpoint out/sft_lora/final.pt \
  --output-dir out/grpo
```

## 和简历项目的关系

这个仓库当前是“第一版骨架”，它已经把简历项目的核心脊柱搭好了：

- 从 0 写 Decoder-only 主干
- 支持可选 MoE
- 支持 Pretrain / SFT / LoRA / GRPO
- 支持 reward 设计继续向 RLHF / RLAIF 扩展

但还没有一次性把所有东西都塞满。后续最值得继续补的部分是：

- tokenizer 训练与特殊 token 设计
- MNBVC / COIG 风格的数据清洗脚本
- C-Eval / CMMLU 评测脚本
- 更强的 reward model 接口
- checkpoint 导出与推理脚本

## 设计原则

- 先消灭不必要的特殊情况，再谈复杂技巧。
- 先让链路闭环，再做性能优化。
- 先做单卡能学明白的版本，再做更大的工程化版本。

如果一句话概括这个仓库的哲学：

“让数据顺着主干流，让复杂度死在边界上。”
