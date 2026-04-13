"""Decoder-only + 可选 MoE 的 PocketLLM 主干实现。"""
import math, torch, torch.nn.functional as F
from torch import nn, Tensor
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

class PocketLLMConfig(PretrainedConfig):
    model_type = "PocketLLM"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size  # Transformer 隐藏维度，也是 token 表示的主宽度
        self.num_hidden_layers = num_hidden_layers  # Transformer Block 的层数
        self.use_moe = use_moe  # 是否将前馈层替换为 MoE 版本
        self.dropout = kwargs.get("dropout", 0.0)  # 全局 dropout 概率
        self.vocab_size = kwargs.get("vocab_size", 6400)  # 词表大小，决定输入嵌入和输出 logits 的宽度
        self.bos_token_id = kwargs.get("bos_token_id", 1)  # 句子起始 token id
        self.eos_token_id = kwargs.get("eos_token_id", 2)  # 句子结束 token id
        self.flash_attn = kwargs.get("flash_attn", True)  # 是否优先使用 PyTorch 的 Flash Attention
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)  # Query 头数量
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)  # Key/Value 头数量，用于 GQA
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)  # 每个注意力头的维度
        self.hidden_act = kwargs.get("hidden_act", 'silu')  # 前馈层激活函数
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)  # FFN 隐藏层维度
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)  # 预计算 RoPE 时支持的最大位置长度
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)  # RMSNorm 的数值稳定项
        self.rope_theta = kwargs.get("rope_theta", 1e6)  # RoPE 的基数，决定位置编码频率分布
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)  # 推理时是否启用 YaRN 长度外推
        self.rope_scaling = {
            "beta_fast": 32,  # 高频段插值边界
            "beta_slow": 1,  # 低频段插值边界
            "factor": 16,  # 外推缩放倍数
            "original_max_position_embeddings": 2048,  # 训练阶段原始上下文长度
            "attention_factor": 1.0,  # 额外的注意力缩放系数
            "type": "yarn"  # 当前使用的 RoPE 外推类型
        } if self.inference_rope_scaling else None
        # ==================== MoE 专属配置 ====================
        self.num_experts = kwargs.get("num_experts", 4)  # 专家总数
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)  # 每个 token 路由到多少个专家
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)  # 每个专家内部 FFN 的隐藏维度
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)  # 是否对 top-k 路由概率重新归一化
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)  # 路由均衡辅助损失的权重

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed * self.weight


# 生成旋转矩阵所需的正弦和余弦值
def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None: # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

# 执行旋转变换
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed

# 适配 GQA（分组查询注意力）  
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
        KV 头重复函数，将 KV 头的数量扩展到与 Query 头数量一致
    """
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))

class Attention(nn.Module):
    def __init__(self, config: PocketLLMConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads # Key/Value 头数量可以独立于 Query 头数量配置，以支持 GQA
        self.n_local_heads = config.num_attention_heads # 实际使用的 Query 头数量
        self.n_local_kv_heads = self.num_key_value_heads # 实际使用的 Key/Value 头数量
        self.n_rep = self.n_local_heads // self.n_local_kv_heads # 每个 Key/Value 头需要被重复多少次以匹配 Query 头数量，必须是整数，否则会在 forward 中报错
        self.head_dim = config.head_dim # 每个注意力头的维度，通常是 hidden_size // num_attention_heads
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.is_causal = True
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape  # 批量大小，序列长度，嵌入维度
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)  # 计算Q、K、V

        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)  # 拆成多个注意力头
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        xq, xk = self.q_norm(xq), self.k_norm(xk)  # 应用RMSNorm归一化
        cos, sin = position_embeddings  # 获取旋转位置嵌入
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)  # 应用旋转位置嵌入

        if past_key_value is not None:  # 如果提供了有历史KV，则将其与当前计算的 KV 拼接起来
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None  # 将past_key_value中的KV拼接到当前KV中
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2),
                      repeat_kv(xv, self.n_rep).transpose(1, 2))  # 调整维度以适配注意力计算，重复KV以匹配Query头数量（GQA）

        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            # 使用 PyTorch 的 Flash Attention 计算注意力输出，自动处理因果掩码和 dropout
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
        else:
            # 公式：(Q * K ^ T) / sqrt(d_k)
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)  # 计算注意力分数，缩放因子为头维度的平方根
            if self.is_causal: scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1) # 添加因果掩码 防止偷看未来
            if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9 # 添加注意力掩码 让模型不关注无用的padding
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv # 计算注意力权重并应用于V，得到注意力输出

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1) # 将多头注意力输出重新组合成原始的嵌入维度
        output = self.resid_dropout(self.o_proj(output)) # 最后通过输出投影层，并应用残差连接前的 dropout
        return output, past_kv

# 普通的前馈层实现，包含一个门控机制（Gated Linear Unit）和两个线性变换，激活函数可配置
class FeedForward(nn.Module):
# 公式：FFN(x) = W_down * (act(W_gate * x) ⊙ W_up * x)，其中 ⊙ 表示逐元素乘法，W_gate、W_up 和 W_down 分别是三个线性变换矩阵，act 是激活函数
    def __init__(self, config: PocketLLMConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size 
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# MOE专家类
class MOEFeedForward(nn.Module):
    def __init__(self, config: PocketLLMConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim) # 将输入展平为 (batch_size * seq_len, hidden_dim)，以便对每个 token 进行独立的专家路由和计算
        scores = F.softmax(self.gate(x_flat), dim=-1) # 计算每个 token 路由到各个专家的概率分布，得到 (batch_size * seq_len, num_experts) 的分数矩阵 “路由器”
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False) # 对每个 token 选择 top-k 个专家，得到对应的权重和索引，形状为 (batch_size * seq_len, num_experts_per_tok)

        # 可选地对 top-k 权重进行重新归一化，使其和为 1，增加数值稳定性
        if self.config.norm_topk_prob: topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20) 
        y = torch.zeros_like(x_flat)

        # 遍历每个专家，检查哪些 token 被路由到该专家，并将其计算结果加权累加到输出 y 中
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())

        # 计算路由均衡的辅助损失，鼓励模型在专家之间分配负载，防止某些专家过载而其他专家闲置
        if self.training and self.config.router_aux_loss_coef > 0:
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()

        return y.view(batch_size, seq_len, hidden_dim)

# Transformer Block
class PocketLLMBlock(nn.Module):
    def __init__(self, layer_id: int, config: PocketLLMConfig):
        super().__init__()
        self.self_attn = Attention(config) 
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) # 在注意力输入前添加一个 LayerNorm
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) # 在注意力输出后添加一个 LayerNorm
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config) # 根据配置选择使用普通前馈层还是 MoE 前馈层

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        # 执行自注意力计算，输入经过 LayerNorm 归一化，并传入位置嵌入、历史 KV 和注意力掩码等信息，得到新的隐藏状态和当前的 KV 用于缓存
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual # 添加残差连接，将注意力输出与输入相加
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states)) # 经过另一个 LayerNorm 归一化后，输入前馈层计算，并添加残差连接
        return hidden_states, present_key_value


# PocketLLM 主干模型
class PocketLLMModel(nn.Module):
    def __init__(self, config: PocketLLMConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([PocketLLMBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers) # 初始化 past_key_values 为 None，用于存储历史 KV
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0 # 计算起始位置，即历史 KV 中最后一个 token 的位置
        hidden_states = self.dropout(self.embed_tokens(input_ids)) # 输入经过嵌入层，得到隐藏状态
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length]) # 获取位置嵌入，用于计算 Rope
        presents = [] # 初始化 presents 列表，用于存储每个 layer 的当前 KV
        # 遍历每个 layer，执行自注意力计算和前馈计算，并添加残差连接
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        hidden_states = self.norm(hidden_states) # 添加 LayerNorm 归一化
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze()) # 计算 MoE 的辅助损失
        return hidden_states, presents, aux_loss # 返回隐藏状态、当前 KV 和 MoE 的辅助损失


class PocketLLMForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = PocketLLMConfig
    def __init__(self, config: PocketLLMConfig = None):
        self.config = config or PocketLLMConfig()
        super().__init__(self.config)
        self.model = PocketLLMModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight
    
    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)

    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer: streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            logits = outputs.logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]): logits[i, torch.unique(input_ids[i])] /= repetition_penalty
            if top_k > 0: 
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer: streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break
        if streamer: streamer.end()
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids


if __name__ == "__main__":
    config = PocketLLMConfig()
    model = PocketLLMModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))  # batch_size=2, seq_length=16
    attention_mask = torch.ones_like(input_ids)  # 全部位置都有效
    hidden_states, presents, aux_loss = model(input_ids, attention_mask=attention_mask)
    print("Hidden states shape:", hidden_states.shape)  # 应该是 (2, 16, hidden_size)
    print("Auxiliary loss:", aux_loss.item())
