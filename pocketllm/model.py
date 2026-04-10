"""Decoder-only + 可选 MoE 的 PocketLLM 主干实现。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import ModelConfig


@dataclass
class CausalLMOutput:
    logits: Tensor
    hidden_states: Tensor
    loss: Tensor | None = None
    aux_loss: Tensor | None = None
    past_key_values: list[tuple[Tensor, Tensor] | None] | None = None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed * self.weight


def precompute_rotary_cache(
    dim: int,
    max_position_embeddings: int,
    theta: float,
) -> tuple[Tensor, Tensor]:
    positions = torch.arange(max_position_embeddings, dtype=torch.float32)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    angles = torch.outer(positions, freqs)
    cos = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1)
    sin = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1)
    return cos, sin


def rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def repeat_kv(x: Tensor, repeats: int) -> Tensor:
    if repeats == 1:
        return x
    batch_size, seq_len, kv_heads, head_dim = x.shape
    expanded = x[:, :, :, None, :].expand(batch_size, seq_len, kv_heads, repeats, head_dim)
    return expanded.reshape(batch_size, seq_len, kv_heads * repeats, head_dim)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.kv_repeats = self.num_heads // self.num_kv_heads
        self.dropout = config.dropout
        self.use_flash_attention = config.flash_attention and hasattr(
            F,
            "scaled_dot_product_attention",
        )

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps) if config.use_qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps) if config.use_qk_norm else nn.Identity()
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def _manual_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Tensor | None,
    ) -> Tensor:
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        query_len = q.size(-2)
        key_len = k.size(-2)
        query_positions = torch.arange(key_len - query_len, key_len, device=q.device)
        key_positions = torch.arange(key_len, device=q.device)
        causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        scores = scores.masked_fill(~causal_mask.view(1, 1, query_len, key_len), float("-inf"))
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask[:, None, None, :key_len] == 0, float("-inf"))
        probs = F.softmax(scores.float(), dim=-1).to(q.dtype)
        probs = self.attn_dropout(probs)
        return torch.matmul(probs, v)

    def forward(
        self,
        x: Tensor,
        rotary_cos: Tensor,
        rotary_sin: Tensor,
        attention_mask: Tensor | None = None,
        past_key_value: tuple[Tensor, Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rotary(q, k, rotary_cos, rotary_sin)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)

        present = (k, v) if use_cache else None
        q = q.transpose(1, 2)
        k = repeat_kv(k, self.kv_repeats).transpose(1, 2)
        v = repeat_kv(v, self.kv_repeats).transpose(1, 2)

        can_use_flash = self.use_flash_attention and attention_mask is None and past_key_value is None
        if can_use_flash:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            y = self._manual_attention(q, k, v, attention_mask)

        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.resid_dropout(self.o_proj(y)), present


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig, intermediate_size: int | None = None) -> None:
        super().__init__()
        inner_dim = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, inner_dim, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, inner_dim, bias=False)
        self.down_proj = nn.Linear(inner_dim, config.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEFeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.router = nn.Linear(config.hidden_size, config.moe_num_experts, bias=False)
        self.experts = nn.ModuleList(
            FeedForward(config, intermediate_size=config.moe_intermediate_size)
            for _ in range(config.moe_num_experts)
        )
        self.aux_loss: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, hidden_size = x.shape
        flat = x.reshape(-1, hidden_size)
        router_logits = self.router(flat)
        router_probs = F.softmax(router_logits, dim=-1)
        topk_weight, topk_idx = torch.topk(router_probs, k=self.config.moe_top_k, dim=-1)
        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        output = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            expert_locations = (topk_idx == expert_id).nonzero(as_tuple=False)
            if expert_locations.numel() == 0:
                continue
            token_ids = expert_locations[:, 0]
            topk_ids = expert_locations[:, 1]
            expert_out = expert(flat[token_ids])
            routed_weight = topk_weight[token_ids, topk_ids].unsqueeze(-1)
            output.index_add_(0, token_ids, expert_out * routed_weight)

        routed = F.one_hot(topk_idx, num_classes=self.config.moe_num_experts).sum(dim=1).float()
        expert_load = routed.mean(dim=0)
        expert_importance = router_probs.mean(dim=0)
        self.aux_loss = (
            self.config.moe_num_experts
            * torch.sum(expert_load * expert_importance)
            * self.config.moe_aux_loss_coef
        )
        return output.view(batch_size, seq_len, hidden_size)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = CausalSelfAttention(config)
        use_moe = config.use_moe and ((layer_idx + 1) % config.moe_layer_interval == 0)
        self.ffn = MoEFeedForward(config) if use_moe else FeedForward(config)

    def forward(
        self,
        x: Tensor,
        rotary_cos: Tensor,
        rotary_sin: Tensor,
        attention_mask: Tensor | None = None,
        past_key_value: tuple[Tensor, Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None, Tensor]:
        attn_out, present = self.attn(
            self.attn_norm(x),
            rotary_cos,
            rotary_sin,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        x = x + attn_out
        ffn_out = self.ffn(self.ffn_norm(x))
        x = x + ffn_out
        aux_loss = getattr(self.ffn, "aux_loss", None)
        if aux_loss is None:
            aux_loss = x.new_zeros(())
        return x, present, aux_loss


class PocketLLMModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            TransformerBlock(config, layer_idx=layer_idx)
            for layer_idx in range(config.num_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        cos, sin = precompute_rotary_cache(
            dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            theta=config.rope_theta,
        )
        self.register_buffer("rotary_cos", cos, persistent=False)
        self.register_buffer("rotary_sin", sin, persistent=False)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        past_key_values: list[tuple[Tensor, Tensor] | None] | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor] | None] | None, Tensor]:
        _, seq_len = input_ids.shape
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        start_pos = 0
        if past_key_values and past_key_values[0] is not None:
            start_pos = past_key_values[0][0].shape[1]
        rotary_cos = self.rotary_cos[start_pos : start_pos + seq_len]
        rotary_sin = self.rotary_sin[start_pos : start_pos + seq_len]

        hidden_states = self.dropout(self.embed_tokens(input_ids))
        new_past = [] if use_cache else None
        aux_loss = hidden_states.new_zeros(())
        for layer, past in zip(self.layers, past_key_values):
            hidden_states, present, layer_aux = layer(
                hidden_states,
                rotary_cos,
                rotary_sin,
                attention_mask=attention_mask,
                past_key_value=past,
                use_cache=use_cache,
            )
            aux_loss = aux_loss + layer_aux
            if use_cache:
                new_past.append(present)
        hidden_states = self.norm(hidden_states)
        return hidden_states, new_past, aux_loss


class PocketLLMForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.model = PocketLLMModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        past_key_values: list[tuple[Tensor, Tensor] | None] | None = None,
        use_cache: bool = False,
    ) -> CausalLMOutput:
        hidden_states, new_past, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return CausalLMOutput(
            logits=logits,
            hidden_states=hidden_states,
            loss=loss,
            aux_loss=aux_loss,
            past_key_values=new_past,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
    ) -> Tensor:
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)
        generated = input_ids
        past_key_values = None
        finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)

        for _ in range(max_new_tokens):
            current_input = generated if past_key_values is None else generated[:, -1:]
            outputs = self(
                input_ids=current_input,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :]
            if repetition_penalty != 1.0:
                for batch_idx in range(generated.size(0)):
                    logits[batch_idx, torch.unique(generated[batch_idx])] /= repetition_penalty
            if temperature <= 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k > 0:
                    threshold = torch.topk(logits, k=top_k, dim=-1).values[:, -1].unsqueeze(-1)
                    logits = logits.masked_fill(logits < threshold, float("-inf"))
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    probs = F.softmax(sorted_logits, dim=-1)
                    sorted_mask = torch.cumsum(probs, dim=-1) > top_p
                    sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
                    sorted_mask[:, 0] = 0
                    logits = logits.scatter(
                        1,
                        sorted_indices,
                        sorted_logits.masked_fill(sorted_mask, float("-inf")),
                    )
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            next_token = torch.where(
                finished.unsqueeze(-1),
                next_token.new_full((generated.size(0), 1), self.config.eos_token_id),
                next_token,
            )
            generated = torch.cat([generated, next_token], dim=-1)
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.size(0), 1))],
                dim=-1,
            )
            past_key_values = outputs.past_key_values
            finished = finished | next_token.squeeze(-1).eq(self.config.eos_token_id)
            if finished.all():
                break
        return generated
