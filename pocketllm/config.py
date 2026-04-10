"""训练与模型配置。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    intermediate_size: int = 2048
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    dropout: float = 0.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    use_qk_norm: bool = True
    flash_attention: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    use_moe: bool = False
    moe_num_experts: int = 4
    moe_top_k: int = 1
    moe_intermediate_size: int = 2048
    moe_aux_loss_coef: float = 1e-2
    moe_layer_interval: int = 1

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size 必须能被 num_attention_heads 整除。")
        return self.hidden_size // self.num_attention_heads

    def clone(self) -> "ModelConfig":
        return replace(self)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


@dataclass
class GRPOConfig:
    num_generations: int = 4
    max_new_tokens: int = 128
    beta: float = 0.02
    epsilon: float = 0.2
    reward_strategy: str = "combined"


MODEL_PRESETS: dict[str, ModelConfig] = {
    "dense-44m": ModelConfig(
        hidden_size=512,
        num_layers=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        intermediate_size=1536,
    ),
    "dense-110m": ModelConfig(
        hidden_size=768,
        num_layers=12,
        num_attention_heads=12,
        num_key_value_heads=4,
        intermediate_size=2048,
    ),
    "moe-200m": ModelConfig(
        hidden_size=768,
        num_layers=12,
        num_attention_heads=12,
        num_key_value_heads=4,
        intermediate_size=2048,
        use_moe=True,
        moe_num_experts=4,
        moe_top_k=1,
        moe_intermediate_size=2048,
        moe_layer_interval=2,
    ),
}


def get_model_preset(name: str) -> ModelConfig:
    if name not in MODEL_PRESETS:
        supported = ", ".join(sorted(MODEL_PRESETS))
        raise KeyError(f"未知模型预设 {name!r}，可选值: {supported}")
    return MODEL_PRESETS[name].clone()
