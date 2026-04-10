"""LoRA 适配器。"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import LoRAConfig


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int, alpha: int, dropout: float) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.empty(rank, base_layer.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base_layer.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        base = self.base_layer(x)
        delta = F.linear(self.dropout(x), self.lora_a)
        delta = F.linear(delta, self.lora_b) * self.scale
        return base + delta


def _set_module(root: nn.Module, dotted_name: str, module: nn.Module) -> None:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def apply_lora(model: nn.Module, config: LoRAConfig) -> nn.Module:
    modules_to_replace: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name.split(".")[-1] not in config.target_modules:
            continue
        modules_to_replace.append((name, module))

    for name, module in modules_to_replace:
        _set_module(
            model,
            name,
            LoRALinear(module, rank=config.rank, alpha=config.alpha, dropout=config.dropout),
        )
    return model


def mark_only_lora_trainable(model: nn.Module) -> list[nn.Parameter]:
    trainable_params: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        requires_grad = "lora_" in name
        param.requires_grad = requires_grad
        if requires_grad:
            trainable_params.append(param)
    return trainable_params


def lora_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: param.detach().cpu()
        for name, param in model.state_dict().items()
        if "lora_" in name
    }
