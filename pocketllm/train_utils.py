"""训练通用工具。"""

from __future__ import annotations

import json
import math
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if name not in mapping:
        supported = ", ".join(sorted(mapping))
        raise KeyError(f"未知 dtype {name!r}，可选值: {supported}")
    return mapping[name]


def build_autocast(device: torch.device, dtype: torch.dtype):
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def count_parameters(model: nn.Module, only_trainable: bool = False) -> int:
    params = model.parameters()
    if only_trainable:
        params = (param for param in params if param.requires_grad)
    return sum(param.numel() for param in params)


def create_cosine_scheduler(optimizer, total_steps: int, warmup_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_json(path: str | Path, payload: dict) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer=None,
    scheduler=None,
    step: int = 0,
    extra: dict | None = None,
) -> None:
    payload = {
        "model": model.state_dict(),
        "step": step,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if extra:
        payload["extra"] = extra
    torch.save(payload, Path(path))


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer=None,
    scheduler=None,
    map_location: str | torch.device = "cpu",
) -> dict:
    payload = torch.load(Path(path), map_location=map_location)
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    return payload


def prepare_tokenizer(tokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.bos_token_id is not None:
            tokenizer.pad_token = tokenizer.bos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
