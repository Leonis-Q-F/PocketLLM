"""
单卡 GRPO rollout 引擎。

职责：
1. 对接 SGLang 推理服务
2. 返回训练需要的 output_ids、completion_ids 和旧策略 logprob
3. 保持和训练主循环解耦
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import requests
import torch
from torch import Tensor
from transformers import AutoTokenizer


def compute_per_token_logps(
    model: torch.nn.Module,
    input_ids: Tensor,
    n_keep: int,
    attention_mask: Optional[Tensor] = None,
) -> Tensor:
    """计算最后 n_keep 个 token 的条件 logprob。"""
    if n_keep <= 0:
        return input_ids.new_empty((input_ids.size(0), 0), dtype=torch.float32)

    outputs = model(
        input_ids,
        attention_mask=attention_mask,
        logits_to_keep=n_keep + 1,
    )
    logits = outputs.logits[:, :-1, :]
    per_token_logps = []
    for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
        per_token_logps.append(
            torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1)
        )
    return torch.stack(per_token_logps)


@dataclass
class RolloutResult:
    output_ids: Tensor
    completion_ids: Tensor
    per_token_logps: Tensor
    completions: List[str]


class RolloutEngine(ABC):
    tokenizer = None

    @abstractmethod
    def rollout(
        self,
        prompt_ids: Tensor,
        attention_mask: Tensor,
        num_generations: int,
        max_new_tokens: int,
        temperature: float = 0.8,
    ) -> RolloutResult:
        raise NotImplementedError

    @abstractmethod
    def update_policy(self, model: torch.nn.Module):
        raise NotImplementedError


class SGLangRolloutEngine(RolloutEngine):
    def __init__(self, base_url: str, model_path: str, shared_ckpt_path: str = "./sglang_ckpt", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.shared_ckpt_path = shared_ckpt_path
        self.timeout = timeout
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.http = requests

    def rollout(
        self,
        prompt_ids: Tensor,
        attention_mask: Tensor,
        num_generations: int,
        max_new_tokens: int,
        temperature: float = 0.8,
    ) -> RolloutResult:
        input_ids_list = []
        for ids, mask in zip(prompt_ids, attention_mask):
            input_ids_list.append(ids[mask.bool()].tolist())
        all_input_ids = [ids for ids in input_ids_list for _ in range(num_generations)]

        payload = {
            "input_ids": all_input_ids,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "stop_token_ids": [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else [],
            },
            "return_logprob": True,
        }

        response = self.http.post(f"{self.base_url}/generate", json=payload, timeout=self.timeout)
        response.raise_for_status()

        results = response.json()
        if not isinstance(results, list):
            results = [results]

        all_output_ids = []
        all_completion_ids = []
        all_logprobs = []
        completions = []

        for prompt, result in zip(all_input_ids, results):
            meta = result.get("meta_info", {})
            completion_ids = meta.get("output_ids", result.get("output_ids", []))
            raw_logprobs = meta.get("output_token_logprobs", [])

            logprobs = []
            for item in raw_logprobs:
                if isinstance(item, (list, tuple)) and item:
                    logprobs.append(float(item[0]))
                elif isinstance(item, (int, float)):
                    logprobs.append(float(item))

            full_output = prompt + completion_ids
            all_output_ids.append(full_output)
            all_completion_ids.append(completion_ids)
            all_logprobs.append(logprobs)
            completions.append(self.tokenizer.decode(completion_ids, skip_special_tokens=True))

        device = prompt_ids.device
        max_out_len = max(len(ids) for ids in all_output_ids)
        max_comp_len = max(len(ids) for ids in all_completion_ids)
        max_logp_len = max(len(logprobs) for logprobs in all_logprobs)

        def pad_to_tensor(seqs, max_len, pad_val=0):
            return torch.tensor([seq + [pad_val] * (max_len - len(seq)) for seq in seqs], device=device)

        return RolloutResult(
            output_ids=pad_to_tensor(all_output_ids, max_out_len),
            completion_ids=pad_to_tensor(all_completion_ids, max_comp_len),
            per_token_logps=pad_to_tensor(all_logprobs, max_logp_len, pad_val=0.0),
            completions=completions,
        )

    def update_policy(self, model: torch.nn.Module):
        raw_model = getattr(model, "_orig_mod", model)
        abs_path = os.path.abspath(self.shared_ckpt_path)

        raw_model.lm_head.weight = torch.nn.Parameter(raw_model.lm_head.weight.clone())
        state_dict = {key: value.detach().half().cpu() for key, value in raw_model.state_dict().items()}
        raw_model.save_pretrained(abs_path, state_dict=state_dict, safe_serialization=False)
        raw_model.model.embed_tokens.weight = raw_model.lm_head.weight
        self.tokenizer.save_pretrained(abs_path)

        response = self.http.post(
            f"{self.base_url}/update_weights_from_disk",
            json={"model_path": abs_path},
            timeout=self.timeout,
        )
        if response.status_code != 200:
            print(f"[SGLANG WARNING] update_weights 失败: {response.status_code}, {response.text}")
        return response.status_code == 200

    def flush_cache(self) -> bool:
        response = self.http.post(f"{self.base_url}/flush_cache", timeout=30)
        return response.status_code == 200

    def health(self) -> bool:
        try:
            response = self.http.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False


def create_rollout_engine(
    sglang_base_url: str,
    sglang_model_path: str,
    sglang_shared_path: str,
) -> RolloutEngine:
    return SGLangRolloutEngine(sglang_base_url, sglang_model_path, sglang_shared_path)
