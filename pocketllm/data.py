"""数据集与文本格式化。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from datasets import load_dataset
from torch.utils.data import Dataset


ROLE_PREFIX = {
    "system": "<system>\n",
    "user": "<user>\n",
    "assistant": "<assistant>\n",
}


@dataclass
class PromptSample:
    prompt: str
    answer: str | None = None


def _normalize_messages(sample: dict) -> list[dict[str, str]]:
    if "messages" in sample:
        return sample["messages"]
    if "conversations" in sample:
        return sample["conversations"]
    if "prompt" in sample and "response" in sample:
        return [
            {"role": "user", "content": sample["prompt"]},
            {"role": "assistant", "content": sample["response"]},
        ]
    if "instruction" in sample and "output" in sample:
        instruction = str(sample["instruction"])
        extra_input = str(sample.get("input", "")).strip()
        if extra_input:
            instruction = f"{instruction}\n{extra_input}"
        return [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": sample["output"]},
        ]
    raise KeyError("样本缺少可识别字段，需包含 messages/conversations/prompt-response/instruction-output。")


def render_messages(messages: list[dict[str, str]], add_generation_prompt: bool = False) -> str:
    chunks: list[str] = []
    for message in messages:
        role = message["role"].strip().lower()
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        prefix = ROLE_PREFIX.get(role, f"<{role}>\n")
        chunks.append(prefix)
        chunks.append(content)
        chunks.append("\n")
    if add_generation_prompt:
        chunks.append(ROLE_PREFIX["assistant"])
    return "".join(chunks)


class PretrainDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("json", data_files=data_path, split="train")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        text = str(self.samples[idx]["text"])
        token_ids = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length - 2,
        ).input_ids
        token_ids = [self.tokenizer.bos_token_id] + token_ids + [self.tokenizer.eos_token_id]
        padding = [self.tokenizer.pad_token_id] * (self.max_length - len(token_ids))
        input_ids = torch.tensor(token_ids + padding, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels


class SFTDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("json", data_files=data_path, split="train")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        messages = _normalize_messages(self.samples[idx])
        input_ids = [self.tokenizer.bos_token_id]
        labels = [-100]
        for message in messages:
            role = message["role"].strip().lower()
            content = str(message.get("content", "")).strip()
            prefix_ids = self.tokenizer(
                ROLE_PREFIX.get(role, f"<{role}>\n"),
                add_special_tokens=False,
            ).input_ids
            content_ids = self.tokenizer(content + "\n", add_special_tokens=False).input_ids
            eos_ids = [self.tokenizer.eos_token_id] if role == "assistant" else []
            input_ids.extend(prefix_ids + content_ids + eos_ids)
            if role == "assistant":
                labels.extend([-100] * len(prefix_ids))
                labels.extend(content_ids + eos_ids)
            else:
                labels.extend([-100] * (len(prefix_ids) + len(content_ids) + len(eos_ids)))

        input_ids = input_ids[: self.max_length]
        labels = labels[: self.max_length]
        pad_len = self.max_length - len(input_ids)
        input_ids.extend([self.tokenizer.pad_token_id] * pad_len)
        labels.extend([-100] * pad_len)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class PreferencePromptDataset(Dataset):
    def __init__(self, data_path: str) -> None:
        self.samples = load_dataset("json", data_files=data_path, split="train")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> PromptSample:
        sample = self.samples[idx]
        if "prompt" in sample:
            answer = sample.get("answer")
            return PromptSample(prompt=str(sample["prompt"]), answer=None if answer is None else str(answer))
        messages = _normalize_messages(sample)
        answer = None
        if messages and messages[-1]["role"].strip().lower() == "assistant":
            answer = str(messages[-1]["content"])
            messages = messages[:-1]
        prompt = render_messages(messages, add_generation_prompt=True)
        return PromptSample(prompt=prompt, answer=answer)


def preference_collate_fn(batch: list[PromptSample]) -> dict[str, list[str | None]]:
    return {
        "prompts": [sample.prompt for sample in batch],
        "answers": [sample.answer for sample in batch],
    }
