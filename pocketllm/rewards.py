"""简化版 reward 设计，方便先学会 GRPO 闭环。"""

from __future__ import annotations

import re

import torch


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _repetition_penalty(text: str, ngram: int = 3) -> float:
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    if len(tokens) < ngram:
        return 0.0
    grams = [tuple(tokens[i : i + ngram]) for i in range(len(tokens) - ngram + 1)]
    if not grams:
        return 0.0
    repeats = len(grams) - len(set(grams))
    return min(1.0, repeats / len(grams))


def score_completion(
    prompt: str,
    completion: str,
    answer: str | None,
    strategy: str = "combined",
) -> float:
    del prompt
    text = completion.strip()
    if not text:
        return -1.0
    score = 0.0
    if 8 <= len(text) <= 512:
        score += 0.3
    else:
        score -= 0.2
    score -= 0.5 * _repetition_penalty(text)

    if strategy in {"exact-match", "combined"} and answer:
        score += 1.5 if _normalize_text(text) == _normalize_text(answer) else -0.2
    if strategy == "length":
        return score
    return score


def batched_rewards(
    prompts: list[str],
    completions: list[str],
    answers: list[str | None],
    strategy: str,
    device: torch.device,
) -> torch.Tensor:
    scores = [
        score_completion(prompt, completion, answer, strategy=strategy)
        for prompt, completion, answer in zip(prompts, completions, answers)
    ]
    return torch.tensor(scores, dtype=torch.float32, device=device)
