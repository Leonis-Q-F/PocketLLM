"""简化版 GRPO 入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from pocketllm.config import GRPOConfig, get_model_preset
from pocketllm.data import PreferencePromptDataset, preference_collate_fn
from pocketllm.model import PocketLLMForCausalLM
from pocketllm.rewards import batched_rewards
from pocketllm.train_utils import (
    build_autocast,
    count_parameters,
    create_cosine_scheduler,
    ensure_output_dir,
    load_checkpoint,
    prepare_tokenizer,
    resolve_device,
    resolve_dtype,
    save_checkpoint,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PocketLLM GRPO")
    parser.add_argument("--tokenizer-path", type=str, required=True, help="Tokenizer 路径或 Hugging Face 名称")
    parser.add_argument("--data-path", type=str, required=True, help="Prompt jsonl 路径")
    parser.add_argument("--policy-checkpoint", type=str, required=True, help="Policy 权重路径")
    parser.add_argument("--reference-checkpoint", type=str, default=None, help="Reference 权重路径，默认与 policy 相同")
    parser.add_argument("--output-dir", type=str, default="out/grpo", help="输出目录")
    parser.add_argument("--preset", type=str, default="dense-110m", help="模型预设")
    parser.add_argument("--max-length", type=int, default=1024, help="Prompt 最大长度")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=1, help="每批 prompt 数")
    parser.add_argument("--learning-rate", type=float, default=5e-6, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="权重衰减")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="warmup 比例")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="梯度裁剪")
    parser.add_argument("--dtype", type=str, default="bf16", help="训练精度")
    parser.add_argument("--device", type=str, default=None, help="训练设备")
    parser.add_argument("--num-workers", type=int, default=0, help="数据加载线程数")
    parser.add_argument("--log-every", type=int, default=1, help="日志步数")
    parser.add_argument("--save-every", type=int, default=20, help="保存步数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num-generations", type=int, default=4, help="每个 prompt 采样数")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="最大生成长度")
    parser.add_argument("--beta", type=float, default=0.02, help="KL 惩罚")
    parser.add_argument("--epsilon", type=float, default=0.2, help="clip 区间")
    parser.add_argument("--temperature", type=float, default=0.8, help="采样温度")
    parser.add_argument("--top-p", type=float, default=0.95, help="top-p")
    parser.add_argument("--reward-strategy", type=str, default="combined", help="reward 策略")
    return parser.parse_args()


def decode_completions(tokenizer, completion_ids: torch.Tensor) -> list[str]:
    texts: list[str] = []
    for row in completion_ids:
        tokens = row.tolist()
        if tokenizer.eos_token_id in tokens:
            tokens = tokens[: tokens.index(tokenizer.eos_token_id)]
        texts.append(tokenizer.decode(tokens, skip_special_tokens=True).strip())
    return texts


def completion_mask_from_ids(completion_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    mask = torch.ones_like(completion_ids, dtype=torch.float32)
    for row_idx in range(completion_ids.size(0)):
        row = completion_ids[row_idx]
        eos_positions = (row == eos_token_id).nonzero(as_tuple=False)
        if eos_positions.numel() == 0:
            continue
        first_eos = int(eos_positions[0].item())
        if first_eos + 1 < row.size(0):
            mask[row_idx, first_eos + 1 :] = 0.0
    return mask


def compute_per_token_logps(
    model: PocketLLMForCausalLM,
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_length: int,
) -> torch.Tensor:
    outputs = model(
        input_ids=sequences[:, :-1],
        attention_mask=attention_mask[:, :-1],
    )
    log_probs = F.log_softmax(outputs.logits, dim=-1)
    token_logps = log_probs.gather(2, sequences[:, 1:].unsqueeze(-1)).squeeze(-1)
    return token_logps[:, -completion_length:]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    output_dir = ensure_output_dir(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    prepare_tokenizer(tokenizer)

    model_config = get_model_preset(args.preset)
    model_config.vocab_size = len(tokenizer)
    model_config.max_position_embeddings = max(args.max_length, args.max_new_tokens + args.max_length)
    model_config.pad_token_id = tokenizer.pad_token_id
    model_config.bos_token_id = tokenizer.bos_token_id or tokenizer.cls_token_id or tokenizer.pad_token_id
    model_config.eos_token_id = tokenizer.eos_token_id or tokenizer.sep_token_id or tokenizer.pad_token_id

    policy = PocketLLMForCausalLM(model_config).to(device)
    reference = PocketLLMForCausalLM(model_config).to(device)
    load_checkpoint(args.policy_checkpoint, model=policy, map_location=device)
    load_checkpoint(args.reference_checkpoint or args.policy_checkpoint, model=reference, map_location=device)
    reference.eval()
    for param in reference.parameters():
        param.requires_grad = False

    dataset = PreferencePromptDataset(args.data_path)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=preference_collate_fn,
    )

    grpo_config = GRPOConfig(
        num_generations=args.num_generations,
        max_new_tokens=args.max_new_tokens,
        beta=args.beta,
        epsilon=args.epsilon,
        reward_strategy=args.reward_strategy,
    )
    optimizer = AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = max(1, len(loader) * args.epochs)
    scheduler = create_cosine_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=max(1, int(total_steps * args.warmup_ratio)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and dtype == torch.float16))
    autocast_ctx = build_autocast(device, dtype)

    save_json(output_dir / "model_config.json", model_config.to_dict())
    save_json(
        output_dir / "grpo_config.json",
        {
            "num_generations": grpo_config.num_generations,
            "max_new_tokens": grpo_config.max_new_tokens,
            "beta": grpo_config.beta,
            "epsilon": grpo_config.epsilon,
            "reward_strategy": grpo_config.reward_strategy,
        },
    )

    print(f"Policy 参数量: {count_parameters(policy) / 1e6:.2f}M")
    update_step = 0
    for epoch in range(args.epochs):
        policy.train()
        for batch in loader:
            prompts = batch["prompts"]
            answers = batch["answers"]
            batch_losses = []
            batch_rewards = []
            for prompt, answer in zip(prompts, answers):
                prompt_batch = [prompt] * grpo_config.num_generations
                encoded = tokenizer(
                    prompt_batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                    add_special_tokens=False,
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}

                with torch.no_grad():
                    sequences = policy.generate(
                        input_ids=encoded["input_ids"],
                        attention_mask=encoded["attention_mask"],
                        max_new_tokens=grpo_config.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )

                completion_ids = sequences[:, encoded["input_ids"].size(1) :]
                full_attention_mask = sequences.ne(tokenizer.pad_token_id).long()
                completion_mask = completion_mask_from_ids(completion_ids, model_config.eos_token_id).to(device)
                completions = decode_completions(tokenizer, completion_ids)
                rewards = batched_rewards(
                    prompt_batch,
                    completions,
                    [answer] * grpo_config.num_generations,
                    strategy=grpo_config.reward_strategy,
                    device=device,
                )
                reward_mean = rewards.mean()
                reward_std = rewards.std(unbiased=False).clamp_min(1e-4)
                advantages = (rewards - reward_mean) / reward_std

                with torch.no_grad():
                    old_logps = compute_per_token_logps(
                        policy,
                        sequences=sequences,
                        attention_mask=full_attention_mask,
                        completion_length=completion_ids.size(1),
                    )
                    ref_logps = compute_per_token_logps(
                        reference,
                        sequences=sequences,
                        attention_mask=full_attention_mask,
                        completion_length=completion_ids.size(1),
                    )

                with autocast_ctx:
                    current_logps = compute_per_token_logps(
                        policy,
                        sequences=sequences,
                        attention_mask=full_attention_mask,
                        completion_length=completion_ids.size(1),
                    )
                    ratio = torch.exp(current_logps - old_logps)
                    clipped_ratio = torch.clamp(ratio, 1 - grpo_config.epsilon, 1 + grpo_config.epsilon)
                    advantage_matrix = advantages.unsqueeze(1)
                    surrogate = torch.min(ratio * advantage_matrix, clipped_ratio * advantage_matrix)
                    kl = torch.exp(ref_logps - current_logps) - (ref_logps - current_logps) - 1.0
                    loss_matrix = -surrogate + grpo_config.beta * kl
                    prompt_loss = (loss_matrix * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp_min(1.0)
                    batch_losses.append(prompt_loss.mean())
                    batch_rewards.append(rewards.mean())

            loss = torch.stack(batch_losses).mean()
            reward_value = torch.stack(batch_rewards).mean()

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
                optimizer.step()
            scheduler.step()
            update_step += 1

            if update_step % args.log_every == 0:
                print(
                    f"[grpo] epoch={epoch + 1} step={update_step} "
                    f"loss={loss.item():.4f} reward={reward_value.item():.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.6e}"
                )
            if update_step % args.save_every == 0:
                save_checkpoint(
                    output_dir / f"checkpoint-step-{update_step}.pt",
                    model=policy,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=update_step,
                    extra={"stage": "grpo", "model_config": model_config.to_dict()},
                )

    save_checkpoint(
        Path(output_dir) / "final.pt",
        model=policy,
        optimizer=optimizer,
        scheduler=scheduler,
        step=update_step,
        extra={"stage": "grpo", "model_config": model_config.to_dict()},
    )


if __name__ == "__main__":
    main()
