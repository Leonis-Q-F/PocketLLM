"""SFT / LoRA SFT 入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from pocketllm.config import LoRAConfig, get_model_preset
from pocketllm.data import SFTDataset
from pocketllm.lora import apply_lora, lora_state_dict, mark_only_lora_trainable
from pocketllm.model import PocketLLMForCausalLM
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
    parser = argparse.ArgumentParser(description="PocketLLM SFT")
    parser.add_argument("--tokenizer-path", type=str, required=True, help="Tokenizer 路径或 Hugging Face 名称")
    parser.add_argument("--data-path", type=str, required=True, help="SFT jsonl 路径")
    parser.add_argument("--output-dir", type=str, default="out/sft", help="输出目录")
    parser.add_argument("--preset", type=str, default="dense-110m", help="模型预设")
    parser.add_argument("--init-checkpoint", type=str, default=None, help="预训练权重路径")
    parser.add_argument("--max-length", type=int, default=1024, help="训练最大长度")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--micro-batch-size", type=int, default=2, help="单步微批大小")
    parser.add_argument("--global-batch-size", type=int, default=16, help="全局 batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="warmup 比例")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="梯度裁剪")
    parser.add_argument("--dtype", type=str, default="bf16", help="训练精度")
    parser.add_argument("--device", type=str, default=None, help="训练设备")
    parser.add_argument("--num-workers", type=int, default=2, help="数据加载线程数")
    parser.add_argument("--log-every", type=int, default=10, help="日志步数")
    parser.add_argument("--save-every", type=int, default=100, help="保存步数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank，设为 0 表示全参数 SFT")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    return parser.parse_args()


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
    model_config.max_position_embeddings = args.max_length
    model_config.pad_token_id = tokenizer.pad_token_id
    model_config.bos_token_id = tokenizer.bos_token_id or tokenizer.cls_token_id or tokenizer.pad_token_id
    model_config.eos_token_id = tokenizer.eos_token_id or tokenizer.sep_token_id or tokenizer.pad_token_id

    model = PocketLLMForCausalLM(model_config).to(device)
    if args.init_checkpoint:
        load_checkpoint(args.init_checkpoint, model=model, map_location=device)

    trainable_params = list(model.parameters())
    lora_config = None
    if args.lora_rank > 0:
        lora_config = LoRAConfig(rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout)
        apply_lora(model, lora_config)
        trainable_params = mark_only_lora_trainable(model)

    dataset = SFTDataset(args.data_path, tokenizer, max_length=args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    accum_steps = max(1, args.global_batch_size // args.micro_batch_size)

    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    total_steps = max(1, len(loader) * args.epochs // accum_steps)
    scheduler = create_cosine_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=max(1, int(total_steps * args.warmup_ratio)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and dtype == torch.float16))
    autocast_ctx = build_autocast(device, dtype)

    save_json(output_dir / "model_config.json", model_config.to_dict())
    if lora_config:
        save_json(
            output_dir / "lora_config.json",
            {
                "rank": lora_config.rank,
                "alpha": lora_config.alpha,
                "dropout": lora_config.dropout,
                "target_modules": list(lora_config.target_modules),
            },
        )

    print(f"总参数量: {count_parameters(model) / 1e6:.2f}M")
    print(f"可训练参数量: {count_parameters(model, only_trainable=True) / 1e6:.2f}M")

    optimizer.zero_grad(set_to_none=True)
    update_step = 0
    for epoch in range(args.epochs):
        model.train()
        for step, (input_ids, labels) in enumerate(loader, start=1):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = input_ids.ne(tokenizer.pad_token_id).long()

            with autocast_ctx:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = (outputs.loss + outputs.aux_loss) / accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % accum_steps == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                update_step += 1

                if update_step % args.log_every == 0:
                    print(
                        f"[sft] epoch={epoch + 1} step={update_step} "
                        f"loss={(loss.item() * accum_steps):.4f} lr={scheduler.get_last_lr()[0]:.6e}"
                    )
                if update_step % args.save_every == 0:
                    checkpoint_path = output_dir / f"checkpoint-step-{update_step}.pt"
                    save_checkpoint(
                        checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=update_step,
                        extra={"stage": "sft", "model_config": model_config.to_dict()},
                    )
                    if lora_config:
                        torch.save(lora_state_dict(model), output_dir / f"adapter-step-{update_step}.pt")
        if len(loader) % accum_steps != 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            update_step += 1

    save_checkpoint(
        Path(output_dir) / "final.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=update_step,
        extra={"stage": "sft", "model_config": model_config.to_dict()},
    )
    if lora_config:
        torch.save(lora_state_dict(model), output_dir / "adapter-final.pt")


if __name__ == "__main__":
    main()
