"""预训练入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from pocketllm.config import get_model_preset
from pocketllm.data import PretrainDataset
from pocketllm.model import PocketLLMForCausalLM
from pocketllm.train_utils import (
    build_autocast,
    count_parameters,
    create_cosine_scheduler,
    ensure_output_dir,
    prepare_tokenizer,
    resolve_device,
    resolve_dtype,
    save_checkpoint,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PocketLLM 预训练")
    parser.add_argument("--tokenizer-path", type=str, required=True, help="Tokenizer 路径或 Hugging Face 名称")
    parser.add_argument("--data-path", type=str, required=True, help="预训练 jsonl 路径，字段需包含 text")
    parser.add_argument("--output-dir", type=str, default="out/pretrain", help="输出目录")
    parser.add_argument("--preset", type=str, default="dense-110m", help="模型预设")
    parser.add_argument("--max-length", type=int, default=512, help="训练最大长度")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--micro-batch-size", type=int, default=4, help="单步微批大小")
    parser.add_argument("--global-batch-size", type=int, default=32, help="全局 batch size")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="权重衰减")
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="warmup 比例")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="梯度裁剪")
    parser.add_argument("--dtype", type=str, default="bf16", help="训练精度")
    parser.add_argument("--device", type=str, default=None, help="训练设备")
    parser.add_argument("--num-workers", type=int, default=2, help="数据加载线程数")
    parser.add_argument("--log-every", type=int, default=20, help="日志步数")
    parser.add_argument("--save-every", type=int, default=200, help="保存步数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
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

    dataset = PretrainDataset(args.data_path, tokenizer, max_length=args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    accum_steps = max(1, args.global_batch_size // args.micro_batch_size)

    model = PocketLLMForCausalLM(model_config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    total_steps = max(1, len(loader) * args.epochs // accum_steps)
    scheduler = create_cosine_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=max(1, int(total_steps * args.warmup_ratio)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and dtype == torch.float16))

    save_json(output_dir / "model_config.json", model_config.to_dict())
    print(f"总参数量: {count_parameters(model) / 1e6:.2f}M")
    print(f"可训练参数量: {count_parameters(model, only_trainable=True) / 1e6:.2f}M")

    optimizer.zero_grad(set_to_none=True)
    autocast_ctx = build_autocast(device, dtype)
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
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
                        f"[pretrain] epoch={epoch + 1} step={update_step} "
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
                        extra={"stage": "pretrain", "model_config": model_config.to_dict()},
                    )
        if len(loader) % accum_steps != 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
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
        extra={"stage": "pretrain", "model_config": model_config.to_dict()},
    )


if __name__ == "__main__":
    main()
