import torch
from torch import nn


class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank # 计算缩放因子

        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)

        self.A.weight.data.normal_(mean=0.0, std=0.02)
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x)) * self.scaling


def apply_lora(model, rank=16, alpha=16, target_modules=None):
    target_layers = []

    # 先冻结目标列表，再做注入
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if ".lora." in name or hasattr(module, "lora"):
            continue
        if target_modules is not None and not any(target in name for target in target_modules):
            continue
        target_layers.append(module)

    for module in target_layers:
        lora = LoRA(
            module.in_features,
            module.out_features,
            rank=rank,
            alpha=alpha,
        ).to(device=module.weight.device, dtype=module.weight.dtype)
        setattr(module, "lora", lora)
        original_forward = module.forward

        # 显式绑定当前层，避免闭包引用错位。
        def forward_with_lora(x, base_forward=original_forward, lora_layer=lora):
            return base_forward(x) + lora_layer(x)

        module.forward = forward_with_lora

    for name, param in model.named_parameters():
        param.requires_grad = "lora" in name

    print(f"LoRA applied. Rank: {rank}, Alpha: {alpha}, Layers: {len(target_layers)}")


def load_lora(model, path):
    device = next(model.parameters()).device
    state_dict = torch.load(path, map_location=device)

    for name, module in model.named_modules():
        if not hasattr(module, "lora"):
            continue

        prefix = f"{name}.lora."
        lora_state = {}
        for key, value in state_dict.items():
            if key.startswith(prefix):
                lora_state[key.replace(prefix, "")] = value

        module.lora.load_state_dict(lora_state, strict=False)


def save_lora(model, path):
    raw_model = getattr(model, "_orig_mod", model)
    state_dict = {}

    for name, module in raw_model.named_modules():
        if not hasattr(module, "lora"):
            continue

        clean_name = name[7:] if name.startswith("module.") else name
        lora_state = {
            f"{clean_name}.lora.{key}": value.detach().cpu().half()
            for key, value in module.lora.state_dict().items()
        }
        state_dict.update(lora_state)

    torch.save(state_dict, path)


def merge_lora(model, lora_path, save_path):
    load_lora(model, lora_path)
    raw_model = getattr(model, "_orig_mod", model)

    state_dict = {
        key: value.detach().cpu().half()
        for key, value in raw_model.state_dict().items()
        if ".lora." not in key
    }

    with torch.no_grad():
        for name, module in raw_model.named_modules():
            if not isinstance(module, nn.Linear) or ".lora." in name:
                continue

            merged_weight = module.weight.detach().float().cpu().clone()
            if hasattr(module, "lora"):
                delta = (
                    module.lora.B.weight.detach().float()
                    @ module.lora.A.weight.detach().float()
                ) * float(module.lora.scaling)
                merged_weight += delta.cpu()

            state_dict[f"{name}.weight"] = merged_weight.half()

    torch.save(state_dict, save_path)
