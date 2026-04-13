import torch
from torch import optim, nn


# 定义Lora网络结构
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank  # 计算缩放因子

        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)

        # 矩阵A高斯初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全0初始化
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x)) * self.scaling


# 应用LoRA到模型
def apply_lora(model, rank=16, alpha=16, target_modules=None):
    for name, module in model.named_modules():
        # 检查是否为目标层
        is_target = False
        if isinstance(module, nn.Linear):
            if target_modules is None:
                is_target = True  # 默认应用于所有线性层
            else:
                is_target = any(target in name for target in target_modules)

        if is_target:
            device = module.weight.device
            lora = LoRA(module.in_features, module.out_features, rank=rank, alpha=alpha).to(device)
            setattr(module, "lora", lora) # 将lora模块作为属性添加到原始模块中
            original_forward = module.forward

            # 显式绑定，防止闭包陷阱
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora

    for name, param in model.named_parameters():
        if 'lora' not in name: 
            param.requires_grad = False # 冻结参数
        else:
            param.requires_grad = True  # 该层参与训练

    print(f"LoRA applied. Rank: {rank}, Alpha: {alpha}")


def load_lora(model, path):
    device = next(model.parameters()).device
    state_dict = torch.load(path, map_location=device)

    # 遍历当前内存中的模型，寻找那些被我们改造过、带有 lora 属性的层。
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            # 提取属于当前这一层的 LoRA 权重，并加载进去。
            lora_state = {}

            for k, v in state_dict.items():
                if f'{name}.lora.' in k:
                    new_key = k.replace(f'{name}.lora.', '')
                    lora_state[new_key] = v

            module.lora.load_state_dict(lora_state, strict=False)


def save_lora(model, path):
    raw_model = getattr(model, '_orig_mod', model) # 获取原始模型,因为有些时候模型会被包装，真正的原始模型可能在 _orig_mod 里面。
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            clean_name = name[7:] if name.startswith("module.") else name # 去掉 module. 前缀
            # 提取当前模块的 LoRA 参数
            lora_state = {f'{clean_name}.lora.{k}': v.cpu().half() for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)


def merge_lora(model, lora_path, save_path):
    load_lora(model, lora_path)
    raw_model = getattr(model, '_orig_mod', model)

    # 先复制一份不含 LoRA 参数的权重字典 LoRA 相关参数不会被直接保存。
    state_dict = {
        k: v.detach().cpu().half()
        for k, v in raw_model.state_dict().items()
        if '.lora.' not in k
    }

    with torch.no_grad():
        for name, module in raw_model.named_modules():
            if isinstance(module, nn.Linear) and '.lora.' not in name:
                # 把当前这个线性层自己的原始权重复制出来，作为合并的起点。
                merged_weight = module.weight.detach().float().cpu().clone()
                if hasattr(module, 'lora'):
                    # 如果这层有 LoRA，就加上增量
                    BA_scaled = (
                        module.lora.B.weight.detach().float()
                        @ module.lora.A.weight.detach().float()
                    ) * float(module.lora.scaling)
                    merged_weight += BA_scaled.cpu()

                state_dict[f'{name}.weight'] = merged_weight.half()

    torch.save(state_dict, save_path)
