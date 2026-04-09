import math  # 导入数学库；本文件当前没有直接使用它。

import torch  # 导入 PyTorch 主库。

import torch.nn as nn  # 导入神经网络模块。

import torch.nn.functional as F  # 导入函数式接口。


class ModelArgs:  # 用来集中保存模型超参数。
    dim: int = 768  # 隐藏层维度，也就是每个 token 向量的长度。
    n_layers: int = 16  # Transformer Block 的层数。
    n_heads: int = 12  # 多头注意力的头数。
    vocab_size: int = 6400  # 词表大小。
    hidden_dim: int = 2048  # 前馈网络中间层维度。
    norm_eps: float = 1e-5  # RMSNorm 的稳定项。
    max_seq_len: int = 512  # 模型支持的最大序列长度。


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):  # 预计算 RoPE 的复数旋转表。
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))  # 生成每对通道对应的频率。
    t = torch.arange(end, device=freqs.device)  # 生成位置索引 0 到 end-1。
    freqs = torch.outer(t, freqs).float()  # 用外积得到每个位置上的旋转角度。
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # 变成 cos + i*sin 的复数形式。
    return freqs_cis  # 返回 RoPE 旋转因子。


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):  # 把频率表变成可广播的形状。
    ndim = x.ndim  # 取出输入张量的维度数。
    assert 0 <= 1 < ndim  # 检查序列维索引是合法的。
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])  # 检查频率表与序列长度和最后一维匹配。
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]  # 只保留序列维和最后一维，其余维度设为 1。
    return freqs_cis.view(*shape)  # 返回广播后形状的张量。


def apply_rotary_emb(  # 将 RoPE 应用到 Q 和 K 上。
        xq: torch.Tensor,  # 查询张量 Q。
        xk: torch.Tensor,  # 键张量 K。
        freqs_cis: torch.Tensor,  # 当前序列对应的旋转因子。
):  # 多行函数定义到这里结束。
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))  # 把 Q 的最后一维按两个一组转成复数。
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))  # 把 K 也转成复数。
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)  # 调整旋转因子的形状以便广播。
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)  # 对 Q 做复数乘法，相当于执行旋转。
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)  # 对 K 做同样的旋转。
    return xq_out.type_as(xq), xk_out.type_as(xk)  # 转回原数据类型并返回。


class RMSNorm(nn.Module):  # 定义 RMSNorm 归一化层。

    def __init__(self, dim: int, eps: float = 1e-6):  # 初始化维度和稳定项。
        super().__init__()  # 调用父类初始化。
        self.eps = eps  # 保存 eps。
        self.weight = nn.Parameter(torch.ones(dim))  # 定义可学习缩放参数。

    def _norm(self, x):  # 实现 RMS 归一化公式。
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)  # 按最后一维做均方根归一化。

    def forward(self, x):  # 定义前向传播。
        output = self._norm(x.float()).type_as(x)  # 先转 float 计算，再转回原精度。
        return output * self.weight  # 乘上可学习权重后输出。


class FeedForward(nn.Module):  # 定义前馈网络模块。

    def __init__(self, args: ModelArgs):  # 根据模型配置初始化。
        super().__init__()  # 调用父类初始化。
        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)  # 第一条投影分支。
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)  # 输出投影分支。
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)  # 门控投影分支。

    def forward(self, x):  # 定义前馈网络的前向传播。
        return self.w2(F.silu(self.w1(x)) * self.w3(x))  # SwiGLU：激活分支与门控分支相乘后再投影回来。


class Attention(nn.Module):  # 定义多头自注意力模块。

    def __init__(self, args: ModelArgs):  # 初始化头数、头维度和线性层。
        super().__init__()  # 调用父类初始化。
        self.n_heads = args.n_heads  # 保存注意力头数。
        self.head_dim = args.dim // args.n_heads  # 计算每个头的维度。
        self.wq = nn.Linear(args.dim, args.dim, bias=False)  # Q 的线性映射。
        self.wk = nn.Linear(args.dim, args.dim, bias=False)  # K 的线性映射。
        self.wv = nn.Linear(args.dim, args.dim, bias=False)  # V 的线性映射。
        self.wo = nn.Linear(args.dim, args.dim, bias=False)  # 注意力输出的线性映射。

    def forward(self, x, freqs_cis):  # x 是隐藏状态，freqs_cis 是 RoPE 频率表。
        B, Seq_Len, Dim = x.shape  # 取出 batch、序列长度和隐藏维度。
        q = self.wq(x)  # 得到查询向量 Q。
        k = self.wk(x)  # 得到键向量 K。
        v = self.wv(x)  # 得到值向量 V。

        q = q.view(B, Seq_Len, self.n_heads, self.head_dim)  # 把 Q reshape 成多头格式。
        k = k.view(B, Seq_Len, self.n_heads, self.head_dim)  # 把 K reshape 成多头格式。
        v = v.view(B, Seq_Len, self.n_heads, self.head_dim)  # 把 V reshape 成多头格式。
        q, k = apply_rotary_emb(q, k, freqs_cis)  # 对 Q、K 注入 RoPE。

        q = q.transpose(1, 2)  # 调整 Q 维度顺序为 [B, n_heads, Seq_Len, head_dim]。
        k = k.transpose(1, 2)  # 调整 K 维度顺序。
        v = v.transpose(1, 2)  # 调整 V 维度顺序。

        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # 计算因果自注意力。
        output = output.transpose(1, 2).contiguous().view(B, Seq_Len, Dim)  # 把多头结果拼回原始维度。

        return self.wo(output)  # 通过输出投影得到最终注意力结果。


class TransformerBlock(nn.Module):  # 定义一个完整的 Transformer Block。
    def __init__(self, args: ModelArgs):  # 初始化内部子模块。
        super().__init__()  # 调用父类初始化。
        self.attention = Attention(args)  # 创建注意力模块。
        self.feed_forward = FeedForward(args)  # 创建前馈网络模块。
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)  # 注意力分支前的归一化。
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)  # 前馈分支前的归一化。

    def forward(self, x, freqs_cis):  # 定义 Block 的前向传播。
        h = x + self.attention(self.attention_norm(x), freqs_cis)  # Pre-Norm 注意力加残差。
        out = h + self.feed_forward(self.ffn_norm(h))  # Pre-Norm 前馈网络加残差。
        return out  # 返回本层输出。


class PocketLLM(nn.Module):  # 定义完整的语言模型。
    def __init__(self, args: ModelArgs):  # 根据配置组装模型。
        super().__init__()  # 调用父类初始化。
        self.args = args  # 保存配置对象。
        self.vocab_size = args.vocab_size  # 保存词表大小。
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)  # 创建 token embedding 层。
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])  # 堆叠多个 Transformer Block。
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)  # 定义最终归一化层。
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)  # 定义输出投影层。
        freqs_cis = precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len)  # 预计算最大长度下的 RoPE 表。
        self.register_buffer("freqs_cis", freqs_cis)  # 把 RoPE 表注册成 buffer。

    def forward(self, tokens):  # 定义模型整体前向传播。
        B, Seq_Len = tokens.shape  # 取出 batch 大小和序列长度。
        h = self.tok_embeddings(tokens)  # 把 token id 映射成向量。
        freqs_cis = self.freqs_cis[:Seq_Len]  # 截取当前序列长度需要的 RoPE。

        for layer in self.layers:  # 依次通过每一个 Transformer Block。
            h = layer(h, freqs_cis)  # 更新隐藏状态。

        h = self.norm(h)  # 最后一层归一化。
        logits = self.output(h)  # 映射到词表维度得到 logits。

        return logits  # 返回预测分数。


if __name__ == "__main__":  # 直接运行文件时执行下面的测试代码。
    args = ModelArgs()  # 创建默认配置。
    model = PocketLLM(args)  # 实例化模型。
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 统计可训练参数量。
    print(f"模型总参数量: {total_params / 1e6:.2f} M (百万)")  # 打印参数量。
    dummy_input = torch.randint(0, args.vocab_size, (2, 10))  # 构造一个 batch=2、长度=10 的假输入。
    logits = model(dummy_input)  # 执行一次前向传播。
    print(f"输入形状: {dummy_input.shape}")  # 打印输入形状。
    print(f"输出 Logits 形状: {logits.shape}")  # 打印输出形状。
