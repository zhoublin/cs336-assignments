import os
import torch
import torch.nn as nn
from einops import rearrange, reduce, einsum

class Linear(nn.Module):
    def __init__(self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super(Linear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        mean = 0
        variance = 2 / (in_features + out_features)
        std = variance ** 0.5
        self.weight = torch.nn.init.trunc_normal_(
            torch.randn((out_features, in_features), device=device, dtype=dtype),
            mean=mean,
            std=std,
            a=-3.0*std,
            b=3.0*std
        )
        self.weight = nn.Parameter(data=self.weight, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.weight = torch.nn.init.trunc_normal_(
            torch.randn(
                (self.num_embeddings, embedding_dim),
                device=device,
                dtype=dtype
            ),
            mean=0,
            std=1,
            a=-3.0,
            b=3.0
        )
        self.weight = nn.Parameter(data=self.weight, requires_grad=True)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super(RMSNorm, self).__init__()

        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(
            torch.ones((d_model, ), device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # @note: torch 广播机制只会在匹配维度的左边自动补1兼容，而右边不会，需要显式补1兼容
        # x / rms 扩展维度利用广播机制在内层维度做逐元素除法
        layer_sum = reduce(x ** 2, "batch_size sequence_length d_model -> batch_size sequence_length", "sum").unsqueeze(-1)
        # g * x 利用广播机制在内层匹配维度做逐元素乘
        result = self.weight * torch.div(x, torch.sqrt(1 / self.d_model * layer_sum + self.eps))
        return result.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self,
        d_model: int,
        d_ff: int | None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None        
    ):
        super(SwiGLU, self).__init__()

        self.d_model = d_model
        self.device = device
        self.dtype = dtype

        self.d_ff = round(8 / 3.0 * d_model / 64) * 64 if d_ff is None else d_ff

        self.w1 = Linear(
            in_features=self.d_model,
            out_features=self.d_ff,
            device=device,
            dtype=dtype
        )
        self.w3 = Linear(
            in_features=self.d_model,
            out_features=self.d_ff,
            device=device,
            dtype=dtype
        )
        self.w2 = Linear(
            in_features=self.d_ff,
            out_features=self.d_model,
            device=device,
            dtype=dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res1 = self.w1(x)
        res2 = self.w3(x)
        silu = res1 * torch.sigmoid(res1)
        glu = silu * res2
        return self.w2(glu)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        r = 1.0 / (
            theta ** (
                torch.arange(0, d_k, 2) /
                d_k
            )
        )
        seq = torch.arange(max_seq_len)
        div = torch.outer(seq, r)

        self.register_buffer(
            "cos", div.cos(), persistent=False
        )
        self.register_buffer(
            "sin", div.sin(), persistent=False
        )

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        if token_positions is not None:
            cos = self.cos[token_positions, :]
            sin = self.sin[token_positions, :]
        else:
            cos = self.cos[:x.shape[-2], :]
            sin = self.sin[:x.shape[-2], :]

        even = x[..., 0::2]
        odd = x[..., 1::2]
        x_even = even * cos - odd * sin
        x_odd = odd * cos + even * sin
        return torch.stack((x_even, x_odd), dim=-1).flatten(-2)

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    normalized_x = x - x.max(dim=dim).values.unsqueeze(dim=dim)
    return torch.exp(normalized_x) / torch.exp(normalized_x).sum(dim=dim).unsqueeze(dim=dim)

def scaled_dot_product_attention(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
    qk = einsum(Q, K, "... n d_k, ... m d_k -> ... n m")
    qk[~mask] = -torch.inf
    qk = qk / torch.sqrt(torch.tensor(Q.shape[-1]))
    return einsum(softmax(qk), V, "... n m, ... m d_v -> ... n d_v")

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,
        d_model: int,
        num_heads: int,
        theta: float = None,
        max_seq_len: int = None,
        device: torch.device | None = None
    ):
        super(MultiHeadSelfAttention, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.d_sd = d_model // num_heads
        self.q_proj = Linear(in_features=d_model, out_features=d_model, device=device)
        self.k_proj = Linear(in_features=d_model, out_features=d_model, device=device)
        self.v_proj = Linear(in_features=d_model, out_features=d_model, device=device)
        self.output_proj = Linear(in_features=d_model, out_features=d_model, device=device)
        self.embed = RotaryPositionalEmbedding(theta, self.d_sd, max_seq_len, device) if theta else None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        Q = rearrange(Q, "... seq (head d_sd) -> ... head seq d_sd", head=self.num_heads, d_sd=self.d_sd)
        K = rearrange(K, "... seq (head d_sd) -> ... head seq d_sd", head=self.num_heads, d_sd=self.d_sd)
        V = rearrange(V, "... seq (head d_sd) -> ... head seq d_sd", head=self.num_heads, d_sd=self.d_sd)
        # RoPE是在解码器每一层的Self-Attention模块内部，对线性变换后的Query($Q$)和Key($K$)向量施加旋转变换，而非在最底层的词向量输入端直接相加
        if self.embed:
            Q = self.embed(Q, token_positions)
            K = self.embed(K, token_positions)
        mask_shape = list(Q.shape)
        mask_shape[-1] = mask_shape[-2]
        # 因果掩码构造对角线以下为True的矩阵，行表示当前正在计算的token，列表示要被关注的token
        mask = torch.tril(torch.ones(mask_shape, dtype=torch.bool))
        output = scaled_dot_product_attention(Q, K, V, mask=mask)
        output = rearrange(output, "... head seq d_sd -> ... seq (head d_sd)", head=self.num_heads, d_sd=self.d_sd)
        return self.output_proj(output)

class TransformerBlock(nn.Module):
    def __init__(self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        eps: float = 0.00001,
        max_seq_len: int = None,
        theta: float | None = None,
    ):
        super(TransformerBlock, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        
        self.ln1 = RMSNorm(d_model, eps)
        self.ln2 = RMSNorm(d_model, eps)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, theta, max_seq_len)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor, positions: torch.Tensor = None) -> torch.Tensor:
        out = x + self.attn(self.ln1(x), positions)
        return out + self.ffn(self.ln2(out))

class TranformerLM(nn.Module):
    def __init__(self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super(TranformerLM, self).__init__()
        
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers

        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta
            ) for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model=d_model)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.token_embeddings(x)
        for block in self.layers:
            out = block(out)
        return self.lm_head(self.ln_final(out))

def cal_params_and_flops_of_model(
    vocab_size = 50257,
    context_length = 1024,
    num_layers = 48,
    d_model = 1600,
    num_heads = 25,
):
    d_ff = round((8 / 3 * d_model) / 64) * 64

    param_num = 0
    # embedding
    param_num += vocab_size * d_model
    # transformer block
    param_num += num_layers * sum([
        # RMSNorm prenorm 1
        d_model,
        # Attention, Wq, Wk, Wv, Wo
        d_model ** 2 * 4,
        # RMSNorm prenorm 2
        d_model,
        # Position-wise FFN (W1, W2, W3)
        d_ff * d_model * 3
    ])
    # last prenorm
    param_num += d_model
    # LM head
    param_num += d_model * vocab_size

    print(f"Breakdown #param:")
    print(f"- embedding: {vocab_size * d_model}\n  ↓")
    print(f"- RMSNorm prenorm: {2 * d_model} * {num_layers}\n  ↓")
    print(f"- Attention projection matrix: {d_model ** 2 * 4} * {num_layers}\n  ↓")
    print(f"- Position-wise FFN: {d_ff * d_model * 3} * {num_layers}\n  ↓")
    print(f"- Last RMSNorm prenorm: {d_model}\n  ↓")
    print(f"- LM head: {d_model * vocab_size}")

    flops = 0
    flops += num_layers * sum([
        # Attention Projection
        2 * context_length * (d_model ** 2) * 4,
        # Attention Softmax(QK^T/sqrt(d_k))@V
        2 * (context_length ** 2) * d_model * 2,
        # Position-wise FFN
        2 * d_ff * d_model * context_length * 3
    ])
    # LM head
    flops += 2 * d_model * vocab_size * context_length

    print(f"Breakdown #Flops:")
    print(f"- Attention projection: {2 * context_length * (d_model ** 2) * 4} * {num_layers}\n  ↓")
    print(f"- Softmax Attention: {2 * (context_length ** 2) * d_model * 2} * {num_layers}\n  ↓")
    print(f"- Position-wise FFN: {2 * d_ff * d_model * context_length * 3} * {num_layers}\n  ↓")
    print(f"- LM head: {2 * d_model * vocab_size * context_length}")

    print(f"\nModel learnable parameter num: {param_num}, Flops: {flops}")

if __name__ == "__main__":
    print("\nGPT-2 small:")
    cal_params_and_flops_of_model(
        vocab_size = 50257,
        context_length = 1024,
        num_layers = 12,
        d_model = 768,
        num_heads = 12,
    )

    print("\nGPT-2 medium:")
    cal_params_and_flops_of_model(
        vocab_size = 50257,
        context_length = 1024,
        num_layers = 24,
        d_model = 1024,
        num_heads = 16,
    )

    print("\nGPT-2 Large:")
    cal_params_and_flops_of_model(
        vocab_size = 50257,
        context_length = 1024,
        num_layers = 36,
        d_model = 1280,
        num_heads = 20,
    )

    print("\nGPT-2 XL-sized:")
    cal_params_and_flops_of_model(
        vocab_size = 50257,
        context_length = 1024,
        num_layers = 48,
        d_model = 1600,
        num_heads = 25,
    )

    print("\nGPT-2 XL-sized with larger context length:")
    cal_params_and_flops_of_model(
        vocab_size = 50257,
        context_length = 16384,
        num_layers = 48,
        d_model = 1600,
        num_heads = 25,
    )
