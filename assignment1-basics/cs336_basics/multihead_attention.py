import torch
from torch import nn

from cs336_basics.linear import Linear
from cs336_basics.attention import scaled_dot_product_attention

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: nn.Module=None,
        device=None,
        dtype=None,
    ):
        super().__init__()

        if num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError

        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.rope = rope

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(
            *x.shape[:-1], self.num_heads, self.d_k
        )

        return x.transpose(-3, -2)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor=None,
    ) -> torch.Tensor:
        seq_len = x.shape[-2]

        # 投影并拆分多头
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(
                    seq_len, device=x.device
                )

            positions = token_positions.unsqueeze(-2)
            q = self.rope(q, positions)
            k = self.rope(k, positions)

        mask = torch.ones(
            seq_len,
            seq_len,
            device=x.device,
            dtype=torch.bool
        ).tril()

        # 每个头独立计算注意力
        output = scaled_dot_product_attention(q, k, v, mask)

        # 合并头
        output = output.transpose(-3, -2)

        output = output.reshape(*x.shape)

        return self.o_proj(output)
