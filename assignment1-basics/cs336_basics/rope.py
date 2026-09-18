import torch
from torch import nn

class RotrayPositionalEmbdding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device = None,
    ):
        super().__init__()

        # 检查偶数
        if d_k % 2 != 0:
            raise ValueError

        self.d_k = d_k

        pair_indices = torch.arange(
            0, d_k, 2, device=device, dtype=torch.float32
        )

        inv_freq = theta ** (-pair_indices / d_k)

        positions = torch.arange(
            max_seq_len, device=device, dtype=torch.float32
        )

        angles = positions[:, None] * inv_freq[None, :]
        self.register_buffer(
            "cos_cache", angles.cos(), persistent=False
        )

        self.register_buffer(
            "sin_cache", angles.sin(), persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        input_dtype = x.dtype
        positions = token_positions.to(
            device=x.device, dtype=torch.long
        )

        cos = self.cos_cache[positions]
        sin = self.sin_cache[positions]

        x = x.to(torch.float32)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        output = torch.stack(
            (out_even, out_odd), dim=-1
        ).flatten(-2)

        return output.to(input_dtype)
