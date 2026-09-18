import torch
from torch import nn

from cs336_basics.linear import Linear

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device=None,
        dtype: torch.dtype=None,
    ):
        super().__init__()

        if d_ff is None:
            d_ff = ((8 * d_model + 191) // 192) * 64

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = silu(self.w1(x))
        features = self.w3(x)

        return self.w2(gate * features)


class SiLUFeedForward(nn.Module):
    """Ungated FFN for the assignment's SwiGLU ablation."""

    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x):
        return self.w2(silu(self.w1(x)))
