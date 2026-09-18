import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float,
        device: torch.device=None,
        dtype: torch.dtype=None,
    ):
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32) # 降低低精度平方溢出风险

        mean_square = x.square().mean(dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(mean_square + self.eps)

        norm = x * inv_rms
        output = norm * self.weight.to(torch.float32)

        return output.to(input_dtype)
