import torch

from torch import nn

from cs336_basics.rms_norm import RMSNorm
from cs336_basics.swiglu import SwiGLU, SiLUFeedForward
from cs336_basics.rope import RotrayPositionalEmbdding
from cs336_basics.multihead_attention import MultiHeadSelfAttention

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        eps: float = 1e-5,
        device: torch.device=None,
        dtype=None,
        norm_style: str = "pre",
        use_rope: bool = True,
        ffn_type: str = "swiglu",
    ):
        super().__init__()
        if norm_style not in {"pre", "post", "none"}:
            raise ValueError("norm_style must be pre, post, or none")
        if ffn_type not in {"swiglu", "silu"}:
            raise ValueError("ffn_type must be swiglu or silu")
        self.norm_style = norm_style

        if num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError

        self.ln1 = RMSNorm(
            d_model, eps=eps, device=device, dtype=dtype
        )

        self.ln2 = RMSNorm(
            d_model, eps=eps, device=device, dtype=dtype
        )

        if norm_style == "none":
            self.ln1 = nn.Identity()
            self.ln2 = nn.Identity()

        rope = RotrayPositionalEmbdding(
            theta=theta,
            d_k=d_model // num_heads,
            max_seq_len=max_seq_len,
            device=device
        ) if use_rope else None

        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            rope=rope,
            device=device,
            dtype=dtype,
        )

        ffn_cls = SwiGLU if ffn_type == "swiglu" else SiLUFeedForward
        self.ffn = ffn_cls(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.norm_style == "post":
            x = self.ln1(x + self.attn(x, token_positions))
            return self.ln2(x + self.ffn(x))
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))

        return x
