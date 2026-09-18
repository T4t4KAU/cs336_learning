import torch
from torch import nn

from cs336_basics.embdding import Embedding
from cs336_basics.linear import Linear
from cs336_basics.rms_norm import RMSNorm
from cs336_basics.transformer_block import TransformerBlock

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device=None,
        dtype: torch.dtype=None,
        norm_style: str = "pre",
        use_rope: bool = True,
        ffn_type: str = "swiglu",
    ):
        super().__init__()

        self.context_length = context_length
        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                device=device,
                dtype=dtype,
                norm_style=norm_style,
                use_rope=use_rope,
                ffn_type=ffn_type,
            )

            for _ in range(num_layers)
        ])

        self.ln_final = RMSNorm(
            d_model=d_model,
            eps=1e-5,
            device=device,
            dtype=dtype,
        )

        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
        )
        if norm_style == "none":
            self.ln_final = nn.Identity()

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.shape[-1] > self.context_length:
            raise ValueError

        x = self.token_embeddings(token_ids)

        for block in self.layers:
            x = block(x)

        x = self.ln_final(x)

        return self.lm_head(x)
