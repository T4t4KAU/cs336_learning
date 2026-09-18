import math
import torch

from cs336_basics.softmax import softmax

def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    # q: (..., n_queries, d_k)
    # k: (..., n_keys, d_k)
    # v: (..., n_keys, d_v)

    d_k = q.shape[-1]

    # query 和 key 做点积
    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    weights = softmax(scores, dim=-1)

    return weights @ v
