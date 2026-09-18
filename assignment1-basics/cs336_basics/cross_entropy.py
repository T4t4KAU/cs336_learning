import torch

def cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:

    # 低精度用 float32
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()

    # 每个位置独立减去词表分数中的最大值
    shifted = logits - logits.amax(dim=-1, keepdim=True)

    log_partition = shifted.exp().sum(dim=-1).log()

    # 取出每个位置上，正确 token 对应的分数。
    target_logits = shifted.gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    losses = log_partition - target_logits

    return losses.mean()
