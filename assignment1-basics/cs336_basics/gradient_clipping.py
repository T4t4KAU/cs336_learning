from collections.abc import Iterable

import torch


@torch.no_grad()
def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6,
) -> None:
    """Clip dense gradients on one device by their combined L2 norm."""
    if max_l2_norm < 0:
        raise ValueError("max_l2_norm must be nonnegative")
    if eps <= 0:
        raise ValueError("eps must be positive")

    # 保存引用，支持只能遍历一次的 model.parameters() 迭代器。
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return

    # 低精度梯度使用 float32 累加，float64 梯度保留原精度。
    squared_norms = []
    for grad in grads:
        values = grad.float() if grad.dtype in (torch.float16, torch.bfloat16) else grad
        squared_norms.append(values.square().sum())
    total_norm = torch.stack(squared_norms).sum().sqrt()

    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for grad in grads:
            grad.mul_(scale)
