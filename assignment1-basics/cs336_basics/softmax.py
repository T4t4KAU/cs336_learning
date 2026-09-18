import torch

def softmax(x: torch.Tensor, dim: int=-1) -> torch.Tensor:
    max_value = x.amax(dim=dim, keepdim=True)
    exp_values = torch.exp(x - max_value)

    return exp_values / exp_values.sum(dim=dim, keepdim=True)
