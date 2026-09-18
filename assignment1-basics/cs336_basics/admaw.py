import math
import torch

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float=1e-8,
        weight_decay: float=0.01
    ):
        if lr < 0:
            raise ValueError

        if eps <= 0:
            raise ValueError

        if weight_decay < 0:
            raise ValueError

        if not all(0 <= beta < 1 for beta in betas):
            raise ValueError

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss=None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]
                m = state["m"]
                v = state["v"]

                p.mul_(1 - lr * weight_decay)
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(
                    grad, grad, value=1 - beta2
                )

                step_size = (
                    lr * math.sqrt(1 - beta2 ** t)
                    / (1 - beta1 ** t)
                )

                denominator = v.sqrt() + eps
                p.addcdiv_(m, denominator, value=-step_size)

        return loss
