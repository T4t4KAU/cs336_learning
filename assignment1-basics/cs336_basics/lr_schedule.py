import math


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < 0:
        raise ValueError("it must be nonnegative")
    if not 0 <= warmup_iters < cosine_cycle_iters:
        raise ValueError("require 0 <= warmup_iters < cosine_cycle_iters")
    if not 0 <= min_learning_rate <= max_learning_rate:
        raise ValueError("require 0 <= min_learning_rate <= max_learning_rate")

    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters

    if it >= cosine_cycle_iters:
        return min_learning_rate

    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    coefficient = 0.5 * (1 + math.cos(math.pi * progress))
    return min_learning_rate + coefficient * (max_learning_rate - min_learning_rate)
