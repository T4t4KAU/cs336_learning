"""Memory-mapped token data and random next-token batches."""

import numpy as np
import torch


def load_tokens(path, dtype="uint16"):
    data = np.load(path, mmap_mode="r") if str(path).endswith(".npy") else np.memmap(path, mode="r", dtype=dtype)
    if data.ndim != 1 or not np.issubdtype(data.dtype, np.integer):
        raise ValueError("token data must be a one-dimensional integer array")
    return data


def get_batch(dataset, batch_size, context_length, device, *, rng=None):
    if dataset.ndim != 1 or not np.issubdtype(dataset.dtype, np.integer):
        raise ValueError("dataset must be a one-dimensional integer array")
    if batch_size <= 0 or context_length <= 0 or len(dataset) <= context_length:
        raise ValueError("positive batch/context sizes and at least context_length + 1 tokens are required")
    high = len(dataset) - context_length
    starts = np.random.randint(high, size=batch_size) if rng is None else rng.integers(high, size=batch_size)
    # Only materialize the sampled windows, including their next-token targets.
    windows = np.stack([dataset[i : i + context_length + 1] for i in starts]).astype(np.int64)
    return (torch.from_numpy(windows[:, :-1].copy()).to(device), torch.from_numpy(windows[:, 1:].copy()).to(device))
