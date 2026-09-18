"""Model/optimizer checkpoints supporting paths and binary streams."""

import os
from pathlib import Path
import tempfile

import torch


def save_checkpoint(model, optimizer, iteration, out, *, extra=None):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "extra": extra or {},
    }
    if not isinstance(out, (str, os.PathLike)):
        torch.save(payload, out)
        return
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Replace only once serialization succeeds, preserving the previous checkpoint.
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["iteration"])
