"""Configurable training and evaluation: python -m cs336_basics.train --help."""

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from cs336_basics.admaw import AdamW
from cs336_basics.checkpoint import save_checkpoint
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.data import get_batch, load_tokens
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.lr_schedule import get_lr_cosine_schedule
from cs336_basics.transformer_lm import TransformerLM


@torch.no_grad()
def evaluate(model, dataset, batch_size, context_length, device, batches=20, seed=0):
    """Use an independent RNG so validation cannot alter training batches."""
    was_training = model.training
    model.eval()
    rng = np.random.default_rng(seed)
    losses = []
    try:
        for _ in range(batches):
            x, y = get_batch(dataset, batch_size, context_length, device, rng=rng)
            losses.append(cross_entropy(model(x), y).item())
    finally:
        model.train(was_training)
    loss = sum(losses) / len(losses)
    return {"val_loss": loss, "perplexity": math.exp(loss) if loss < 709 else float("inf")}


def validate_tokens(data, vocab_size, context_length):
    if len(data) <= context_length:
        raise ValueError("dataset must contain more tokens than context_length")
    # Bound temporary memory even when scanning a large mmap.
    for offset in range(0, len(data), 1_000_000):
        chunk = data[offset : offset + 1_000_000]
        if chunk.min() < 0 or chunk.max() >= vocab_size:
            raise ValueError("dataset contains token IDs outside the configured vocabulary")


def train(config):
    c = dict(config)
    for key in (
        "steps",
        "batch_size",
        "context_length",
        "eval_batches",
        "log_every",
        "eval_every",
        "save_every",
        "threads",
    ):
        if c[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if c["max_grad_norm"] <= 0:
        raise ValueError("max_grad_norm must be positive")
    schedule_steps = c["schedule_steps"] or c["steps"]
    get_lr_cosine_schedule(0, c["lr"], c["min_lr"], c["warmup_steps"], schedule_steps)
    torch.set_num_threads(c["threads"])
    torch.manual_seed(c["seed"])
    rng = np.random.default_rng(c["seed"])
    device = c["device"]
    model_config = {
        key: c[key]
        for key in (
            "vocab_size",
            "context_length",
            "d_model",
            "num_layers",
            "num_heads",
            "rope_theta",
            "norm_style",
            "ffn_type",
        )
    }
    model_config["use_rope"] = not c["no_rope"]
    model_config["d_ff"] = c["d_ff"] or (
        4 * c["d_model"] if c["ffn_type"] == "silu" else ((8 * c["d_model"] + 191) // 192) * 64
    )
    train_data = load_tokens(c["train_data"], c["data_dtype"])
    val_data = load_tokens(c["val_data"], c["data_dtype"])
    for data in (train_data, val_data):
        validate_tokens(data, c["vocab_size"], c["context_length"])
    model = TransformerLM(**model_config, device=device)
    optimizer = AdamW(
        model.parameters(), lr=c["lr"], betas=(c["beta1"], c["beta2"]), eps=c["eps"], weight_decay=c["weight_decay"]
    )
    start = 0
    previous_elapsed = 0.0
    if c["resume"]:
        checkpoint = torch.load(c["resume"], map_location="cpu", weights_only=True)
        extra = checkpoint["extra"]
        if extra["model_config"] != model_config:
            raise ValueError("resume model configuration does not match checkpoint")
        # Prevent a silently changed schedule or data stream on resume.
        old = extra["train_config"]
        for key in (
            "train_data",
            "val_data",
            "data_dtype",
            "batch_size",
            "lr",
            "min_lr",
            "warmup_steps",
            "beta1",
            "beta2",
            "eps",
            "weight_decay",
            "max_grad_norm",
            "seed",
        ):
            if c[key] != old[key]:
                raise ValueError(f"resume configuration differs: {key}")
        if schedule_steps != (old["schedule_steps"] or old["steps"]):
            raise ValueError("resume must preserve schedule_steps")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start = int(checkpoint["iteration"])
        rng.bit_generator.state = extra["numpy_rng"]
        torch.set_rng_state(extra["torch_rng"])
        if torch.cuda.is_available() and extra["cuda_rng"]:
            torch.cuda.set_rng_state_all(extra["cuda_rng"])
        previous_elapsed = extra["elapsed_seconds"]
    if start >= c["steps"] and not c["eval_only"]:
        raise ValueError("steps must exceed the checkpoint's completed iteration count")
    if c["eval_only"]:
        if not c["resume"]:
            raise ValueError("--eval-only requires --resume")
        result = evaluate(model, val_data, c["batch_size"], c["context_length"], device, c["eval_batches"], c["seed"])
        print(json.dumps(result))
        return result
    out = Path(c["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if not c["resume"] and any(out.iterdir()):
        raise ValueError("out_dir is not empty; select a new directory or use --resume")
    (out / "config.json").write_text(json.dumps(c, indent=2), encoding="utf-8")
    began = time.perf_counter()

    def elapsed():
        return previous_elapsed + time.perf_counter() - began

    def save(iteration, name):
        save_checkpoint(
            model,
            optimizer,
            iteration,
            out / name,
            extra={
                "model_config": model_config,
                "train_config": c,
                "numpy_rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "elapsed_seconds": elapsed(),
            },
        )

    with (out / "metrics.jsonl").open("a", encoding="utf-8") as log:

        def record(values):
            row = {"elapsed_seconds": elapsed(), **values}
            line = json.dumps(row)
            log.write(line + "\n")
            log.flush()
            print(line, flush=True)

        record(
            {
                "step": start,
                **evaluate(model, val_data, c["batch_size"], c["context_length"], device, c["eval_batches"], c["seed"]),
            }
        )
        model.train()
        for step in range(start, c["steps"]):
            lr = get_lr_cosine_schedule(step, c["lr"], c["min_lr"], c["warmup_steps"], schedule_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            x, y = get_batch(train_data, c["batch_size"], c["context_length"], device, rng=rng)
            optimizer.zero_grad(set_to_none=True)
            loss = cross_entropy(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at step {step}")
            loss.backward()
            gradient_clipping(model.parameters(), c["max_grad_norm"])
            optimizer.step()
            completed = step + 1
            if completed % c["log_every"] == 0 or completed == c["steps"]:
                record(
                    {
                        "step": completed,
                        "train_loss": loss.item(),
                        "lr": lr,
                        "tokens_seen": completed * c["batch_size"] * c["context_length"],
                    }
                )
            if completed % c["eval_every"] == 0 or completed == c["steps"]:
                record(
                    {
                        "step": completed,
                        **evaluate(
                            model, val_data, c["batch_size"], c["context_length"], device, c["eval_batches"], c["seed"]
                        ),
                    }
                )
            if completed % c["save_every"] == 0 or completed == c["steps"]:
                save(completed, "checkpoint.pt")
        return {"checkpoint": str(out / "checkpoint.pt"), "steps": c["steps"]}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-data", required=True)
    p.add_argument("--val-data", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--resume")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--data-dtype", choices=["uint16", "uint32", "int64"], default="uint16")
    p.add_argument("--vocab-size", type=int, required=True)
    for flag, default in (
        ("context-length", 256),
        ("d-model", 512),
        ("num-layers", 4),
        ("num-heads", 16),
        ("batch-size", 32),
        ("steps", 1000),
        ("warmup-steps", 100),
        ("eval-batches", 20),
        ("log-every", 10),
        ("eval-every", 100),
        ("save-every", 100),
        ("seed", 42),
        ("threads", 4),
    ):
        p.add_argument("--" + flag, type=int, default=default)
    p.add_argument("--d-ff", type=int)
    p.add_argument("--schedule-steps", type=int, help="Total schedule horizon, preserved on resume")
    for flag, default in (
        ("rope-theta", 10000.0),
        ("lr", 3e-4),
        ("min-lr", 3e-5),
        ("beta1", 0.9),
        ("beta2", 0.999),
        ("eps", 1e-8),
        ("weight-decay", 0.1),
        ("max-grad-norm", 1.0),
    ):
        p.add_argument("--" + flag, type=float, default=default)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--norm-style", choices=["pre", "post", "none"], default="pre")
    p.add_argument("--ffn-type", choices=["swiglu", "silu"], default="swiglu")
    p.add_argument("--no-rope", action="store_true")
    return p


if __name__ == "__main__":
    train(vars(parser().parse_args()))
