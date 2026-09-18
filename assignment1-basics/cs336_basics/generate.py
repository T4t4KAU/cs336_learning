"""Autoregressive decoding with temperature and nucleus sampling."""

import argparse

import torch

from cs336_basics.softmax import softmax
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer_lm import TransformerLM


def sample_next_token(logits, temperature=1.0, top_p=1.0):
    if temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("require temperature >= 0 and 0 < top_p <= 1")
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    probs = softmax(logits.float() / temperature, dim=-1)
    sorted_probs, indices = probs.sort(dim=-1, descending=True)
    # Keep the first token that reaches the threshold, as well as all before it.
    if top_p < 1:
        previous_mass = sorted_probs.cumsum(dim=-1) - sorted_probs
        sorted_probs = sorted_probs.masked_fill(previous_mass >= top_p, 0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    sampled = torch.multinomial(sorted_probs, num_samples=1)
    return indices.gather(-1, sampled)


@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens=100, temperature=1.0, top_p=1.0, eos_token_id=None):
    """Generate one sequence, returning prompt plus continuation as a 1-D tensor."""
    if max_new_tokens < 0 or temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("invalid generation length, temperature, or top_p")
    device = next(model.parameters()).device
    tokens = torch.as_tensor(prompt_ids, dtype=torch.long, device=device)
    if tokens.ndim != 1 or tokens.numel() == 0:
        raise ValueError("prompt must be a nonempty one-dimensional sequence")
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            # A sliding context bounds memory and stays within RoPE's cached range.
            context = tokens[-model.context_length :].unsqueeze(0)
            logits = model(context)[:, -1, :]
            next_token = sample_next_token(logits, temperature, top_p).reshape(1)
            tokens = torch.cat((tokens, next_token))
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
    finally:
        model.train(was_training)
    return tokens


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--vocab", required=True)
    p.add_argument("--merges", required=True)
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--eos-token", default="<|endoftext|>")
    p.add_argument("--special-token", action="append", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = TransformerLM(**checkpoint["extra"]["model_config"], device=args.device)
    model.load_state_dict(checkpoint["model"])
    tokenizer = Tokenizer.from_files(args.vocab, args.merges, args.special_token or [args.eos_token])
    if max(tokenizer.vocab) >= model.token_embeddings.num_embeddings:
        raise ValueError("tokenizer vocabulary exceeds model vocabulary")
    eos_id = tokenizer.token_to_id[args.eos_token.encode("utf-8")]
    prompt = tokenizer.encode(args.prompt) or [eos_id]
    result = generate(model, prompt, args.max_new_tokens, args.temperature, args.top_p, eos_id)
    print(tokenizer.decode(result.tolist()))


if __name__ == "__main__":
    main()
