"""Stream UTF-8 text through the tokenizer into a memory-mappable token file."""

import argparse
from itertools import islice
import json
from pathlib import Path
import time

import numpy as np

from cs336_basics.tokenizer import Tokenizer


def encode_file(input_path, output_path, tokenizer, dtype="uint16"):
    output = Path(output_path)
    input_path = Path(input_path)
    if output.suffix == ".npy":
        raise ValueError("this command writes raw tokens; use a .bin extension")
    if input_path.resolve() == output.resolve():
        raise ValueError("input and output paths must differ")
    dtype = np.dtype(dtype)
    if max(tokenizer.vocab) > np.iinfo(dtype).max:
        raise ValueError("token IDs exceed output dtype; use uint32")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    count = 0
    with input_path.open(encoding="utf-8") as src, output.open("xb") as dst:
        chunks = iter(lambda: src.read(65536), "")
        ids = tokenizer.encode_iterable(chunks)
        while batch := list(islice(ids, 65536)):
            np.asarray(batch, dtype=dtype).tofile(dst)
            count += len(batch)
    size = input_path.stat().st_size
    elapsed = time.perf_counter() - started
    metadata = {
        "tokens": count,
        "dtype": dtype.name,
        "source_bytes": size,
        "bytes_per_token": size / count if count else None,
        "elapsed_seconds": elapsed,
        "bytes_per_second": size / elapsed,
        "vocab_size": max(tokenizer.vocab) + 1,
    }
    Path(str(output) + ".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--vocab", required=True)
    p.add_argument("--merges", required=True)
    p.add_argument("--dtype", choices=["uint16", "uint32", "int64"], default="uint16")
    p.add_argument("--special-token", action="append", default=None)
    args = p.parse_args()
    tokenizer = Tokenizer.from_files(args.vocab, args.merges, args.special_token or ["<|endoftext|>"])
    print(json.dumps(encode_file(args.input, args.output, tokenizer, args.dtype)))


if __name__ == "__main__":
    main()
