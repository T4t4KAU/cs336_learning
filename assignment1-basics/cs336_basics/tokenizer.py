import json
from collections.abc import Iterable, Iterator
from functools import lru_cache
from os import PathLike

import regex

from .bpe import PRETOKEN_PATTERN, merge_pair


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(dict.fromkeys(special_tokens or []))
        if any(not token for token in self.special_tokens):
            raise ValueError("special tokens must not be empty")
        self.token_to_id = {token: index for index, token in self.vocab.items()}
        next_id = max(self.vocab, default=-1) + 1
        for special in self.special_tokens:
            token = special.encode("utf-8")
            if token not in self.token_to_id:
                self.vocab[next_id] = token
                self.token_to_id[token] = next_id
                next_id += 1

        self.merge_ranks = {}
        for rank, pair in enumerate(self.merges):
            self.merge_ranks.setdefault(pair, rank)
        specials = sorted(self.special_tokens, key=len, reverse=True)
        self._special_pattern = (
            regex.compile("|".join(regex.escape(token) for token in specials)) if specials else None
        )
        self._max_special_length = max(map(len, specials), default=0)
        self._byte_tokens = tuple(bytes([byte]) for byte in range(256))
        self._cached_piece = lru_cache(maxsize=4096)(self._encode_piece)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | PathLike,
        merges_filepath: str | PathLike,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        with open(vocab_filepath, encoding="utf-8") as f:
            vocab = {int(index): bytes.fromhex(token) for index, token in json.load(f).items()}
        with open(merges_filepath, encoding="utf-8") as f:
            merges = [(bytes.fromhex(left), bytes.fromhex(right)) for left, right in json.load(f)]
        return cls(vocab, merges, special_tokens)

    def to_files(self, vocab_filepath: str | PathLike, merges_filepath: str | PathLike) -> None:
        with open(vocab_filepath, "w", encoding="utf-8") as f:
            json.dump({index: token.hex() for index, token in self.vocab.items()}, f)
        with open(merges_filepath, "w", encoding="utf-8") as f:
            json.dump([(left.hex(), right.hex()) for left, right in self.merges], f)

    def _encode_piece(self, piece: str) -> tuple[int, ...]:
        word = [self._byte_tokens[byte] for byte in piece.encode("utf-8")]
        while len(word) > 1:
            best = None
            best_rank = len(self.merges)
            for pair in zip(word, word[1:]):
                rank = self.merge_ranks.get(pair, len(self.merges))
                if rank < best_rank:
                    best, best_rank = pair, rank
            if best is None:
                break
            word = merge_pair(word, best)
        return tuple(self.token_to_id[token] for token in word)

    def _pieces(self, text: str):
        start = 0
        if self._special_pattern is not None:
            for special in self._special_pattern.finditer(text):
                for match in PRETOKEN_PATTERN.finditer(text, start, special.start()):
                    yield match.start(), match.end(), match.group(), False
                yield special.start(), special.end(), special.group(), True
                start = special.end()
        for match in PRETOKEN_PATTERN.finditer(text, start):
            yield match.start(), match.end(), match.group(), False

    def _piece_ids(self, piece: str, special: bool) -> tuple[int, ...]:
        if special:
            return (self.token_to_id[piece.encode("utf-8")],)
        return self._cached_piece(piece)

    def encode(self, text: str) -> list[int]:
        return [
            token_id
            for _, _, piece, special in self._pieces(text)
            for token_id in self._piece_ids(piece, special)
        ]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pending = ""
        for chunk in iterable:
            for offset in range(0, len(chunk), 4096):
                pending += chunk[offset:offset + 4096]
                safe_end = len(pending) - self._max_special_length
                tail = []
                consumed = 0
                for item in self._pieces(pending):
                    tail.append(item)
                    if len(tail) <= 2:
                        continue
                    _, end, piece, special = tail.pop(0)
                    if end > safe_end:
                        break
                    yield from self._piece_ids(piece, special)
                    consumed = end
                pending = pending[consumed:]

        for _, _, piece, special in self._pieces(pending):
            yield from self._piece_ids(piece, special)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[index] for index in ids).decode("utf-8", errors="replace")
