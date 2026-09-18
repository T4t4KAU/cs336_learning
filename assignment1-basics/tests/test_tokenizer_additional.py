import random

import pytest

from cs336_basics.bpe import train_bpe_v3
from cs336_basics.tokenizer import Tokenizer


@pytest.fixture
def tokenizer():
    text = "hello hello cats can't we'll I'd 你好🙂 \n\n <e>tail hello!"
    vocab, merges = train_bpe_v3(text, 290, ["<e>", "<e>tail"])
    return Tokenizer(vocab, merges, ["<e>", "<e>tail"])


def test_stream_all_split_positions(tokenizer):
    text = "hello cats can't we'll I'd你好🙂  \n\n<e>tailhello<e>!  "
    expected = tokenizer.encode(text)
    for position in range(len(text) + 1):
        assert list(tokenizer.encode_iterable([text[:position], "", text[position:]])) == expected
    assert list(tokenizer.encode_iterable(iter(text))) == expected


def test_stream_random_chunks(tokenizer):
    rng = random.Random(336)
    for _ in range(50):
        text = "".join(rng.choices(["hello", "'", "ll", " ", "\n", "\t", "🙂", "<e>", "<e>tail"], k=40))
        chunks = []
        start = 0
        while start < len(text):
            end = start + rng.randrange(1, 12)
            chunks.append(text[start:end])
            start = end
        assert list(tokenizer.encode_iterable(chunks)) == tokenizer.encode(text)


def test_stream_internal_block_boundary(tokenizer):
    text = "hello " * 682 + "<e>tail你好 " + "x" * 4200 + "  \nhello"
    assert list(tokenizer.encode_iterable([text])) == tokenizer.encode(text)


def test_stream_is_lazy(tokenizer):
    def chunks():
        yield "hello cats hello cats hello cats "
        raise AssertionError("must yield before consuming the next chunk")

    stream = tokenizer.encode_iterable(chunks())
    assert next(stream) == tokenizer.encode("hello")[0]
    stream.close()


def test_file_roundtrip(tokenizer, tmp_path):
    vocab_path = tmp_path / "vocab.json"
    merges_path = tmp_path / "merges.json"
    tokenizer.to_files(vocab_path, merges_path)
    loaded = Tokenizer.from_files(vocab_path, merges_path, tokenizer.special_tokens)
    assert loaded.vocab == tokenizer.vocab
    assert loaded.merges == tokenizer.merges
    text = "你好🙂 hello<e>tail"
    assert loaded.encode(text) == tokenizer.encode(text)
    assert loaded.decode(loaded.encode(text)) == text


def test_missing_special_sparse_ids_and_no_input_mutation():
    vocab = {0: b"a", 8: b"b"}
    tokenizer = Tokenizer(vocab, [], ["<e>", "<e>"])
    assert tokenizer.encode("a<e>b") == [0, 9, 8]
    assert vocab == {0: b"a", 8: b"b"}


def test_decode_joins_bytes_before_utf8():
    tokenizer = Tokenizer({i: bytes([i]) for i in range(256)}, [])
    assert tokenizer.decode(list("你".encode())) == "你"
    assert tokenizer.decode([255, 97]) == "\ufffda"


def test_document_example():
    tokens = [b" ", b"a", b"c", b"e", b"h", b"t", b"th", b" c", b" a", b"the", b" at"]
    merges = [(b"t", b"h"), (b" ", b"c"), (b" ", b"a"), (b"th", b"e"), (b" a", b"t")]
    tokenizer = Tokenizer(dict(enumerate(tokens)), merges)
    assert tokenizer.encode("the cat ate") == [9, 7, 1, 5, 10, 3]
