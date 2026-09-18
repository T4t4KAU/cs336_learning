"""Pretokenization keeps special tokens intact and preserves the original text."""

import pytest

from cs336_basics.bpe import pretokenize


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("", [], id="empty"),
        pytest.param("Hello world!", ["Hello", " world", "!"], id="words"),
        pytest.param(
            "I'm we've he'll she'd can't",
            ["I", "'m", " we", "'ve", " he", "'ll", " she", "'d", " can", "'t"],
            id="contractions",
        ),
        pytest.param("abc123 45.6", ["abc", "123", " 45", ".", "6"], id="numbers"),
        pytest.param("你好 世界 café", ["你好", " 世界", " café"], id="unicode-letters"),
        pytest.param("１２３٤٥", ["１２３٤٥"], id="unicode-numbers"),
        pytest.param("Hi🙂🚀!", ["Hi", "🙂🚀!"], id="emoji"),
        pytest.param("a  b", ["a", " ", " b"], id="repeated-spaces"),
        pytest.param("a\tb\nc", ["a", "\t", "b", "\n", "c"], id="tabs-newlines"),
        pytest.param("a  ", ["a", "  "], id="trailing-spaces"),
        pytest.param(" \t\n", [" \t\n"], id="only-whitespace"),
    ],
)
def test_pretokenize_ordinary_text(text, expected):
    assert pretokenize(text, []) == expected


@pytest.mark.parametrize(
    ("text", "special_tokens", "expected"),
    [
        pytest.param("", ["<|endoftext|>"], [], id="empty-with-specials"),
        pytest.param("<|endoftext|>", ["<|endoftext|>"], ["<|endoftext|>"], id="only-special"),
        pytest.param(
            "hello<|endoftext|>world", ["<|endoftext|>"], ["hello", "<|endoftext|>", "world"], id="between-words"
        ),
        pytest.param("<s><s></s>", ["<s>", "</s>"], ["<s>", "<s>", "</s>"], id="adjacent-specials"),
        pytest.param("<s>hello", ["<s>", "<s>"], ["<s>", "hello"], id="duplicate-specials"),
        pytest.param("<s>extra", ["<s>", "<s>extra"], ["<s>extra"], id="longest-special-first"),
        pytest.param("a.*[x]b", [".*[x]"], ["a", ".*[x]", "b"], id="literal-regex-characters"),
        pytest.param("hello world", ["<s>"], ["hello", " world"], id="absent-special"),
        pytest.param(
            "hello <|endoftext|>world",
            ["<|endoftext|>"],
            ["hello", " ", "<|endoftext|>", "world"],
            id="space-before-special",
        ),
        pytest.param(
            "hello!<|endoftext|>world",
            ["<|endoftext|>"],
            ["hello", "!", "<|endoftext|>", "world"],
            id="punctuation-before-special",
        ),
        pytest.param("abcSPECIALdef", ["SPECIAL"], ["abc", "SPECIAL", "def"], id="special-inside-word"),
    ],
)
def test_pretokenize_special_tokens(text, special_tokens, expected):
    original_specials = special_tokens.copy()
    actual = pretokenize(text, special_tokens)

    assert special_tokens == original_specials, "Must not mutate the caller's token list"
    assert "".join(actual) == text, "Must preserve every character"
    assert actual == expected
