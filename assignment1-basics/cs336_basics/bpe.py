import heapq
import mmap
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
import regex
from collections import defaultdict, Counter

PRETOKEN_PATTERN = regex.compile(
    r"'(?:[sdmt]|ll|ve|re)"
    r"| ?\p{L}+"
    r"| ?\p{N}+"
    r"| ?[^\s\p{L}\p{N}]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

def pretokenize(text: str, special_tokens: list[str]) -> list[str]:
    # 去重,将长序列放在前面优先匹配
    specials = sorted(set(special_tokens), key=len, reverse=True)
    special_set = set(specials)

    # 没有 special token
    if not specials:
        return PRETOKEN_PATTERN.findall(text)

    special_pattern = "(" + "|".join(
        regex.escape(s) for s in specials
    ) + ")"

    chunks = regex.split(special_pattern, text)
    pieces = list[str]()

    for chunk in chunks:
        if not chunk:
            continue

        # 本身已经是special token已经不用分词
        if chunk in special_set:
            pieces.append(chunk)
        else:
            pieces.extend(
                PRETOKEN_PATTERN.findall(chunk)
            )

    return pieces

def merge_pair(word, pair):
    result = []
    i = 0

    while i < len(word):
        if (
            i + 1 < len(word)
            and word[i] == pair[0]
            and word[i + 1] == pair[1]
        ):
            result.append(word[i] + word[i + 1])
            i += 2

        else:
            result.append(word[i])
            i += 1

    return result


def train_bpe_naive(
    text: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # special token 去重，保留顺序
    special_tokens = list(dict.fromkeys(special_tokens))
    special_set = set(special_tokens)

    if vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab_size is too small")

    # 1. 预分词
    pieces = pretokenize(text, special_tokens)

    # 2. 普通 piece 转成单 byte token 序列
    words = []

    for piece in pieces:
        if piece in special_set:
            continue

        word = [bytes([b]) for b in piece.encode("utf-8")]
        words.append(word)

    # 3. 初始化 vocab
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    # special token 接在 256 个 byte token 后面
    for special in special_tokens:
        vocab[len(vocab)] = special.encode("utf-8")

    merges: list[tuple[bytes, bytes]] = []

    # 4. BPE 训练
    while len(vocab) < vocab_size:
        counts = defaultdict(int)

        for word in words:
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                counts[pair] += 1

        if not counts:
            break

        best = max(counts, key=lambda pair: (counts[pair], pair))

        # 合并
        for i in range(len(words)):
            words[i] = merge_pair(words[i], best)

        # 创建新 token
        new_token = best[0] + best[1]

        # 新 token 分配新的 token ID
        vocab[len(vocab)] = new_token

        # 保存 merge rule
        merges.append(best)

    return vocab, merges


# naive 实现有 3 个问题：
# 1. 统计pair: 当前遍历所有 word 包括重复出现的 word，后续可以不同 word 只存一次，乘以词频
# 2. 更新word: 当前对所有 word 调用 merge_pair, 后续只处理包含目标 pair 的 word
# 3. 选择best: 当前 max 遍历所有pair, 后续可用优先队列


def train_bpe_v1(
    text: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # special token 去重，保留顺序
    special_tokens = list(dict.fromkeys(special_tokens))
    special_set = set(special_tokens)

    if vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab_size is too small")

    pieces = pretokenize(text, special_tokens)
    words_count: Counter[tuple[bytes, ...]] = Counter()

    for piece in pieces:
        if piece in special_set:
            continue

        word = tuple(bytes([b]) for b in piece.encode("utf-8"))
        words_count[word] += 1


    # 3. 初始化 vocab
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    # special token 接在 256 个 byte token 后面
    for special in special_tokens:
        vocab[len(vocab)] = special.encode("utf-8")

    merges: list[tuple[bytes, bytes]] = []

    while len(vocab) < vocab_size:
        counts = defaultdict(int)

        for word, freq in words_count.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                counts[pair] += freq

        if not counts:
            break

        best = max(counts, key=lambda pair: (counts[pair], pair))

        new_words_counts = Counter()
        for word, freq in words_count.items():
            new_word = tuple(merge_pair(word, best))
            new_words_counts[new_word] += freq

        words_count = new_words_counts
        new_token = best[0] + best[1]
        vocab[len(vocab)] = new_token
        merges.append(best)

    return vocab, merges


def count_pairs(word):
    return Counter(zip(word, word[1:]))

def train_bpe_v2(
    text: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # special token 去重，保留顺序
    special_tokens = list(dict.fromkeys(special_tokens))
    special_set = set(special_tokens)

    if vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab_size is too small")

    pieces = pretokenize(text, special_tokens)
    words_count: Counter[tuple[bytes, ...]] = Counter()

    for piece in pieces:
        if piece in special_set:
            continue

        word = tuple(bytes([b]) for b in piece.encode("utf-8"))
        words_count[word] += 1

    word_keys = list(words_count.keys())
    word_freq = list(words_count.values())
    pair_counts = Counter()
    pair_words = defaultdict(set)

    # word_id 和词频固定，合并时只更新 word_keys 中的 token 序列。
    for wid, word in enumerate(word_keys):
        for pair, count in count_pairs(word).items():
            pair_counts[pair] += count * word_freq[wid]
            pair_words[pair].add(wid)

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for special in special_tokens:
        vocab[len(vocab)] = special.encode("utf-8")
    merges: list[tuple[bytes, bytes]] = []

    while len(vocab) < vocab_size and pair_counts:
        best = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))

        # 更新索引时会修改集合，因此先保存受影响 word_id 的快照。
        affected = list(pair_words[best])
        for wid in affected:
            old_word = word_keys[wid]
            new_word = tuple(merge_pair(old_word, best))
            old_counts = count_pairs(old_word)
            new_counts = count_pairs(new_word)

            for pair in old_counts.keys() | new_counts.keys():
                # 局部前后差分也能正确处理 aaa 中重叠的 (a, a)。
                delta = (new_counts[pair] - old_counts[pair]) * word_freq[wid]
                pair_counts[pair] += delta
                if pair_counts[pair] == 0:
                    del pair_counts[pair]

                if new_counts[pair] > 0:
                    pair_words[pair].add(wid)
                else:
                    pair_words[pair].discard(wid)
                    if not pair_words[pair]:
                        del pair_words[pair]

            word_keys[wid] = new_word

        vocab[len(vocab)] = best[0] + best[1]
        merges.append(best)

    return vocab, merges

class _PairCandidate:
    """反转比较顺序，让 Python 3.12 的最小堆按 (频次, pair) 取最大值。"""

    __slots__ = ("count", "pair")

    def __init__(self, count: int, pair: tuple[bytes, bytes]):
        self.count = count
        self.pair = pair

    def __lt__(self, other: "_PairCandidate") -> bool:
        if self.count != other.count:
            return self.count > other.count
        return self.pair > other.pair


def train_bpe_v3(
    text: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # special token 去重，保留顺序
    special_tokens = list(dict.fromkeys(special_tokens))
    special_set = set(special_tokens)

    if vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab_size is too small")

    pieces = pretokenize(text, special_tokens)
    words_count: Counter[tuple[bytes, ...]] = Counter()

    for piece in pieces:
        if piece in special_set:
            continue

        word = tuple(bytes([b]) for b in piece.encode("utf-8"))
        words_count[word] += 1

    return _train_bpe_from_counts(words_count, vocab_size, special_tokens)


def _train_bpe_from_counts(
    words_count: Counter[tuple[bytes, ...]],
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """v3、v4 共用的堆选择和增量合并阶段。"""
    word_keys = list(words_count.keys())
    word_freq = list(words_count.values())
    pair_counts = Counter()
    pair_words = defaultdict(set)

    # word_id 和词频固定，合并时只更新 word_keys 中的 token 序列。
    for wid, word in enumerate(word_keys):
        for pair, count in count_pairs(word).items():
            pair_counts[pair] += count * word_freq[wid]
            pair_words[pair].add(wid)

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for special in special_tokens:
        vocab[len(vocab)] = special.encode("utf-8")
    merges: list[tuple[bytes, bytes]] = []

    heap = [_PairCandidate(count, pair) for pair, count in pair_counts.items()]
    heapq.heapify(heap)

    while len(vocab) < vocab_size and pair_counts:
        # 旧频次保留在堆里，弹出时才检查；pair_counts 是真实计数。
        while heap:
            candidate = heapq.heappop(heap)
            if candidate.count == pair_counts.get(candidate.pair, 0):
                best = candidate.pair
                break
        else:
            raise RuntimeError("pair heap is missing a live candidate")

        changed_pairs = set()

        # 更新索引时会修改集合，因此先保存受影响 word_id 的快照。
        affected = list(pair_words[best])
        for wid in affected:
            old_word = word_keys[wid]
            new_word = tuple(merge_pair(old_word, best))
            old_counts = count_pairs(old_word)
            new_counts = count_pairs(new_word)

            for pair in old_counts.keys() | new_counts.keys():
                # 局部前后差分也能正确处理 aaa 中重叠的 (a, a)。
                delta = (new_counts[pair] - old_counts[pair]) * word_freq[wid]
                if delta:
                    changed_pairs.add(pair)
                pair_counts[pair] += delta
                if pair_counts[pair] == 0:
                    del pair_counts[pair]

                if new_counts[pair] > 0:
                    pair_words[pair].add(wid)
                else:
                    pair_words[pair].discard(wid)
                    if not pair_words[pair]:
                        del pair_words[pair]

            word_keys[wid] = new_word

        # 一轮完成后再入堆，每个变化的 pair 只加入最终频次。
        # 过期条目过多时重建，避免懒删除导致堆无限积累。
        if len(heap) > 4 * len(pair_counts):
            heap = [_PairCandidate(count, pair) for pair, count in pair_counts.items()]
            heapq.heapify(heap)
        else:
            for pair in changed_pairs:
                count = pair_counts.get(pair, 0)
                if count > 0:
                    heapq.heappush(heap, _PairCandidate(count, pair))

        vocab[len(vocab)] = best[0] + best[1]
        merges.append(best)

    return vocab, merges



def _count_pretokens(text: str, special_tokens: list[str]) -> Counter[str]:
    """逐段 finditer，只保存不同预分词的频次，不创建完整 pieces 列表。"""
    counts: Counter[str] = Counter()
    specials = sorted(set(special_tokens), key=len, reverse=True)
    start = 0
    if specials:
        pattern = regex.compile("|".join(regex.escape(s) for s in specials))
        for match in pattern.finditer(text):
            counts.update(m.group() for m in PRETOKEN_PATTERN.finditer(text[start:match.start()]))
            start = match.end()
    counts.update(m.group() for m in PRETOKEN_PATTERN.finditer(text[start:]))
    return counts


def _bpe_file_boundaries(
    input_path: str | os.PathLike,
    num_chunks: int,
    special_tokens: list[str],
) -> list[int]:
    """映射文件并匹配真实特殊 token，边界不会切断 UTF-8 或重叠特殊 token。"""
    size = os.path.getsize(input_path)
    # 含换行的分隔符可能受文本模式换行规范化影响，保守地使用单块。
    if (not size or num_chunks == 1 or not special_tokens
            or any("\r" in s or "\n" in s for s in special_tokens)):
        return [0, size]

    specials = sorted(set(special_tokens), key=len, reverse=True)
    pattern = regex.compile(b"|".join(regex.escape(s.encode("utf-8")) for s in specials))
    boundaries = [0]
    target_index = 1
    with open(input_path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
            for match in pattern.finditer(data):
                position = match.start()
                if position >= size * target_index // num_chunks:
                    if position > boundaries[-1]:
                        boundaries.append(position)
                    # 一个文档可能跨越多个目标分块位置。
                    while target_index < num_chunks and position >= size * target_index // num_chunks:
                        target_index += 1
                    if target_index == num_chunks:
                        break
    boundaries.append(size)
    return boundaries


def _count_file_chunk(args) -> Counter[str]:
    # 模块顶层函数可由 spawn 子进程导入；只传路径和偏移，不传全文。
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8")
    # 与旧版本文本模式读取的通用换行行为保持一致。
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return _count_pretokens(text, special_tokens)


def train_bpe_v4(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_workers: int | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    special_tokens = list(dict.fromkeys(special_tokens))
    if any(not special for special in special_tokens):
        raise ValueError("special tokens must not be empty")
    if vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab_size is too small")
    if num_workers is not None and (not isinstance(num_workers, int) or num_workers < 1):
        raise ValueError("num_workers must be a positive integer")

    size = os.path.getsize(input_path)
    if num_workers is None:
        num_workers = min(4, os.cpu_count() or 1, max(1, size // (8 * 1024 * 1024)))
    boundaries = _bpe_file_boundaries(input_path, num_workers, special_tokens)
    tasks = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries, boundaries[1:])
    ]
    piece_counts: Counter[str] = Counter()
    if len(tasks) == 1:
        piece_counts.update(_count_file_chunk(tasks[0]))
    else:
        with ProcessPoolExecutor(
            max_workers=min(num_workers, len(tasks)),
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            for counts in executor.map(_count_file_chunk, tasks):
                piece_counts.update(counts)

    # 每种预分词只转换一次字节序列，复用 256 个单字节对象。
    byte_tokens = [bytes([b]) for b in range(256)]
    words_count = Counter({
        tuple(byte_tokens[b] for b in piece.encode("utf-8")): freq
        for piece, freq in piece_counts.items()
    })
    return _train_bpe_from_counts(words_count, vocab_size, special_tokens)
