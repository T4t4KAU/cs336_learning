# CS336 Basics：算法与代码讲解

## 一、数学定义与算法

本部分给出符号、算子定义与算法流程；第二部分按文件将实现说明与对应代码放在一起。

### 1. 字节级 BPE 训练

输入是文本、目标词表大小和特殊 token 集合；输出是 ID 到字节串的词表，以及有序合并规则。文本先按特殊 token 分段，再正则预分词，每个预分词表示为 UTF-8 字节序列。合并仅发生在同一预分词内部；特殊 token 不参加普通 pair 统计。

设不同预分词的当前序列为 $w_j=(t_{j,1},\ldots,t_{j,n_j})$，出现次数为 $f_j$。相邻 pair 的加权频次为：

$$
C(a,b)=\sum_j f_j\sum_{k=1}^{n_j-1}
\mathbf{1}[(t_{j,k},t_{j,k+1})=(a,b)].
$$

每轮选取最高频 pair；同频时选字节元组字典序最大的 pair：

$$
(a^*,b^*)=\arg\max_{(a,b)}\bigl(C(a,b),(a,b)\bigr).
$$

新 token 为 $a^*\Vert b^*$，其中 $\Vert$ 表示字节拼接。统计包含重叠位置，替换却从左到右不重叠：三个连续 a 中 pair (a,a) 计数为 2，一轮合并得到 (aa,a)。

**朴素算法**每轮重新统计和替换所有预分词：

```text
TRAIN_BPE_NAIVE(text, target_size, special_tokens):
    specials ← 去重后的特殊 token，保留原顺序
    vocab ← 256 个单字节 token，再追加 specials
    merges ← 空列表
    words ← 按 specials 分段、正则预分词，再逐个转成单字节序列

    while |vocab| < target_size:
        counts ← 空计数表
        for word in words:                 # 重复预分词逐次保留
            for 相邻 pair in word:
                counts[pair] += 1
        if counts 为空:
            break

        best ← 按 (counts[pair], pair) 选最大 pair
        new_token ← best.left 拼接 best.right
        vocab 追加 new_token
        merges 追加 best
        for word in words:
            word ← MERGE_LEFT_TO_RIGHT(word, best)
    return vocab, merges

MERGE_LEFT_TO_RIGHT(word, pair):
    result ← 空列表
    i ← 0
    while i < |word|:
        if i+1 < |word| 且 (word[i], word[i+1]) = pair:
            result 追加 word[i] 拼接 word[i+1]
            i ← i+2
        else:
            result 追加 word[i]
            i ← i+1
    return result
```

若第 r 轮共有 $N_r$ 个 token、$P_r$ 种 pair，该轮约需 $O(N_r+P_r)$ 的统计、选取与替换工作。

**增量算法**只更新包含被选 pair 的词。设词 j 修改前后的局部 pair 计数分别为 $c_{old,j}$、$c_{new,j}$，全局频次更新为：

$$
C'(p)=C(p)+f_j\left(c_{\mathrm{new},j}(p)-c_{\mathrm{old},j}(p)\right).
$$

```text
TRAIN_BPE_INCREMENTAL(input, target_size, specials):
    vocab, merges ← 初始化词表、空合并列表
    word_counts ← 对流式预分词结果计数
    words, frequencies ← 不同预分词的字节序列及其词频
    pair_counts, pair_words ← 根据 words 初始化

    while |vocab| < target_size 且 pair_counts 非空:
        best ← 按 (pair_counts[pair], pair) 选最大 pair
        vocab 追加 best.left 拼接 best.right
        merges 追加 best
        affected ← pair_words[best] 的快照

        for j in affected:
            old_word ← words[j]
            new_word ← MERGE_LEFT_TO_RIGHT(old_word, best)
            old_counts ← old_word 的局部相邻 pair 计数
            new_counts ← new_word 的局部相邻 pair 计数

            for pair in 两个局部计数表的键并集:
                delta ← (new_counts[pair] - old_counts[pair]) × frequencies[j]
                pair_counts[pair] += delta
                删除频次归零的全局 pair
                根据新词中是否仍含 pair 更新 pair_words[pair]
                删除空的倒排集合
            words[j] ← new_word
    return vocab, merges
```

两种算法采用相同的计数、同频比较和替换规则，因此应得到相同的词表与合并序列。

### 2. BPE 编码与解码

编码使用训练得到的合并顺序，不再统计频次。令 $r(a,b)$ 为 pair 在有序合并规则中的位置；未出现的 pair 记为 $+\infty$。每一步在当前相邻 pair 中选择最小 rank，同 rank 取最左位置：

$$
k^*=\arg\min_k r(t_k,t_{k+1}),
$$

```text
ENCODE(text):
    ids ← 空列表
    按特殊 token 匹配位置顺序处理文本：
        普通文本段：
            for pretoken in REGEX_PRETOKENIZE(文本段):
                tokens ← pretoken 的 UTF-8 单字节序列
                while 存在可合并的相邻 pair:
                    k ← 当前 pair 中 rank 最小的位置，同 rank 取最左侧
                    tokens[k:k+2] ← 两者拼接得到的一个 token
                ids 追加各 token 对应的 ID
        特殊 token：
            ids 追加其专属 ID
    return ids

DECODE(ids):
    bytestring ← 按 ids 顺序拼接 vocab[id]
    return UTF8_DECODE(bytestring, errors="replace")
```

特殊 token 应整体编码；若匹配位置相同，优先识别更长的特殊 token。解码必须先拼接全部字节，因为单个 Unicode 字符可能跨越多个 token。

### 3. Linear 与 Embedding

**Linear。** 输入 $X\in\mathbb R^{\cdots\times d_{in}}$，权重 $W\in\mathbb R^{d_{out}\times d_{in}}$，无偏置：

$$
Y=XW^\top,\qquad Y\in\mathbb{R}^{\cdots\times d_{out}}.
$$

初始化分布为：

$$
W_{ij}\sim\mathcal N(0,\sigma^2)\text{ 截断于 }[-3\sigma,3\sigma],
\qquad \sigma=\sqrt{\frac{2}{d_{in}+d_{out}}}.
$$

**Embedding。** 词向量矩阵 $E\in\mathbb R^{V\times d}$，整数索引 $I\in\{0,\ldots,V-1\}^{\cdots}$：

$$
Y_{\ldots,j}=E_{I_{\ldots},j},\qquad Y\in\mathbb{R}^{\cdots\times d}.
$$

矩阵元素按均值 0、标准差 1、区间 [-3,3] 的截断正态初始化。重复索引对应的梯度累加到同一行。

### 4. RMSNorm、SiLU 与 SwiGLU

**RMSNorm。** 对 $x\in\mathbb R^d$，可训练缩放为 $g\in\mathbb R^d$，$\epsilon>0$：

$$
\operatorname{RMSNorm}(x)_i=
\frac{x_i}{\sqrt{\frac1d\sum_{j=1}^{d}x_j^2+\epsilon}}g_i.
$$

$g$ 初始化为全 1。归一化沿最后一维进行，不减均值，输入输出形状相同。

**SiLU。** 对每个元素定义：

$$
\sigma(x)=\frac1{1+e^{-x}},\qquad
\operatorname{SiLU}(x)=x\sigma(x).
$$

**SwiGLU。** 使用列向量表示，$W_1,W_3\in\mathbb R^{f\times d}$，$W_2\in\mathbb R^{d\times f}$：

$$
\operatorname{SwiGLU}(x)=W_2\left(\operatorname{SiLU}(W_1x)\odot W_3x\right).
$$

$\odot$ 表示逐元素乘法，隐藏宽度 f 通常取接近 $8d/3$ 的 64 倍数。不带门控的变体为：

$$
\operatorname{FFN}_{SiLU}(x)=W_2\operatorname{SiLU}(W_1x).
$$

两次投影的 SiLU 前馈通常取 $f=4d$，与三次投影的 SwiGLU 近似匹配参数量。

### 5. RoPE

设每头维度 $d_k$ 为偶数，token 位置为 p，相邻元素对编号为 $i=0,\ldots,d_k/2-1$，频率基数为 $\Theta>0$：

$$
\omega_i=\Theta^{-2i/d_k},\qquad \phi_{p,i}=p\omega_i.
$$

每对相邻元素按下式旋转：

$$
\begin{bmatrix}y_{2i}\\y_{2i+1}\end{bmatrix}
=
\begin{bmatrix}
\cos\phi_{p,i}&-\sin\phi_{p,i}\\
\sin\phi_{p,i}&\cos\phi_{p,i}
\end{bmatrix}
\begin{bmatrix}x_{2i}\\x_{2i+1}\end{bmatrix}.
$$

位置 0 对应单位旋转，旋转保持每对元素的平方和。RoPE 只作用于注意力中的 Q、K，不改变 V。

### 6. Softmax 与注意力

**Softmax。** 对归一化轴上的 $x\in\mathbb R^n$：

$$
m=\max_j x_j,\qquad
\operatorname{softmax}(x)_i=\frac{e^{x_i-m}}{\sum_j e^{x_j-m}}.
$$

减去同一常数不改变分布，选择最大值可避免正指数溢出。

**缩放点积注意力。** 给定 $Q\in\mathbb R^{\cdots\times S_q\times d_k}$、$K\in\mathbb R^{\cdots\times S_k\times d_k}$、$V\in\mathbb R^{\cdots\times S_k\times d_v}$：

$$
A=\frac{QK^\top}{\sqrt{d_k}}+M,\qquad
O=\operatorname{softmax}_{\text{key}}(A)V.
$$

$M_{ij}=0$ 表示允许访问，$M_{ij}=-\infty$ 表示禁止访问。softmax 沿 key 维归一化，输出形状为 $\cdots\times S_q\times d_v$。每个 query 至少允许一个 key，否则该行分布不定义。

**因果多头自注意力。** 令 $d=Hd_k$，输入 $X\in\mathbb R^{\cdots\times S\times d}$，投影为：

$$
Q=XW_Q^\top,\quad K=XW_K^\top,\quad V=XW_V^\top.
$$

将特征分为 H 个头，各头分别对 Q/K 应用 RoPE，使用因果掩码：

$$
M_{ij}=\begin{cases}0,&j\le i,\\-\infty,&j>i.\end{cases}
$$

各头输出为 $O_h$，拼接后进行输出投影：

$$
\operatorname{MHA}(X)=\operatorname{Concat}(O_1,\ldots,O_H)W_O^\top.
$$

### 7. Transformer Block 与语言模型

Pre-norm block：

$$
Z=X+\operatorname{MHA}(\operatorname{RMSNorm}_1(X)),
\qquad Y=Z+\operatorname{FFN}(\operatorname{RMSNorm}_2(Z)).
$$

Post-norm block：

$$
Z=\operatorname{RMSNorm}_1(X+\operatorname{MHA}(X)),
\qquad Y=\operatorname{RMSNorm}_2(Z+\operatorname{FFN}(Z)).
$$

无归一化变体移除归一化操作、保留残差与子层。语言模型由 L 个 block 组成：

$$
H_0=\operatorname{Embedding}(I),\quad H_{\ell+1}=\operatorname{Block}_{\ell}(H_\ell),
\quad U=\operatorname{RMSNorm}_{final}(H_L)W_{vocab}^\top.
$$

$H_0\in\mathbb R^{B\times S\times d}$，输出 $U\in\mathbb R^{B\times S\times V}$ 为 logits；输入词向量与输出投影不共享权重。

### 8. 交叉熵

N 个预测位置的 logits 为 $z_n\in\mathbb R^V$，真实类别为 $y_n$：

$$
\mathcal L=\frac1N\sum_{n=1}^N\left[\log\sum_{j=1}^{V}e^{z_{n,j}}-z_{n,y_n}\right].
$$

令 $u_{n,j}=z_{n,j}-\max_k z_{n,k}$，可写成数值稳定的形式：

$$
\mathcal L=\frac1N\sum_n\left[\log\sum_j e^{u_{n,j}}-u_{n,y_n}\right].
$$

语言模型中 $N=BS$，损失对全部预测位置平均。

### 9. AdamW、学习率调度与梯度裁剪

**AdamW。** 对参数 $\theta$，第 t 次更新梯度为 $g_t$，初始 $m_0=v_0=0$：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
$$

设学习率为 $\eta_t$、权重衰减系数为 $\lambda$。采用先自适应更新、再衰减的次序：

$$
\alpha_t=\eta_t\frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t},\qquad
\widetilde\theta_t=\theta_{t-1}-\alpha_t\frac{m_t}{\sqrt{v_t}+\epsilon},
\qquad\theta_t=(1-\eta_t\lambda)\widetilde\theta_t.
$$

平方与除法均逐元素进行。此处 epsilon 位于未作偏置校正的二阶矩平方根之后；比较不同形式时需保持其位置与衰减顺序一致。

**Warmup 与余弦调度。** $T_w$ 为 warmup 长度，$T_c>T_w$ 为衰减终点：

$$
\eta(t)=
\begin{cases}
\eta_{max}\,t/T_w,&0\le t<T_w,\\
\eta_{min}+\frac12(\eta_{max}-\eta_{min})\left[1+\cos\left(\pi\frac{t-T_w}{T_c-T_w}\right)\right],&T_w\le t<T_c,\\
\eta_{min},&t\ge T_c.
\end{cases}
$$

当 $T_w=0$ 时直接进入余弦阶段。

**全局梯度裁剪。** 对全部非空参数梯度定义：

$$
G=\sqrt{\sum_p\sum_i(g_{p,i})^2},\qquad
c=\min\left(1,\frac{G_{max}}{G+10^{-6}}\right),\qquad g_p\leftarrow c g_p.
$$

所有参数梯度乘同一个 c；对每个参数分别裁剪不等价于全局裁剪。

### 10. 数据采样、梯度累积与评估

**窗口采样。** 长度 N 的 token 数组为 D，上下文长度为 S。每个 batch 元素的起点独立采样：

$$
a_b\sim\operatorname{Uniform}\{0,\ldots,N-S-1\},\quad
X_{b,j}=D_{a_b+j},\quad Y_{b,j}=D_{a_b+j+1}.
$$

**梯度累积。** K 个等大 microbatch 的平均损失与梯度为：

$$
\overline{\mathcal L}=\frac1K\sum_{k=1}^K\mathcal L_k,
\qquad \nabla\overline{\mathcal L}=\frac1K\sum_{k=1}^K\nabla\mathcal L_k.
$$

```text
for step in 起始步数 .. 总步数-1:
    设置当前学习率
    清空梯度
    repeat K 次:
        X, Y ← 随机采样一个 microbatch
        loss ← CROSS_ENTROPY(model(X), Y)
        对 loss / K 执行 backward
    裁剪全局梯度
    optimizer.step()
    记录 loss、学习率、token 数与运行时间
    到达验证间隔时：评估并保存检查点
```

**验证集计权。** 若窗口 b 包含 $n_b$ 个预测 token，平均损失为 $\mathcal L_b$，整体损失为：

$$
\mathcal L_{valid}=\frac{\sum_b n_b\mathcal L_b}{\sum_b n_b}.
$$

最后一个窗口不足固定长度时，也按其实际 token 数计入。

### 11. 温度与 top-p 生成

取当前上下文最后位置的 logits z。温度 $\tau>0$ 下：

$$
q_i=\frac{e^{z_i/\tau}}{\sum_j e^{z_j/\tau}}.
$$

将概率降序排列为 $q_{\pi_1}\ge\cdots\ge q_{\pi_V}$。阈值 $p\in(0,1]$ 下，定义最小候选集合及其重新归一化分布：

$$
k^*=\min\left\{k:\sum_{j=1}^k q_{\pi_j}\ge p\right\},\qquad
\widetilde q_{\pi_j}=
\begin{cases}
q_{\pi_j}/\sum_{r=1}^{k^*}q_{\pi_r},&j\le k^*,\\
0,&j>k^*.
\end{cases}
$$

保留使累计概率首次达到 p 的 token。温度为 0 时另定义为选择最大 logit 的 greedy 生成。

```text
GENERATE(model, prompt, limit, temperature, top_p, eos):
    ids ← prompt
    repeat 至多 limit 次:
        context ← ids 的最后 context_length 个 ID
        logits ← model(context) 最后位置的输出
        next ← 根据温度与 top-p 采样，或使用 greedy
        ids 追加 next
        if next = eos:
            break
    return ids
```

### 12. 参数与矩阵乘法计算量

令 V 为词表大小，S 为序列长度，L 为层数，d 为模型宽度，f 为前馈隐藏宽度；无 bias、输入输出词向量不共享：

$$
P=2Vd+L(4d^2+3df+2d)+d.
$$

单序列主要矩阵乘法计算量为：

$$
F=8LSd^2+4LS^2d+6LSdf+2SdV.
$$

四项分别来自 QKVO 投影、两个 attention 矩阵乘法、三个 FFN 投影与词表输出投影。一次乘加按 2 FLOPs 计，不包括逐元素运算。

## 二、实现说明与代码

按组件依赖顺序组织文件，每个代码块前给出实现说明，后面给出测试命令。命令在项目根目录执行；应先完成对应模块及其依赖，并接通 `tests/adapters.py` 中相关接口。

### File: cs336_basics/bpe.py

对应第一部分第 1 节。

`train_bpe_naive` 全文读取、保留每个预分词实例，每轮重新扫描。`train_bpe` 用 `Counter` 合并相同预分词，减少重复存储；三个关键结构分别是：

| 结构 | 内容 |
|---|---|
| `frequencies[j]` | 不同预分词 j 在文本中的出现次数 |
| `pair_counts[pair]` | 按词频加权后的全局相邻 pair 计数 |
| `pair_words[pair]` | 包含该 pair 的词编号集合 |

选中 pair 后先复制受影响词编号，再逐个合并。使用旧词与新词的局部 `Counter` 差额更新全局频次；同时增删倒排索引成员。计数归零时删除键，集合为空时删除集合。快照避免遍历期间修改同一集合。

`_iter_pretokens` 处理读取边界：保留尾部预分词和可能未读完的特殊 token，并让正则看到真实文档边界。不同词、计数表和倒排索引仍在内存中；极长预分词可能扩大缓冲区。最大 pair 的选择仍通过线性扫描完成。

```python
"""Byte-level BPE training with incremental pair statistics.

The corpus is read in chunks. Unique pre-tokens and pair statistics still live in
RAM, so this is not a hard-memory-bounded trainer for arbitrarily large corpora.
"""

from collections import Counter, defaultdict, deque
from collections.abc import Iterator
from os import PathLike

import regex

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
Pair = tuple[bytes, bytes]


def _iter_pretokens(
    input_path: str | PathLike[str], special_tokens: list[str], chunk_size: int = 1024 * 1024
) -> Iterator[str]:
    """Yield regex pre-tokens, treating special tokens as discarded boundaries.

    Retain the final matches and any possible special-token prefix across reads:
    a chunk boundary must not become an artificial pre-token boundary.
    """
    special_pattern = "|".join(regex.escape(token) for token in sorted(special_tokens, key=len, reverse=True))
    special_re = regex.compile(special_pattern) if special_pattern else None
    pattern = regex.compile(PAT)
    overlap = max(map(len, special_tokens), default=0)
    buffer = ""
    with open(input_path, encoding="utf-8", newline="") as source:
        while True:
            chunk = source.read(chunk_size)
            buffer += chunk
            # Pre-tokenize each completed document separately: whitespace regex
            # lookaheads must see the document end, not the following special.
            if special_re is not None:
                document_start = 0
                for boundary in special_re.finditer(buffer):
                    if chunk and boundary.start() > len(buffer) - overlap:
                        break
                    for match in pattern.finditer(buffer[document_start : boundary.start()]):
                        yield match.group()
                    document_start = boundary.end()
                buffer = buffer[document_start:]
            if not chunk:
                for match in pattern.finditer(buffer):
                    yield match.group()
                return

            pending = deque()
            consumed = 0
            cutoff = len(buffer) - overlap
            for match in pattern.finditer(buffer[: max(0, cutoff)]):
                pending.append(match)
                if len(pending) > 2:
                    ready = pending.popleft()
                    consumed = ready.end()
                    yield ready.group()
            buffer = buffer[consumed:]


def train_bpe(
    input_path: str | PathLike[str], vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[Pair]]:
    """Train BPE and return the token-ID vocabulary and ordered byte-pair merges.

    Pair frequencies are weighted by pre-token occurrence counts. Ties are
    resolved by the lexicographically greatest pair of byte strings. Only words
    containing the selected pair need their statistics updated after a merge.
    """
    specials = list(dict.fromkeys(special_tokens))
    if any(not token for token in specials):
        raise ValueError("Special tokens must be nonempty strings")
    if vocab_size < 256 + len(specials):
        raise ValueError("vocab_size must include all 256 bytes and the special tokens")

    byte_vocab = tuple(bytes([value]) for value in range(256))
    vocab = dict(enumerate(byte_vocab))
    for token in specials:
        vocab[len(vocab)] = token.encode("utf-8")
    merges: list[Pair] = []
    if len(vocab) == vocab_size:
        return vocab, merges

    pretoken_counts = Counter(_iter_pretokens(input_path, specials))
    words = [tuple(byte_vocab[value] for value in word.encode("utf-8")) for word in pretoken_counts]
    frequencies = list(pretoken_counts.values())
    del pretoken_counts

    pair_counts: Counter[Pair] = Counter()
    pair_words: dict[Pair, set[int]] = defaultdict(set)
    for word_id, word in enumerate(words):
        for pair, occurrences in Counter(zip(word, word[1:])).items():
            pair_counts[pair] += occurrences * frequencies[word_id]
            pair_words[pair].add(word_id)

    while len(vocab) < vocab_size and pair_counts:
        best = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        merged_token = best[0] + best[1]
        vocab[len(vocab)] = merged_token
        merges.append(best)

        # Snapshot because the inverted index is updated during this loop.
        affected_words = tuple(pair_words[best])
        for word_id in affected_words:
            old_word = words[word_id]
            new_word: list[bytes] = []
            position = 0
            while position < len(old_word):
                if position + 1 < len(old_word) and (old_word[position], old_word[position + 1]) == best:
                    new_word.append(merged_token)
                    position += 2
                else:
                    new_word.append(old_word[position])
                    position += 1

            old_pairs = Counter(zip(old_word, old_word[1:]))
            new_pairs = Counter(zip(new_word, new_word[1:]))
            for pair in old_pairs.keys() | new_pairs.keys():
                delta = new_pairs[pair] - old_pairs[pair]
                if delta:
                    pair_counts[pair] += delta * frequencies[word_id]
                    if pair_counts[pair] == 0:
                        del pair_counts[pair]
                if pair not in new_pairs:
                    pair_words[pair].discard(word_id)
                    if not pair_words[pair]:
                        del pair_words[pair]
                elif pair not in old_pairs:
                    pair_words[pair].add(word_id)
            words[word_id] = tuple(new_word)

    return vocab, merges


def train_bpe_naive(
    input_path: str | PathLike[str], vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[Pair]]:
    specials = list(dict.fromkeys(special_tokens))
    if any(not token for token in specials):
        raise ValueError("Special tokens must be nonempty strings")
    if vocab_size < 256 + len(specials):
        raise ValueError("vocab_size must include all 256 bytes and the special tokens")

    # 1. Initialize the vocabulary with all single bytes and special tokens.
    vocab = {value: bytes([value]) for value in range(256)}
    for token in specials:
        vocab[len(vocab)] = token.encode("utf-8")
    merges: list[Pair] = []
    if len(vocab) == vocab_size:
        return vocab, merges

    # 2. Split on special tokens, apply regex pre-tokenization, and convert to UTF-8 bytes.
    with open(input_path, encoding="utf-8", newline="") as source:
        text = source.read()
    special_pattern = "|".join(regex.escape(token) for token in sorted(specials, key=len, reverse=True))
    documents = regex.split(special_pattern, text) if specials else [text]
    words = [
        [vocab[value] for value in match.group().encode("utf-8")]
        for document in documents
        for match in regex.finditer(PAT, document)
    ]

    while len(vocab) < vocab_size:
        # 3. Recount all adjacent pairs, including every occurrence of repeated pre-tokens.
        pair_counts: Counter[Pair] = Counter()
        for word in words:
            pair_counts.update(zip(word, word[1:]))
        if not pair_counts:
            break

        # 4. Select the most frequent pair; break ties by greatest byte lexicographic order.
        best = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        merged_token = best[0] + best[1]
        vocab[len(vocab)] = merged_token
        merges.append(best)

        # 5. Scan all pre-tokens and merge non-overlapping matches from left to right.
        for word_id, word in enumerate(words):
            new_word: list[bytes] = []
            position = 0
            while position < len(word):
                if position + 1 < len(word) and (word[position], word[position + 1]) == best:
                    new_word.append(merged_token)
                    position += 2
                else:
                    new_word.append(word[position])
                    position += 1
            words[word_id] = new_word

    return vocab, merges
```

**测试命令**

验证词表、合并顺序、特殊 token 和分块边界。

```bash
uv run --frozen python -m pytest tests/test_train_bpe.py tests/test_bpe_training_edges.py -q
```

### File: cs336_basics/tokenizer.py

对应第一部分第 2 节。

`__init__` 复制词表，建立 `token_to_id` 与 `merge_ranks`；特殊 token 去重后按长度降序构建转义正则。`_encode_ordinary` 每次选择当前 rank 最小的相邻位置，用列表切片替换为拼接后的 token。

`from_files` 使用 `json.load` 读取十六进制 JSON，再通过 `bytes.fromhex` 还原字节。`decode` 使用 `b''.join(...)` 后整体解码，并设置 `errors='replace'`。

`encode_iterable` 对输入字符串逐个独立编码，不保证任意分块与整段编码等价。

```python
"""Encode and decode text using a trained byte vocabulary and BPE merge rules."""

import json
from collections.abc import Iterable, Iterator
from os import PathLike

import regex

from cs336_basics.bpe import PAT, Pair


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[Pair], special_tokens: list[str] | None = None):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.token_to_id = {token: token_id for token_id, token in self.vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.special_tokens = list(dict.fromkeys(special_tokens or []))
        if any(not token for token in self.special_tokens):
            raise ValueError("Special tokens must be nonempty strings")

        # Append missing special tokens without modifying the caller's vocabulary.
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.token_to_id:
                token_id = max(self.vocab, default=-1) + 1
                self.vocab[token_id] = token_bytes
                self.token_to_id[token_bytes] = token_id
        special_pattern = "|".join(regex.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True))
        self.special_re = regex.compile(special_pattern) if special_pattern else None
        self.pattern = regex.compile(PAT)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | PathLike[str],
        merges_filepath: str | PathLike[str],
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """Load this project's trained vocab.hex.json and merges.hex.json files."""
        with open(vocab_filepath, encoding="utf-8") as vocab_file:
            vocab = {int(key): bytes.fromhex(value) for key, value in json.load(vocab_file).items()}
        with open(merges_filepath, encoding="utf-8") as merges_file:
            merges = [(bytes.fromhex(left), bytes.fromhex(right)) for left, right in json.load(merges_file)]
        return cls(vocab, merges, special_tokens)

    def _encode_ordinary(self, text: str) -> Iterator[int]:
        for match in self.pattern.finditer(text):
            tokens = [bytes([value]) for value in match.group().encode("utf-8")]
            while len(tokens) > 1:
                # Merge the adjacent pair with the earliest training rank, breaking ties left to right.
                position = min(
                    range(len(tokens) - 1),
                    key=lambda i: self.merge_ranks.get((tokens[i], tokens[i + 1]), float("inf")),
                )
                pair = (tokens[position], tokens[position + 1])
                if pair not in self.merge_ranks:
                    break
                tokens[position : position + 2] = [pair[0] + pair[1]]
            for token in tokens:
                yield self.token_to_id[token]

    def encode(self, text: str) -> list[int]:
        """Isolate special tokens, then pre-tokenize each text segment and apply BPE."""
        ids = []
        start = 0
        if self.special_re is not None:
            for match in self.special_re.finditer(text):
                ids.extend(self._encode_ordinary(text[start : match.start()]))
                ids.append(self.token_to_id[match.group().encode("utf-8")])
                start = match.end()
        ids.extend(self._encode_ordinary(text[start:]))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Encode each segment independently, supporting line-by-line file iteration."""
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        # Join bytes before decoding because a Unicode character may span multiple tokens.
        return b"".join(self.vocab[token_id] for token_id in ids).decode("utf-8", errors="replace")
```

**测试命令**

依赖 BPE 的预分词表达式与类型定义，验证编码、解码和文件加载。

```bash
uv run --frozen python -m pytest tests/test_tokenizer.py tests/test_tokenizer_edges.py -q
```

### File: cs336_basics/preprocess.py

配合第一部分第 1—2 节。`train_tokenizer` 保存词表和合并规则；`iter_documents` 保留完整 EOS 文档；`encode_file` 逐篇写出 ID，因此内存仍需容纳一篇文档。最大 ID 小于 65536 时使用 `<u2`，否则使用 `<u4`。输出是无 header 数组，可通过 `np.memmap` 读取。

`measure_tokenizer` 使用固定种子的 reservoir sampling 抽样，统计压缩率与编码吞吐。抽样扫描与 tokenizer 构造不计入编码时间。

```python
"""Train tokenizers, measure compression, and write memory-mappable token arrays."""

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np

from cs336_basics.bpe import train_bpe
from cs336_basics.tokenizer import Tokenizer


EOS = "<|endoftext|>"


def iter_documents(path, chunk_size=1024 * 1024):
    """Yield complete EOS-delimited documents, retaining a partial final document."""
    buffer = ""
    with open(path, encoding="utf-8", newline="") as source:
        while chunk := source.read(chunk_size):
            parts = (buffer + chunk).split(EOS)
            for document in parts[:-1]:
                yield document + EOS
            buffer = parts[-1]
        if buffer:
            yield buffer


def load_tokenizer(directory):
    directory = Path(directory)
    return Tokenizer.from_files(directory / "vocab.hex.json", directory / "merges.hex.json", [EOS])


def train_tokenizer(input_path, output_dir, vocab_size):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    vocab, merges = train_bpe(input_path, vocab_size, [EOS])
    (output / "vocab.hex.json").write_text(json.dumps({i: v.hex() for i, v in vocab.items()}))
    (output / "merges.hex.json").write_text(json.dumps([(a.hex(), b.hex()) for a, b in merges]))
    longest = max(vocab.values(), key=len)
    report = dict(
        elapsed_s=time.perf_counter() - start,
        vocab_size=len(vocab),
        merges=len(merges),
        peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        longest_hex=longest.hex(),
        longest_repr=repr(longest),
        input_bytes=Path(input_path).stat().st_size,
    )
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    return report


def encode_file(input_path, tokenizer_dir, output_path):
    tokenizer = load_tokenizer(tokenizer_dir)
    dtype = np.dtype("<u2" if max(tokenizer.vocab) < 65536 else "<u4")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    start, count = time.perf_counter(), 0
    with open(temporary, "wb") as dest:
        for document in iter_documents(input_path):
            ids = tokenizer.encode(document)
            np.asarray(ids, dtype=dtype).tofile(dest)
            count += len(ids)
    temporary.replace(output)
    report = dict(
        tokens=count,
        dtype=dtype.str,
        elapsed_s=time.perf_counter() - start,
        input_bytes=Path(input_path).stat().st_size,
    )
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    return report


def measure_tokenizer(input_path, tokenizer_dir, num_documents=10, seed=42):
    # Reservoir sampling visits the corpus once without retaining all documents.
    rng = np.random.default_rng(seed)
    sample = []
    for index, document in enumerate(iter_documents(input_path)):
        if index < num_documents:
            sample.append(document)
        else:
            slot = int(rng.integers(index + 1))
            if slot < num_documents:
                sample[slot] = document
    tokenizer = load_tokenizer(tokenizer_dir)
    start = time.perf_counter()
    tokens = sum(len(tokenizer.encode(document)) for document in sample)
    elapsed = time.perf_counter() - start
    size = sum(len(document.encode()) for document in sample)
    report = dict(
        documents=len(sample),
        bytes=size,
        tokens=tokens,
        bytes_per_token=size / max(1, tokens),
        bytes_per_second=size / max(elapsed, 1e-9),
        encode_seconds=elapsed,
        estimated_pile_hours=825e9 / max(size / max(elapsed, 1e-9), 1e-9) / 3600,
    )
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("bpe")
    train.add_argument("input")
    train.add_argument("output")
    train.add_argument("--vocab-size", type=int, default=10000)
    for name in ("encode", "stats"):
        p = sub.add_parser(name)
        p.add_argument("input")
        p.add_argument("tokenizer_dir")
        if name == "encode":
            p.add_argument("output")
    args = parser.parse_args()
    if args.command == "bpe":
        train_tokenizer(args.input, args.output, args.vocab_size)
    elif args.command == "encode":
        encode_file(args.input, args.tokenizer_dir, args.output)
    else:
        measure_tokenizer(args.input, args.tokenizer_dir)


if __name__ == "__main__":
    main()
```

**测试命令**

需完成 BPE 和 Tokenizer；使用临时文件验证训练词表、编码落盘与往返还原。

```bash
uv run --frozen python -m pytest tests/test_training_integration.py::test_preprocess_roundtrip_and_boundaries -q
```

### File: cs336_basics/linear.py

对应第一部分第 3 节。

`nn.Parameter(torch.empty(...))` 注册 Linear 权重，再用 `nn.init.trunc_normal_` 初始化。前向调用 `x @ self.weight.T`，权重保留 `(out_features,in_features)` 的形状。

设备和 dtype 在创建权重时指定。

```python
"""A bias-free linear layer implemented with basic tensor operations."""

import torch
from torch import nn


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        # The assignment specifies variance 2 / (in_features + out_features).
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map input shape (..., in_features) to output shape (..., out_features)."""
        return x @ self.weight.T
```

**测试命令**

接通 `run_linear` 后验证线性变换。

```bash
uv run --frozen python -m pytest tests/test_model.py::test_linear -q
```

### File: cs336_basics/embedding.py

对应第一部分第 3 节。

Embedding 权重为 `(num_embeddings,embedding_dim)`，前向直接使用 `self.weight[token_ids]`。设备和 dtype 在创建权重时指定；索引张量使用整数类型。

```python
"""Token embedding lookup using a trainable weight matrix."""

import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map token IDs of shape (...) to vectors of shape (..., embedding_dim)."""
        return self.weight[token_ids]
```

**测试命令**

接通 `run_embedding` 后验证向量查找。

```bash
uv run --frozen python -m pytest tests/test_model.py::test_embedding -q
```

### File: cs336_basics/rmsnorm.py

对应第一部分第 4 节。

RMSNorm 先记录输入 dtype，将输入转 float32，再执行 `square().mean(dim=-1, keepdim=True)` 和 `torch.rsqrt`。保留末维用于广播；乘可训练缩放后转回输入 dtype。

```python
"""Root mean square normalization over the final feature dimension."""

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize features and apply a learned scale, preserving shape and dtype."""
        input_dtype = x.dtype
        # Compute in float32 to avoid overflow when squaring low-precision inputs.
        x = x.to(torch.float32)
        mean_square = x.square().mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(mean_square + self.eps)
        return (normalized * self.weight).to(input_dtype)
```

**测试命令**

接通 `run_rmsnorm` 后验证归一化结果。

```bash
uv run --frozen python -m pytest tests/test_model.py::test_rmsnorm -q
```

### File: cs336_basics/swiglu.py

对应第一部分第 4 节。SiLU 调用 `x * torch.sigmoid(x)`；SwiGLU 复用三个 `Linear`，前向为 `self.w2(silu(self.w1(x)) * self.w3(x))`。其中 `*` 是门控的逐元素乘。隐藏宽度可显式指定，也可由构造器计算默认值。

```python
"""SiLU activation and the SwiGLU position-wise feed-forward network."""

import torch
from torch import nn

from cs336_basics.linear import Linear


def silu(x: torch.Tensor) -> torch.Tensor:
    """Apply the SiLU activation elementwise."""
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        # Default to the nearest positive multiple of 64 to 8/3 * d_model.
        if d_ff is None:
            d_ff = max(64, round((8 * d_model / 3) / 64) * 64)
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a gated feed-forward transformation to the final dimension."""
        return self.w2(silu(self.w1(x)) * self.w3(x))
```

**测试命令**

需完成 Linear，并接通 SiLU 与 SwiGLU 适配器。

```bash
uv run --frozen python -m pytest tests/test_model.py::test_silu_matches_pytorch tests/test_model.py::test_swiglu -q
```

### File: cs336_basics/rope.py

对应第一部分第 5 节。

构造器验证每头维度为正偶数，预计算 `(max_seq_len,d_k/2)` 的 cos/sin 表。通过 `register_buffer(..., persistent=False)` 保存，使其随模型移动设备，同时无需写入检查点。

`forward` 用 `token_positions` 查表，用 `x[...,0::2]` 与 `x[...,1::2]` 分离两组相邻元素。旋转后通过 `stack(..., dim=-1).flatten(-2)` 交错恢复。最终转回输入 dtype；位置张量必须与前置批次维度兼容。

```python
"""Rotary positional embeddings for query and key vectors."""

import torch
from torch import nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        if d_k <= 0 or d_k % 2:
            raise ValueError("d_k must be a positive even integer")
        if theta <= 0 or max_seq_len <= 0:
            raise ValueError("theta and max_seq_len must be positive")

        # Each adjacent feature pair has frequency theta ** (-2 * i / d_k).
        frequencies = theta ** (-torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = positions[:, None] * frequencies[None, :]
        # These tables follow the module's device but need no training or checkpoint storage.
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """Rotate input of shape (..., seq_len, d_k), preserving its shape.

        token_positions has shape (..., seq_len) and selects the rotation
        for each token. Support arbitrary leading batch dimensions.
        """
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        even = x[..., 0::2]
        odd = x[..., 1::2]
        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos
        # Interleave the rotated coordinates to restore the original feature layout.
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2).to(x.dtype)
```

**测试命令**

接通 `run_rope` 后验证旋转位置编码。

```bash
uv run --frozen python -m pytest tests/test_model.py::test_rope -q
```

### File: cs336_basics/nn_utils.py

对应第一部分第 6、8、9 节。

Softmax 的 `amax`、`sum` 使用同一个归一化轴和 `keepdim=True`，减最大值后计算指数与归一化。

先减去末维最大 logit，用 `gather(-1, targets.unsqueeze(-1))` 取得真实类别的 shifted logit。归一化项用 `exp().sum().log()`，两者作差后平均。

低精度 logits 转 float32，float64 输入保留精度。目标张量形状应等于 logits 去掉词表轴后的形状。先 softmax 再 log 会增加小概率下溢风险。

梯度裁剪先收集非空梯度，计算一个全局缩放系数，再对所有梯度原地 `mul_`。梯度累积期间不逐次裁剪，只在一次完整更新之前裁剪。

```python
"""Numerically stable probability and gradient utilities."""

import torch


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    shifted = x - x.amax(dim=dim, keepdim=True)
    exp = shifted.exp()
    return exp / exp.sum(dim=dim, keepdim=True)


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Retain float64 for numerical checks; upcast low-precision training logits.
    logits = inputs.float() if inputs.dtype in (torch.float16, torch.bfloat16) else inputs
    shifted = logits - logits.amax(dim=-1, keepdim=True)
    correct = shifted.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (shifted.exp().sum(dim=-1).log() - correct).mean()


@torch.no_grad()
def gradient_clipping(parameters, max_l2_norm: float) -> None:
    if max_l2_norm < 0:
        raise ValueError("max_l2_norm must be nonnegative")
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    norm = torch.stack([g.float().square().sum() for g in grads]).sum().sqrt()
    scale = (max_l2_norm / (norm + 1e-6)).clamp(max=1.0)
    for grad in grads:
        grad.mul_(scale)
```

**测试命令**

验证 softmax、交叉熵与全局梯度裁剪。

```bash
uv run --frozen python -m pytest tests/test_nn_utils.py -q
```

### File: cs336_basics/attention.py

对应第一部分第 6 节。

SDPA 用 `K.transpose(-2,-1)` 仅交换最后两维；布尔 mask 中 True 允许访问，`masked_fill(~mask,-inf)` 遮挡其他位置。低精度分数转 float32 计算 softmax，再转为 V 的 dtype。

多头投影后的张量通过 `reshape` 拆开 head 轴，再 `transpose(-3,-2)` 得到 `(...,H,S,d_k)`。Q/K 应用 RoPE；带批次的位置张量插入一个 head 广播轴。一张包含对角线的下三角布尔矩阵实现因果遮挡。最后逆变换头维并调用 `output_proj`，无需逐头 Python 循环。

```python
"""Causal multi-head attention using basic tensor operations."""

import math

import torch
from torch import nn

from cs336_basics.linear import Linear
from cs336_basics.nn_utils import softmax
from cs336_basics.rope import RotaryPositionalEmbedding


def scaled_dot_product_attention(Q, K, V, mask=None):
    scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    # The caller must allow at least one key per query.
    probabilities = softmax(scores.float() if scores.dtype in (torch.float16, torch.bfloat16) else scores)
    return probabilities.to(V.dtype) @ V


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len=2048, theta=10000.0, use_rope=True, device=None, dtype=None):
        super().__init__()
        if num_heads <= 0 or d_model % num_heads:
            raise ValueError("d_model must be divisible by a positive num_heads")
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device) if use_rope else None

    def forward(self, x, token_positions=None):
        length = x.shape[-2]
        shape = (*x.shape[:-1], self.num_heads, self.d_k)
        q = self.q_proj(x).reshape(shape).transpose(-3, -2)
        k = self.k_proj(x).reshape(shape).transpose(-3, -2)
        v = self.v_proj(x).reshape(shape).transpose(-3, -2)
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(length, device=x.device)
            # Insert the head axis; shared 1D positions already broadcast correctly.
            positions = token_positions.unsqueeze(-2) if token_positions.ndim > 1 else token_positions
            q, k = self.rope(q, positions), self.rope(k, positions)
        mask = torch.ones(length, length, device=x.device, dtype=torch.bool).tril()
        attended = scaled_dot_product_attention(q, k, v, mask)
        return self.output_proj(attended.transpose(-3, -2).reshape(x.shape))
```

**测试命令**

需完成 Linear、RoPE 和 softmax；包含普通、多批次及带 RoPE 的注意力测试。

```bash
uv run --frozen python -m pytest tests/test_model.py -k "scaled_dot_product_attention or multihead_self_attention" -q
```

### File: cs336_basics/transformer.py

对应第一部分第 4、7 节。

`TransformerBlock` 按 `norm_style` 选择 pre、post 或 none 的操作顺序。none 不创建块内 RMSNorm；`use_rope=False` 不创建 RoPE；`ffn_type` 在 SwiGLU 和 SiLU 前馈之间切换。

`TransformerLM` 用 `nn.ModuleList` 注册多个 block。输入经过 `token_embeddings`，最后经过 `ln_final` 和 `lm_head`；none 变体省去最终 RMSNorm。返回 logits，不提前 softmax；输入超过 `context_length` 时抛出异常。

同文件的 `SiLUFeedForward` 提供去除门控的消融结构，仅保留 `w1`、SiLU 和 `w2`。

```python
"""Decoder-only Transformer and controlled architecture ablations."""

from torch import nn

from cs336_basics.attention import MultiHeadSelfAttention
from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.swiglu import SwiGLU, silu


class SiLUFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x):
        return self.w2(silu(self.w1(x)))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        max_seq_len,
        theta=10000.0,
        norm_style="pre",
        use_rope=True,
        ffn_type="swiglu",
        device=None,
        dtype=None,
    ):
        super().__init__()
        if norm_style not in {"pre", "post", "none"} or ffn_type not in {"swiglu", "silu"}:
            raise ValueError("Invalid norm_style or ffn_type")
        self.norm_style = norm_style
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta, use_rope, device, dtype)
        if norm_style != "none":
            self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
            self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        cls = SwiGLU if ffn_type == "swiglu" else SiLUFeedForward
        self.ffn = cls(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x, token_positions=None):
        if self.norm_style == "pre":
            x = x + self.attn(self.ln1(x), token_positions)
            return x + self.ffn(self.ln2(x))
        if self.norm_style == "post":
            x = self.ln1(x + self.attn(x, token_positions))
            return self.ln2(x + self.ffn(x))
        x = x + self.attn(x, token_positions)
        return x + self.ffn(x)


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        context_length,
        d_model,
        num_layers,
        num_heads,
        d_ff=None,
        rope_theta=10000.0,
        norm_style="pre",
        use_rope=True,
        ffn_type="swiglu",
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.context_length = context_length
        if d_ff is None:
            d_ff = max(64, round((8 * d_model / 3) / 64) * 64) if ffn_type == "swiglu" else 4 * d_model
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model, num_heads, d_ff, context_length, rope_theta, norm_style, use_rope, ffn_type, device, dtype
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype) if norm_style != "none" else None
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids):
        if token_ids.shape[-1] > self.context_length:
            raise ValueError("Input exceeds context_length")
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x)
        if self.ln_final is not None:
            x = self.ln_final(x)
        return self.lm_head(x)
```

**测试命令**

需完成基础层、前馈与注意力；验证 block、语言模型和较短输入。

```bash
uv run --frozen python -m pytest tests/test_model.py -k "transformer" -q
```

### File: cs336_basics/optimizer.py

对应第一部分第 9 节。

AdamW 在 `self.state[p]` 中保存每个参数的 t/m/v。`mul_`、`add_` 更新一阶矩，`addcmul_` 更新二阶矩，`addcdiv_` 更新参数，再执行权重衰减。整个更新位于 `no_grad` 中；没有梯度的参数跳过，closure 在启用梯度的上下文中执行。

调度器按当前 step 选择分段。训练从 step=0 开始，因此最后一次实际更新使用 `steps-1` 的学习率；函数在 `steps` 处到达最小值。

```python
"""AdamW and a cosine learning rate schedule with linear warmup."""

import math

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0 or eps <= 0 or weight_decay < 0 or any(not 0 <= b < 1 for b in betas):
            raise ValueError("Invalid AdamW hyperparameters")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("Sparse gradients are not supported")
                state = self.state[p]
                if not state:
                    state.update(t=0, m=torch.zeros_like(p), v=torch.zeros_like(p))
                state["t"] += 1
                t, m, v = state["t"], state["m"], state["v"]
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                rate = group["lr"] * math.sqrt(1 - b2**t) / (1 - b1**t)
                p.addcdiv_(m, v.sqrt().add_(group["eps"]), value=-rate)
                # Follow the assignment pseudocode: decay after the adaptive update.
                p.mul_(1 - group["lr"] * group["weight_decay"])
        return loss


def get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    if warmup_iters < 0 or cosine_cycle_iters <= warmup_iters:
        raise ValueError("Require 0 <= warmup_iters < cosine_cycle_iters")
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it >= cosine_cycle_iters:
        return min_learning_rate
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    return min_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (max_learning_rate - min_learning_rate)
```

**测试命令**

验证 AdamW 多步更新和学习率调度边界。

```bash
uv run --frozen python -m pytest tests/test_optimizer.py -q
```

### File: cs336_basics/data.py

对应第一部分第 10 节。

`get_batch` 在 NumPy 中生成起点和连续偏移索引，仅为抽到的窗口分配数组。磁盘 token 转为 int64，再通过 `.to(device)` 搬到模型设备。

```python
"""Sample next-token prediction batches without loading the entire dataset."""

import numpy as np
import torch


def get_batch(dataset, batch_size, context_length, device):
    if dataset.ndim != 1 or len(dataset) <= context_length or batch_size <= 0 or context_length <= 0:
        raise ValueError("Need a 1D dataset longer than context_length and positive batch dimensions")
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    indices = starts[:, None] + np.arange(context_length)[None, :]
    x = torch.from_numpy(np.asarray(dataset[indices], dtype=np.int64)).to(device)
    y = torch.from_numpy(np.asarray(dataset[indices + 1], dtype=np.int64)).to(device)
    return x, y
```

**测试命令**

验证随机起点、标签右移、输出形状和设备参数。

```bash
uv run --frozen python -m pytest tests/test_data.py -q
```

### File: cs336_basics/checkpoint.py

对应第一部分第 10 节。`save_checkpoint` 保存模型、优化器与已完成步数，支持文件路径和二进制文件对象。`load_checkpoint` 先将数据加载到 CPU，再交给模型和优化器恢复状态，返回保存的步数。随机数状态与配置由训练入口进一步管理。

```python
"""Save and restore model, optimizer, and completed training steps."""

import torch


def save_checkpoint(model, optimizer, iteration, out):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}, out)


def load_checkpoint(src, model, optimizer):
    state = torch.load(src, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    return state["iteration"]
```

**测试命令**

需完成 AdamW 及其适配器；验证模型与优化器状态保存恢复。

```bash
uv run --frozen python -m pytest tests/test_serialization.py -q
```

### File: cs336_basics/train.py

对应第一部分第 10 节。

每次更新先 `zero_grad`；每个 microbatch 将 loss 除以累积次数后 `backward`；全部完成后裁剪并 `step`。CUDA autocast 使用 bfloat16，参数及 AdamW 状态保持 float32。

通用检查点保存模型、优化器和已完成步数。训练入口还保存配置、CPU/CUDA/NumPy RNG 与累计时间，临时文件写完后原子替换目标文件。恢复 RNG 使后续随机采样衔接；只恢复模型权重不能完整恢复训练。

`evaluate` 用固定随机采样快速监控并恢复原 RNG；`evaluate_full` 遍历不重叠窗口，累计每个窗口的 token 加权损失，包含尾部。评估后恢复原训练模式。

日志记录 step、累计 token 数、loss、学习率与墙钟时间。命令行入口通过配置构造模型，支持训练、断点恢复和全量验证。

```python
"""Configurable training with memmap data, accumulation, JSONL logs, and resume."""

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from cs336_basics.data import get_batch
from cs336_basics.nn_utils import cross_entropy, gradient_clipping
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule
from cs336_basics.transformer import TransformerLM


def save_training_checkpoint(path, model, optimizer, step, config, elapsed):
    rng = np.random.get_state()
    state = dict(
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        iteration=step,
        config=config,
        elapsed_s=elapsed,
        torch_rng=torch.get_rng_state(),
        numpy_rng=[rng[0], rng[1].tolist(), rng[2], rng[3], rng[4]],
    )
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    temporary = Path(str(path) + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


@torch.no_grad()
def evaluate(model, dataset, batch_size, context_length, device, batches, amp=False):
    was_training = model.training
    rng = np.random.get_state()
    model.eval()
    np.random.seed(12345)
    try:
        losses = []
        for _ in range(batches):
            x, y = get_batch(dataset, batch_size, context_length, device)
            with torch.autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext():
                loss = cross_entropy(model(x), y)
            losses.append(loss.item())
        return sum(losses) / len(losses)
    finally:
        np.random.set_state(rng)
        model.train(was_training)


@torch.no_grad()
def evaluate_full(model, dataset, device):
    """Evaluate every next-token target once with non-overlapping context windows."""
    was_training = model.training
    model.eval()
    total_loss, total_tokens = 0.0, 0
    try:
        for start in range(0, len(dataset) - 1, model.context_length):
            stop = min(start + model.context_length, len(dataset) - 1)
            x = torch.tensor(np.asarray(dataset[start:stop], dtype=np.int64), device=device).unsqueeze(0)
            y = torch.tensor(np.asarray(dataset[start + 1 : stop + 1], dtype=np.int64), device=device).unsqueeze(0)
            total_loss += cross_entropy(model(x), y).item() * y.numel()
            total_tokens += y.numel()
        if not total_tokens:
            raise ValueError("Validation requires at least two tokens")
        return total_loss / total_tokens
    finally:
        model.train(was_training)


def train(config, output_dir, resume=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if resume is None and (output / "metrics.jsonl").exists():
        raise ValueError("Use a new output directory or --resume")
    device = config.get("device", "cpu")
    amp = config.get("amp", False)
    if amp and not str(device).startswith("cuda"):
        raise ValueError("This trainer supports bfloat16 autocast on CUDA only")
    torch.set_num_threads(config.get("num_threads", 4))
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TransformerLM(**config["model"], device=device)
    optimizer = AdamW(model.parameters(), **config.get("optimizer", {}))
    train_data = np.memmap(config["train_data"], dtype=config.get("data_dtype", "<u2"), mode="r")
    valid_data = np.memmap(config["valid_data"], dtype=config.get("data_dtype", "<u2"), mode="r")
    batch_size = config.get("batch_size", 4)
    accumulation = config.get("grad_accum_steps", 1)
    steps = config["steps"]
    context = config["model"]["context_length"]
    eval_batches = config.get("eval_batches", 10)
    if min(batch_size, accumulation, steps, eval_batches, config.get("eval_interval", 100)) <= 0:
        raise ValueError("Training counts and intervals must be positive")
    start_step, elapsed_before = 0, 0.0
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if state["config"] != config:
            raise ValueError("Resume must use the same config; steps is the original total target")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step, elapsed_before = state["iteration"], state["elapsed_s"]
        torch.set_rng_state(state["torch_rng"])
        rng = state["numpy_rng"]
        np.random.set_state((rng[0], np.asarray(rng[1], dtype=np.uint32), rng[2], rng[3], rng[4]))
        if "cuda_rng" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    (output / "config.json").write_text(json.dumps(config, indent=2))
    max_lr = config.get("optimizer", {}).get("lr", 1e-3)
    warmup = config.get("warmup_steps", 0)
    if not 0 <= warmup < steps:
        raise ValueError("Require 0 <= warmup_steps < steps")
    start = time.perf_counter()
    with open(output / "metrics.jsonl", "a") as log:
        for step in range(start_step, steps):
            lr = get_lr_cosine_schedule(step, max_lr, config.get("min_lr", max_lr * 0.1), warmup, steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for _ in range(accumulation):
                x, y = get_batch(train_data, batch_size, context, device)
                with torch.autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext():
                    loss = cross_entropy(model(x), y)
                if not torch.isfinite(loss):
                    log.write(json.dumps(dict(step=step, status="nonfinite_loss")) + "\n")
                    log.flush()
                    raise FloatingPointError("Nonfinite loss; inspect the learning rate")
                (loss / accumulation).backward()
                total_loss += loss.item() / accumulation
            gradient_clipping(model.parameters(), config.get("max_grad_norm", 1.0))
            optimizer.step()
            record = dict(
                step=step + 1,
                tokens=(step + 1) * batch_size * accumulation * context,
                train_loss=total_loss,
                lr=lr,
                elapsed_s=elapsed_before + time.perf_counter() - start,
            )
            if (step + 1) % config.get("eval_interval", 100) == 0 or step + 1 == steps:
                record["valid_loss"] = evaluate(model, valid_data, batch_size, context, device, eval_batches, amp)
            log.write(json.dumps(record) + "\n")
            log.flush()
            if "valid_loss" in record:
                print(json.dumps(record), flush=True)
                save_training_checkpoint(
                    output / "checkpoint.pt",
                    model,
                    optimizer,
                    step + 1,
                    config,
                    elapsed_before + time.perf_counter() - start,
                )
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("output_dir")
    parser.add_argument("--resume")
    parser.add_argument("--evaluate-checkpoint")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.evaluate_checkpoint:
        device = config.get("device", "cpu")
        torch.set_num_threads(config.get("num_threads", 4))
        model = TransformerLM(**config["model"], device=device)
        state = torch.load(args.evaluate_checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        data = np.memmap(config["valid_data"], dtype=config.get("data_dtype", "<u2"), mode="r")
        report = dict(valid_loss=evaluate_full(model, data, device), tokens=len(data) - 1)
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "validation.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report))
    else:
        train(config, args.output_dir, args.resume)


if __name__ == "__main__":
    main()
```

**测试命令**

需完成模型、损失、优化器及数据采样；验证小批次过拟合、恢复一致性和验证集尾部计权。

```bash
uv run --frozen python -m pytest tests/test_training_integration.py -k "overfit_small_batch or checkpoint_resume or full_validation" -q
```

### File: cs336_basics/generate.py

对应第一部分第 11 节。

`sample_next_token` 用 `sort` 排序概率，用 `cumsum - sorted_probs >= top_p` 找出应删除的位置。这个条件保留首次达到阈值的 token；重新归一化后用 `torch.multinomial` 采样。温度为 0 时直接 `argmax`。

`generate` 每步只取最后 `context_length` 个 ID 作为输入，将新 ID 追加到序列，遇 EOS 或达到上限停止。返回值包含 prompt。生成使用 `no_grad` 和 eval，结束后恢复原模式。没有 KV cache，每步重算保留上下文。

```python
"""Autoregressive decoding with temperature, nucleus sampling, and EOS stopping."""

import argparse

import torch

from cs336_basics.nn_utils import softmax
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer import TransformerLM


def sample_next_token(logits, temperature=1.0, top_p=1.0):
    if temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("Require temperature >= 0 and 0 < top_p <= 1")
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    probabilities = softmax(logits.float() / temperature)
    sorted_probs, indices = probabilities.sort(descending=True, dim=-1)
    # Keep the token that first brings cumulative probability to the threshold.
    remove = sorted_probs.cumsum(-1) - sorted_probs >= top_p
    sorted_probs = sorted_probs.masked_fill(remove, 0)
    sorted_probs = sorted_probs / sorted_probs.sum(-1, keepdim=True)
    selected = torch.multinomial(sorted_probs, 1)
    return indices.gather(-1, selected)


@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens=256, temperature=1.0, top_p=1.0, eos_id=None):
    if not prompt_ids or max_new_tokens < 0:
        raise ValueError("Provide a nonempty prompt and nonnegative max_new_tokens")
    device = next(model.parameters()).device
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            logits = model(ids[:, -model.context_length :])[:, -1, :]
            token = sample_next_token(logits, temperature, top_p)
            ids = torch.cat((ids, token), dim=-1)
            if eos_id is not None and token.item() == eos_id:
                break
    finally:
        model.train(was_training)
    return ids[0].tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("tokenizer_dir")
    parser.add_argument("prompt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = TransformerLM(**state["config"]["model"], device=args.device)
    model.load_state_dict(state["model"])
    tokenizer = Tokenizer.from_files(
        args.tokenizer_dir + "/vocab.hex.json", args.tokenizer_dir + "/merges.hex.json", ["<|endoftext|>"]
    )
    prompt = tokenizer.encode(args.prompt) or [tokenizer.token_to_id[b"<|endoftext|>"]]
    ids = generate(
        model, prompt, args.max_new_tokens, args.temperature, args.top_p, tokenizer.token_to_id[b"<|endoftext|>"]
    )
    print(tokenizer.decode(ids))


if __name__ == "__main__":
    main()
```

**测试命令**

需完成 Transformer 与 softmax；验证 greedy、top-p 候选集合和 EOS 停止。

```bash
uv run --frozen python -m pytest tests/test_training_integration.py::test_sampling_nucleus_and_eos -q
```

### File: cs336_basics/experiments.py

对应第一部分第 12 节。

`accounting` 计算参数、矩阵乘法 FLOPs 与给定保存约定下的内存估计；`sgd_example` 演示学习率对简单二次函数优化的影响。`write_configs` 生成控制变量配置，`export_logs` 和 `plot_logs` 转换日志，不参与模型计算。

```python
"""Generate controlled experiment configs, export logs, and estimate resource costs."""

import argparse
import copy
import csv
from html import escape
import json
import math
from pathlib import Path

import torch


def accounting(vocab_size=50257, seq_len=1024, layers=48, d_model=1600, heads=25, d_ff=4288):
    v, s, n_layers, d, h, f = vocab_size, seq_len, layers, d_model, heads, d_ff
    parameters = 2 * v * d + n_layers * (4 * d * d + 3 * d * f + 2 * d) + d
    flops = dict(
        projections=n_layers * 8 * s * d * d,
        attention=n_layers * 4 * s * s * d,
        feedforward=n_layers * 6 * s * d * f,
        output=2 * s * d * v,
    )
    # Count one output tensor per listed operation; exclude allocator/workspace overhead.
    activations_per_batch = n_layers * (8 * s * d + 4 * s * f + 2 * h * s * s) + s * d + 2 * s * v
    return dict(
        parameters=parameters,
        parameter_bytes=4 * parameters,
        forward_flops=sum(flops.values()),
        flops=flops,
        persistent_training_bytes=16 * parameters,
        activation_bytes_per_batch=4 * activations_per_batch,
        max_batch_80GB=max(0, math.floor((80e9 - 16 * parameters) / (4 * activations_per_batch))),
        adamw_flops_approx=13 * parameters,
        h100_hours_400k_steps_batch1024=(3 * sum(flops.values()) * 1024 + 13 * parameters)
        * 400000
        / (495e12 * 0.5)
        / 3600,
    )


def sgd_example():
    records = {}
    for lr in (1, 10, 100, 1000):
        torch.manual_seed(42)
        weights = torch.nn.Parameter(5 * torch.randn(10, 10))
        losses = []
        for t in range(10):
            weights.grad = None
            loss = weights.square().mean()
            losses.append(loss.item())
            loss.backward()
            with torch.no_grad():
                weights.add_(weights.grad, alpha=-lr / math.sqrt(t + 1))
        records[lr] = losses
    return records


def write_configs(base_path, output_dir):
    base = json.loads(Path(base_path).read_text())
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    variants = {"baseline": copy.deepcopy(base)}
    for lr in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2):
        c = copy.deepcopy(base)
        c.setdefault("optimizer", {})["lr"] = lr
        c["min_lr"] = lr * 0.1
        variants[f"lr_{lr:g}"] = c
    for norm in ("none", "post"):
        c = copy.deepcopy(base)
        c["model"]["norm_style"] = norm
        variants[f"norm_{norm}"] = c
    c = copy.deepcopy(base)
    c["model"]["norm_style"] = "none"
    c.setdefault("optimizer", {})["lr"] = base.get("optimizer", {}).get("lr", 1e-3) / 10
    c["min_lr"] = c["optimizer"]["lr"] * 0.1
    variants["norm_none_lower_lr"] = c
    c = copy.deepcopy(base)
    c["model"]["use_rope"] = False
    variants["nope"] = c
    c = copy.deepcopy(base)
    c["model"].update(ffn_type="silu", d_ff=4 * c["model"]["d_model"])
    variants["silu"] = c
    effective_batch = base.get("batch_size", 4) * base.get("grad_accum_steps", 1)
    total_examples = base["steps"] * effective_batch
    # Change accumulation instead of resident microbatch memory; keep the token budget fixed.
    for factor in (1, 2, 4):
        c = copy.deepcopy(base)
        c["grad_accum_steps"] = base.get("grad_accum_steps", 1) * factor
        if total_examples % (effective_batch * factor):
            raise ValueError("Choose a baseline step count divisible by four")
        c["steps"] = total_examples // (effective_batch * factor)
        c["warmup_steps"] = min(c["steps"] - 1, base.get("warmup_steps", 0) // factor)
        variants[f"batch_x{factor}"] = c
    for name, config in variants.items():
        (output / f"{name}.json").write_text(json.dumps(config, indent=2))
    return list(variants)


def export_logs(paths, output_path):
    rows = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            record = json.loads(line)
            rows.append(dict(run=str(Path(path).parent), **record))
    fields = ["run", "step", "tokens", "elapsed_s", "train_loss", "valid_loss", "lr", "status"]
    with open(output_path, "w", newline="") as dest:
        writer = csv.DictWriter(dest, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_logs(paths, output_path):
    """Write standalone SVG validation-loss curves against steps and wall time."""
    runs = []
    for path in paths:
        rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
        rows = [r for r in rows if "valid_loss" in r and math.isfinite(r["valid_loss"])]
        if rows:
            runs.append((str(Path(path).parent), rows))
    if not runs:
        raise ValueError("No finite validation measurements")
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c"]
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="540">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for panel, xkey in enumerate(("step", "elapsed_s")):
        left, top, width, height = 70 + panel * 540, 45, 440, 320
        xmax = max(r[xkey] for _, rows in runs for r in rows) or 1
        ys = [r["valid_loss"] for _, rows in runs for r in rows]
        ymin, ymax = min(ys), max(ys)
        pad = max((ymax - ymin) * 0.1, 0.01)
        ymin, ymax = ymin - pad, ymax + pad
        svg.append(f'<text x="{left}" y="25">Validation loss vs {xkey}</text>')
        for tick in range(6):
            xp, yp = left + width * tick / 5, top + height * tick / 5
            svg.append(f'<path d="M {left} {yp} H {left + width}" stroke="#ddd"/>')
            svg.append(
                f'<text x="{left - 50}" y="{yp + 4}" font-size="12">{ymax - (ymax - ymin) * tick / 5:.3f}</text>'
            )
            svg.append(f'<text x="{xp}" y="{top + height + 20}" font-size="12">{xmax * tick / 5:.0f}</text>')
        for i, (_, rows) in enumerate(runs):
            points = " ".join(
                f"{left + r[xkey] / xmax * width:.2f},{top + (ymax - r['valid_loss']) / (ymax - ymin) * height:.2f}"
                for r in rows
            )
            svg.append(f'<polyline points="{points}" fill="none" stroke="{colors[i % len(colors)]}" stroke-width="2"/>')
    for i, (name, _) in enumerate(runs):
        svg.append(
            f'<text x="70" y="{410 + i * 18}" font-size="12" fill="{colors[i % len(colors)]}">{escape(name)}</text>'
        )
    svg.append("</svg>")
    Path(output_path).write_text("\n".join(svg))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("accounting")
    sub.add_parser("sgd")
    configs = sub.add_parser("configs")
    configs.add_argument("base")
    configs.add_argument("output_dir")
    export = sub.add_parser("export")
    export.add_argument("output")
    export.add_argument("logs", nargs="+")
    plot = sub.add_parser("plot")
    plot.add_argument("output")
    plot.add_argument("logs", nargs="+")
    args = parser.parse_args()
    if args.command == "accounting":
        rows = {}
        for name, n_layers, d, h in [
            ("small", 12, 768, 12),
            ("medium", 24, 1024, 16),
            ("large", 36, 1280, 20),
            ("xl", 48, 1600, 25),
        ]:
            rows[name] = accounting(layers=n_layers, d_model=d, heads=h, d_ff=round(8 * d / 3 / 64) * 64)
        rows["xl_long"] = accounting(seq_len=16384)
        print(json.dumps(rows, indent=2))
    elif args.command == "sgd":
        print(json.dumps(sgd_example(), indent=2))
    elif args.command == "configs":
        print(write_configs(args.base, args.output_dir))
    elif args.command == "plot":
        plot_logs(args.logs, args.output)
    else:
        export_logs(args.logs, args.output)


if __name__ == "__main__":
    main()
```

**测试命令**

验证配置变体、token 预算、资源计数及 CSV/SVG 导出。

```bash
uv run --frozen python -m pytest tests/test_training_integration.py::test_experiment_configs_and_exports -q
```

### File: cs336_basics/__init__.py

包初始化仅查询已安装包的版本元数据，不包含算法计算。

```python
import importlib.metadata

try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    pass
```

**测试命令**

此文件仅管理包元数据，使用导入检查。

```bash
uv run --frozen python -c "import cs336_basics; print(cs336_basics.__name__)"
```

### File: tests/adapters.py

适配器将课程测试连接到实现：导入组件、构造模块、装载测试指定权重并调用。保持测试要求的参数名、维度与返回值，避免在适配器中重复实现算法。

```python
from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """

    from cs336_basics.linear import Linear

    layer = Linear(d_in, d_out, device=weights.device, dtype=weights.dtype)
    layer.load_state_dict({"weight": weights})
    return layer(in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """

    from cs336_basics.embedding import Embedding

    layer = Embedding(vocab_size, d_model, device=weights.device, dtype=weights.dtype)
    layer.load_state_dict({"weight": weights})
    return layer(token_ids)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    from cs336_basics.swiglu import SwiGLU

    layer = SwiGLU(d_model, d_ff, device=w1_weight.device, dtype=w1_weight.dtype)
    layer.load_state_dict({"w1.weight": w1_weight, "w2.weight": w2_weight, "w3.weight": w3_weight})
    return layer(in_features)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... keys d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    from cs336_basics.attention import scaled_dot_product_attention

    return scaled_dot_product_attention(Q, K, V, mask)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    from cs336_basics.attention import MultiHeadSelfAttention

    layer = MultiHeadSelfAttention(
        d_model, num_heads, use_rope=False, device=q_proj_weight.device, dtype=q_proj_weight.dtype
    )
    layer.load_state_dict(
        {
            "q_proj.weight": q_proj_weight,
            "k_proj.weight": k_proj_weight,
            "v_proj.weight": v_proj_weight,
            "output_proj.weight": o_proj_weight,
        }
    )
    return layer(in_features)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    from cs336_basics.attention import MultiHeadSelfAttention

    layer = MultiHeadSelfAttention(
        d_model, num_heads, max_seq_len, theta, device=q_proj_weight.device, dtype=q_proj_weight.dtype
    )
    layer.load_state_dict(
        {
            "q_proj.weight": q_proj_weight,
            "k_proj.weight": k_proj_weight,
            "v_proj.weight": v_proj_weight,
            "output_proj.weight": o_proj_weight,
        }
    )
    return layer(in_features, token_positions)


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    from cs336_basics.rope import RotaryPositionalEmbedding

    layer = RotaryPositionalEmbedding(theta, d_k, max_seq_len, device=in_query_or_key.device)
    return layer(in_query_or_key, token_positions)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    from cs336_basics.transformer import TransformerBlock

    sample = next(iter(weights.values()))
    layer = TransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta, device=sample.device, dtype=sample.dtype)
    layer.load_state_dict(weights)
    return layer(in_features)


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    from cs336_basics.transformer import TransformerLM

    sample = next(iter(weights.values()))
    model = TransformerLM(
        vocab_size,
        context_length,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        rope_theta,
        device=sample.device,
        dtype=sample.dtype,
    )
    model.load_state_dict(weights)
    return model(in_indices)


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    from cs336_basics.rmsnorm import RMSNorm

    layer = RMSNorm(d_model, eps, device=weights.device, dtype=weights.dtype)
    layer.load_state_dict({"weight": weights})
    return layer(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    from cs336_basics.swiglu import silu

    return silu(in_features)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    from cs336_basics.data import get_batch

    return get_batch(dataset, batch_size, context_length, device)


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    from cs336_basics.nn_utils import softmax

    return softmax(in_features, dim)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    from cs336_basics.nn_utils import cross_entropy

    return cross_entropy(inputs, targets)


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    from cs336_basics.nn_utils import gradient_clipping

    gradient_clipping(parameters, max_l2_norm)


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    from cs336_basics.optimizer import AdamW

    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    from cs336_basics.optimizer import get_lr_cosine_schedule

    return get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    from cs336_basics.checkpoint import save_checkpoint

    save_checkpoint(model, optimizer, iteration, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    from cs336_basics.checkpoint import load_checkpoint

    return load_checkpoint(src, model, optimizer)


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    from cs336_basics.tokenizer import Tokenizer

    return Tokenizer(vocab, merges, special_tokens)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    from cs336_basics.bpe import train_bpe

    return train_bpe(input_path, vocab_size, special_tokens)
```

**测试命令**

全部组件完成后运行完整测试集。

```bash
uv run --frozen python -m pytest tests -q
```

### File: tests/test_bpe_training_edges.py

这些用例检查读取块边界、Unicode 和特殊 token 是否改变预分词结果，并覆盖重叠 pair、同频比较、空语料及词表容量校验。

```python
"""Additional BPE boundary checks; the course tests are left unchanged."""

import regex
import pytest

from cs336_basics.bpe import PAT, _iter_pretokens, train_bpe


@pytest.mark.parametrize("chunk_size", [1, 2, 7, 31])
@pytest.mark.parametrize("specials", [[], ["<end>"], ["<end>", "<end>long"]])
def test_chunk_boundaries_preserve_pretokens(tmp_path, chunk_size, specials):
    text = "你好 café we're\t \n\n<end>long  hello<end>world\r\n!  "
    path = tmp_path / "corpus.txt"
    path.write_bytes(text.encode("utf-8"))
    if specials:
        separator = "|".join(regex.escape(s) for s in sorted(specials, key=len, reverse=True))
        documents = regex.split(separator, text)
    else:
        documents = [text]
    expected = [match.group() for document in documents for match in regex.finditer(PAT, document)]
    assert list(_iter_pretokens(path, specials, chunk_size)) == expected


def test_overlapping_pair_occurrences(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("aaaaa", encoding="utf-8")
    _, merges = train_bpe(path, 260, [])
    assert merges == [(b"a", b"a"), (b"aa", b"aa"), (b"aaaa", b"a")]


def test_frequency_tie_uses_byte_lexicographic_order(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("ab<end>ac", encoding="utf-8")
    _, merges = train_bpe(path, 258, ["<end>"])
    assert merges == [(b"a", b"c")]


def test_empty_corpus_and_duplicate_specials(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("", encoding="utf-8")
    vocab, merges = train_bpe(path, 300, ["<end>", "<end>"])
    assert len(vocab) == 257
    assert vocab[256] == b"<end>"
    assert merges == []


def test_vocab_budget_must_cover_initial_vocabulary(tmp_path):
    with pytest.raises(ValueError, match="vocab_size"):
        train_bpe(tmp_path / "unused.txt", 256, ["<end>"])
```

**测试命令**

需完成 BPE；直接运行本文件的边界用例。

```bash
uv run --frozen python -m pytest tests/test_bpe_training_edges.py -q
```

### File: tests/test_tokenizer_edges.py

这些用例检查十六进制文件加载、合并优先级、跨 token 的 UTF-8 解码，以及迭代器是否按需消费输入。

```python
"""Cover file loading, merge priority, and generator behavior."""

import json

from cs336_basics.tokenizer import Tokenizer


def test_from_hex_files_and_merge_priority(tmp_path):
    vocab = {i: bytes([i]) for i in range(256)}
    vocab.update({256: b"bc", 257: b"ab"})
    vocab_path = tmp_path / "vocab.hex.json"
    merges_path = tmp_path / "merges.hex.json"
    vocab_path.write_text(json.dumps({i: token.hex() for i, token in vocab.items()}))
    merges_path.write_text(json.dumps([[b"b".hex(), b"c".hex()], [b"a".hex(), b"b".hex()]]))
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, ["<end>"])
    assert tokenizer.encode("abc<end>") == [97, 256, 258]
    assert tokenizer.decode([97, 256, 258]) == "abc<end>"


def test_decode_joins_unicode_bytes_before_decoding():
    tokenizer = Tokenizer({i: bytes([i]) for i in range(256)}, [])
    assert tokenizer.decode(list("中".encode())) == "中"
    assert tokenizer.decode([0xFF]) == "\ufffd"


def test_iterable_is_lazy_and_specials_do_not_mutate_input():
    vocab = {i: bytes([i]) for i in range(256)}
    tokenizer = Tokenizer(vocab, [], ["<end>"])
    assert len(vocab) == 256
    consumed = []

    def texts():
        consumed.append(1)
        yield "ab"
        consumed.append(2)
        yield "c"

    ids = tokenizer.encode_iterable(texts())
    assert consumed == []
    assert next(ids) == 97
    assert consumed == [1]
    assert list(ids) == [98, 99]
    assert consumed == [1, 2]
```

**测试命令**

需完成 Tokenizer；直接运行本文件的补充用例。

```bash
uv run --frozen python -m pytest tests/test_tokenizer_edges.py -q
```

### File: tests/test_training_integration.py

这些用例组合验证模型因果性、结构消融、梯度、小批次过拟合、断点恢复一致性、数据编码、采样、验证集尾部计权与实验日志导出。

```python
"""Integration coverage for the reference training and decoding pipeline."""

import json

import numpy as np
import pytest
import torch

from cs336_basics.generate import generate, sample_next_token
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_basics.preprocess import encode_file, iter_documents, train_tokenizer
from cs336_basics.train import train
from cs336_basics.transformer import TransformerLM


@pytest.mark.parametrize(
    "norm_style,use_rope,ffn_type",
    [
        ("pre", True, "swiglu"),
        ("post", True, "swiglu"),
        ("none", True, "swiglu"),
        ("pre", False, "swiglu"),
        ("pre", True, "silu"),
    ],
)
def test_ablation_causality_and_gradients(norm_style, use_rope, ffn_type):
    torch.set_num_threads(1)
    model = TransformerLM(16, 8, 16, 1, 2, 32, norm_style=norm_style, use_rope=use_rope, ffn_type=ffn_type)
    x = torch.tensor([[1, 2, 3, 4]])
    changed = torch.tensor([[1, 2, 9, 8]])
    torch.testing.assert_close(model(x)[:, :2], model(changed)[:, :2])
    cross_entropy(model(x), (x + 1) % 16).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_overfit_small_batch():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = TransformerLM(8, 8, 16, 1, 2, 32)
    opt = AdamW(model.parameters(), lr=0.02, weight_decay=0)
    x = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
    y = (x + 1) % 8
    initial = cross_entropy(model(x), y).item()
    for _ in range(80):
        opt.zero_grad()
        loss = cross_entropy(model(x), y)
        loss.backward()
        opt.step()
    final = cross_entropy(model(x), y).item()
    assert final < 0.05 and final < initial / 10


def test_checkpoint_resume_matches_uninterrupted(tmp_path, monkeypatch):
    import cs336_basics.train as training

    data_path = tmp_path / "tokens.bin"
    np.tile(np.arange(8, dtype="<u2"), 32).tofile(data_path)
    config = dict(
        model=dict(vocab_size=8, context_length=8, d_model=16, num_layers=1, num_heads=2, d_ff=32),
        train_data=str(data_path),
        valid_data=str(data_path),
        steps=4,
        batch_size=2,
        eval_interval=2,
        eval_batches=1,
        num_threads=1,
        grad_accum_steps=2,
    )
    full = train(config, tmp_path / "full")
    original = training.save_training_checkpoint

    def interrupt_after_save(*args, **kwargs):
        original(*args, **kwargs)
        raise InterruptedError("Simulated interruption")

    monkeypatch.setattr(training, "save_training_checkpoint", interrupt_after_save)
    with pytest.raises(InterruptedError):
        train(config, tmp_path / "resumed")
    monkeypatch.setattr(training, "save_training_checkpoint", original)
    resumed = train(config, tmp_path / "resumed", tmp_path / "resumed/checkpoint.pt")
    for key, value in full.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)


def test_preprocess_roundtrip_and_boundaries(tmp_path):
    from cs336_basics.preprocess import load_tokenizer

    source = tmp_path / "source.txt"
    text = "hello\n\n<|endoftext|>world!\r\n" * 3
    source.write_bytes(text.encode())
    assert "".join(iter_documents(source, chunk_size=1)) == text
    tokenizer_dir = tmp_path / "bpe"
    train_tokenizer(source, tokenizer_dir, 270)
    report = encode_file(source, tokenizer_dir, tmp_path / "ids.bin")
    ids = np.memmap(tmp_path / "ids.bin", mode="r", dtype=report["dtype"])
    tokenizer = load_tokenizer(tokenizer_dir)
    assert ids.tolist() == tokenizer.encode(text)
    assert tokenizer.decode(ids.tolist()) == text
    assert json.loads((tmp_path / "ids.bin.json").read_text())["tokens"] == len(ids)


def test_sampling_nucleus_and_eos(monkeypatch):
    def check_distribution(probs, count):
        torch.testing.assert_close(probs, torch.tensor([[0.6 / 0.9, 0.3 / 0.9, 0.0]]))
        return torch.tensor([[1]])

    monkeypatch.setattr(torch, "multinomial", check_distribution)
    assert sample_next_token(torch.tensor([[0.6, 0.3, 0.1]]).log(), top_p=0.8).item() == 1
    assert sample_next_token(torch.tensor([[0.0, 2.0, 1.0]]), temperature=0).item() == 1
    model = TransformerLM(8, 4, 8, 1, 1, 16)
    monkeypatch.setattr(model, "forward", lambda ids: torch.zeros(1, ids.shape[-1], 8))
    result = generate(model, [1, 2], 10, temperature=0, eos_id=0)
    assert result == [1, 2, 0]
    assert model.training


def test_experiment_configs_and_exports(tmp_path):
    from cs336_basics.experiments import accounting, export_logs, plot_logs, write_configs

    base = dict(model=dict(d_model=64), steps=40, batch_size=2, grad_accum_steps=2, warmup_steps=4)
    path = tmp_path / "base.json"
    path.write_text(json.dumps(base))
    names = write_configs(path, tmp_path / "configs")
    assert "nope" in names and "silu" in names
    for name in names:
        c = json.loads((tmp_path / "configs" / f"{name}.json").read_text())
        assert c["steps"] * c["batch_size"] * c["grad_accum_steps"] == 160
    log = tmp_path / "metrics.jsonl"
    log.write_text(json.dumps(dict(step=1, elapsed_s=0.1, valid_loss=2.0)) + "\n")
    export_logs([log], tmp_path / "metrics.csv")
    plot_logs([log], tmp_path / "curve.svg")
    import xml.etree.ElementTree as ET

    assert ET.parse(tmp_path / "curve.svg").getroot().tag.endswith("svg")
    assert accounting()["parameters"] == 1640452800


def test_full_validation_weights_tail_tokens():
    from cs336_basics.train import evaluate_full

    model = TransformerLM(8, 4, 8, 1, 1, 16)
    data = np.arange(7, dtype=np.int64)
    with torch.no_grad():
        first = cross_entropy(model(torch.tensor([[0, 1, 2, 3]])), torch.tensor([[1, 2, 3, 4]])).item()
        tail = cross_entropy(model(torch.tensor([[4, 5]])), torch.tensor([[5, 6]])).item()
    assert evaluate_full(model, data, "cpu") == pytest.approx((first * 4 + tail * 2) / 6)
    assert model.training
```

**测试命令**

需完成全部相关组件；运行完整流程和消融用例。

```bash
uv run --frozen python -m pytest tests/test_training_integration.py -q
```

