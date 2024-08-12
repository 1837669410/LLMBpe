import regex as re

from .base import BaseTokenizer, get_stats, merge


GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer(BaseTokenizer):

    def __init__(self, dropout=None):
        super().__init__()

        self.dropout = dropout

        self.pattern = GPT4_SPLIT_PATTERN
        self.compiled_pattern = re.compile(self.pattern)

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        text_chunks = re.findall(self.pattern, text)
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            # 统计子词出现的次数
            stats = {}
            for chunk_ids in ids:
                if self.dropout is None:
                    get_stats(chunk_ids, stats)
                else:
                    get_stats(chunk_ids, stats, self.dropout)

            # 统计出现次数最多的子词
            pair = max(stats, key=stats.get)

            # 合并
            idx = 256 + i
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]

            # 保存信息
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            # 打印信息
            if verbose:
                print(f"[{i+1}/{num_merges}] | {pair}->{idx} | {vocab[idx]}出现{stats[pair]}次")

        # 持久化存储
        self.merges = merges
        self.vocab = vocab

    def register_special_tokens(self, special_tokens):
        for special, idx in special_tokens.items():
            self.special_tokens[special] = idx

    def _encode_chunk(self, text_bytes):
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids, dropout=None)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            if pair not in self.merges:
                break

            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def encode(self, text):
        special_pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"
        special_chunks = re.split(special_pattern, text)

        ids = []
        for chunk in special_chunks:
            if chunk in self.special_tokens:
                ids.append(self.special_tokens[chunk])
            else:
                ids.extend(self.encode_ordinary(chunk))
        return ids

    def decode(self, ids):
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            else:
                raise ValueError(f"无效的idx:{idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text


