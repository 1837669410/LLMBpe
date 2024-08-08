def get_stats(ids, counts=None):
    counts = counts if counts is not None else {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    new_ids = []
    i = 0
    while i < len(ids):
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


class BaseTokenizer:

    def __init__(self):

        self.merges = {} # (int, int) -> int
        self.vocab = {} # int -> bytes

        self.special_tokens = {} # str -> int

        self.pattern = "" # str

    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError

    def encode(self, text):
        raise NotImplementedError

    def decode(self, ids):
        raise NotImplementedError

    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, special_idx in self.special_tokens.items():
            vocab[special_idx] = special.encode("utf-8")
        return vocab

    def save(self, file_prefix):
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            f.write("LLMBpe v1\n")
            f.write(f"{self.pattern}\n")

            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            for p1, p2 in self.merges:
                f.write(f"{p1} {p2}\n")

    def load(self, model_file):
        assert model_file.endswith(".model")

        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, "r", encoding="utf-8") as f:
            # 1、检查版本号
            version = f.readline().strip()
            assert version == "LLMBpe v1"

            # 2、加载pattern
            self.pattern = f.readline().strip()

            # 3、加载special_tokens
            num_special_tokens = int(f.readline().strip())
            for _ in range(num_special_tokens):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)

            # 4、加载merges
            for line in f:
                p1, p2 = map(int, line.split())
                merges[(p1, p2)] = idx
                idx += 1

            self.merges = merges
            self.special_tokens = special_tokens
            self.vocab = self._build_vocab()

