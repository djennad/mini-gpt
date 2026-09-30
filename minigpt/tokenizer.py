"""Byte-level BPE tokenizer, written from scratch.

This is the same idea GPT-2/GPT-4 use: text is turned into UTF-8 bytes, and the
most frequent adjacent pairs are repeatedly merged into new tokens. Because it
works on bytes it handles any language (Arabic, English, emoji...) and never
produces an "unknown" token.
"""

import heapq
import json
import re
from collections import Counter, defaultdict

# Arabic diacritics (harakat) are combining marks, which `\w` does not match,
# so they are listed explicitly to keep them attached to their word.
_MARKS = "ؐ-ًؚ-ٰٟۖ-ۭ"

# Pre-split text into words, numbers, punctuation and whitespace so merges never
# cross word boundaries (similar to GPT-2's pattern, but Unicode-aware).
SPLIT_PATTERN = re.compile(
    rf"'(?:s|t|re|ve|m|ll|d)"
    rf"| ?(?:[^\W\d_]|[{_MARKS}])+"
    rf"| ?\d{{1,3}}"
    rf"| ?[^\s\w{_MARKS}]+"
    rf"|\s+(?!\S)|\s+"
)

SPECIAL_TOKENS = ["<|endoftext|>", "<|user|>", "<|assistant|>", "<|end|>"]

_CACHE_LIMIT = 100_000


def _merge(ids, pair, new_id):
    """Replace every occurrence of `pair` in `ids` with `new_id`."""
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class Tokenizer:
    def __init__(self, merges=None, special_tokens=SPECIAL_TOKENS):
        # merges: {(id_a, id_b): new_id}, with new ids assigned from 256 upwards
        self.merges = dict(merges or {})
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), new_id in sorted(self.merges.items(), key=lambda kv: kv[1]):
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
        first_special = 256 + len(self.merges)
        self.special_tokens = {tok: first_special + i for i, tok in enumerate(special_tokens)}
        self.special_names = {i: tok for tok, i in self.special_tokens.items()}
        self._special_re = (
            re.compile("(" + "|".join(re.escape(t) for t in special_tokens) + ")")
            if special_tokens else None
        )
        self._cache = {}

    @property
    def vocab_size(self):
        return 256 + len(self.merges) + len(self.special_tokens)

    # ------------------------------------------------------------------ training

    @classmethod
    def train(cls, text, vocab_size, special_tokens=SPECIAL_TOKENS, verbose=False):
        """Learn up to `vocab_size - 256 - len(special_tokens)` merges from `text`."""
        num_merges = vocab_size - 256 - len(special_tokens)
        if num_merges < 0:
            raise ValueError(f"vocab_size must be at least {256 + len(special_tokens)}")

        parts = [text]
        if special_tokens:
            special_re = re.compile("|".join(re.escape(t) for t in special_tokens))
            parts = special_re.split(text)
        counts = Counter()
        for part in parts:
            counts.update(SPLIT_PATTERN.findall(part))

        words = [list(w.encode("utf-8")) for w in counts]
        freqs = list(counts.values())

        # Count every adjacent pair once up front, then update counts
        # incrementally: only words containing the merged pair change.
        pair_counts = Counter()
        pair_words = defaultdict(set)
        for i, ids in enumerate(words):
            for pair in zip(ids, ids[1:]):
                pair_counts[pair] += freqs[i]
                pair_words[pair].add(i)
        heap = [(-count, pair) for pair, count in pair_counts.items()]
        heapq.heapify(heap)

        merges = {}
        while len(merges) < num_merges and heap:
            neg_count, pair = heapq.heappop(heap)
            if pair_counts.get(pair) != -neg_count:
                continue  # stale heap entry, the count has changed since
            if -neg_count < 2:
                break  # nothing left that is worth merging
            new_id = 256 + len(merges)
            merges[pair] = new_id

            changed = set()
            for i in pair_words.pop(pair):
                ids, freq = words[i], freqs[i]
                new_ids = _merge(ids, pair, new_id)
                if len(new_ids) == len(ids):
                    continue
                for p in zip(ids, ids[1:]):
                    pair_counts[p] -= freq
                    changed.add(p)
                for p in zip(new_ids, new_ids[1:]):
                    pair_counts[p] += freq
                    pair_words[p].add(i)
                    changed.add(p)
                words[i] = new_ids
            for p in changed:
                if pair_counts[p] > 0:
                    heapq.heappush(heap, (-pair_counts[p], p))
                else:
                    del pair_counts[p]

            if verbose and len(merges) % 500 == 0:
                print(f"  bpe: {len(merges)}/{num_merges} merges")

        return cls(merges, special_tokens)

    # ------------------------------------------------------------ encode/decode

    def _encode_chunk(self, chunk):
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        # Apply merges in the order they were learned (lowest new id first).
        while len(ids) >= 2:
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])
        if len(self._cache) >= _CACHE_LIMIT:
            self._cache.clear()
        self._cache[chunk] = ids
        return ids

    def _encode_ordinary(self, text):
        ids = []
        for chunk in SPLIT_PATTERN.findall(text):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def encode(self, text, allow_special=False):
        """Text -> token ids.

        Special tokens such as <|end|> are only recognised when `allow_special`
        is True; otherwise they are encoded as plain text, so user input can
        never inject control tokens into a conversation.
        """
        if not allow_special or self._special_re is None:
            return self._encode_ordinary(text)
        ids = []
        for part in self._special_re.split(text):
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            elif part:
                ids.extend(self._encode_ordinary(part))
        return ids

    def decode_bytes(self, ids):
        out = []
        for i in ids:
            if i in self.vocab:
                out.append(self.vocab[i])
            elif i in self.special_names:
                out.append(self.special_names[i].encode("utf-8"))
            else:
                raise ValueError(f"unknown token id {i}")
        return b"".join(out)

    def decode(self, ids):
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    # ------------------------------------------------------------- persistence

    def save(self, path):
        merges = [list(pair) for pair, _ in sorted(self.merges.items(), key=lambda kv: kv[1])]
        data = {"type": "byte-bpe", "merges": merges, "special_tokens": list(self.special_tokens)}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        merges = {(a, b): 256 + i for i, (a, b) in enumerate(data["merges"])}
        return cls(merges, data["special_tokens"])
