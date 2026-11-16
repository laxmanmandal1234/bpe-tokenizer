"""Byte-level BPE tokenizer.

The design follows the scheme used by modern LLM tokenizers:

1. Text is split into "pre-tokens" with a regex so merges never cross
   word / number / punctuation boundaries.
2. Each pre-token is encoded as UTF-8 bytes, giving a base vocabulary of 256
   symbols, so *any* string can be encoded (no unknown tokens).
3. Training repeatedly merges the most frequent adjacent pair of symbols into a
   new symbol until the target vocabulary size is reached.
4. Encoding applies the learned merges greedily in the order they were learned
   (lowest merge rank first).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

Pair = Tuple[int, int]

# Contractions, letter runs, digit runs (max 3), punctuation runs, whitespace.
DEFAULT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d{1,3}| ?[^\s\w]+|\s+(?!\S)|\s+"""


def _pair_counts(words: Dict[Tuple[int, ...], int]) -> Counter:
    counts: Counter = Counter()
    for symbols, freq in words.items():
        for pair in zip(symbols, symbols[1:]):
            counts[pair] += freq
    return counts


def _merge(symbols: Tuple[int, ...], pair: Pair, new_id: int) -> Tuple[int, ...]:
    out: List[int] = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


class BPETokenizer:
    def __init__(self, pattern: str = DEFAULT_PATTERN, special_tokens: Iterable[str] = ()):
        self.pattern = pattern
        self._regex = re.compile(pattern)
        self.merges: Dict[Pair, int] = {}
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special_tokens: Dict[str, int] = {}
        for tok in special_tokens:
            self.add_special_token(tok)

    # ------------------------------------------------------------------ train
    def train(self, text: str, vocab_size: int, verbose: bool = False) -> "BPETokenizer":
        if vocab_size < 256 + len(self.special_tokens):
            raise ValueError("vocab_size must cover the 256-byte alphabet plus special tokens")
        if self.merges:
            raise RuntimeError("tokenizer is already trained")

        # Unique pre-tokens with counts: training cost scales with the number
        # of distinct words rather than corpus length.
        word_freq = Counter(self._regex.findall(text))
        words = {tuple(w.encode("utf-8")): f for w, f in word_freq.items()}

        n_merges = vocab_size - 256 - len(self.special_tokens)
        for step in range(n_merges):
            counts = _pair_counts(words)
            if not counts:
                break
            # Most frequent pair; ties broken deterministically on the ids.
            pair = max(counts, key=lambda p: (counts[p], -p[0], -p[1]))
            new_id = 256 + len(self.merges)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            words = {_merge(s, pair, new_id): f for s, f in words.items()}
            if verbose and (step % 50 == 0 or step == n_merges - 1):
                print(f"merge {step + 1}/{n_merges}: {pair} -> {new_id} "
                      f"{self.vocab[new_id]!r} (count={counts[pair]})")

        # Re-number special tokens so they sit after the learned merges.
        specials = list(self.special_tokens)
        self.special_tokens = {}
        for tok in specials:
            self.add_special_token(tok)
        return self

    def add_special_token(self, token: str) -> int:
        if token in self.special_tokens:
            return self.special_tokens[token]
        new_id = 256 + len(self.merges) + len(self.special_tokens)
        self.special_tokens[token] = new_id
        return new_id

    # ------------------------------------------------------------ encode/decode
    def _encode_chunk(self, chunk: bytes) -> List[int]:
        ids = list(chunk)
        while len(ids) >= 2:
            # Apply the earliest-learned merge present in the sequence.
            candidates = {p for p in zip(ids, ids[1:]) if p in self.merges}
            if not candidates:
                break
            pair = min(candidates, key=self.merges.__getitem__)
            ids = list(_merge(tuple(ids), pair, self.merges[pair]))
        return ids

    def _encode_ordinary(self, text: str) -> List[int]:
        ids: List[int] = []
        for piece in self._regex.findall(text):
            ids.extend(self._encode_chunk(piece.encode("utf-8")))
        return ids

    def encode(self, text: str) -> List[int]:
        if not self.special_tokens:
            return self._encode_ordinary(text)
        split = "(" + "|".join(re.escape(t) for t in
                               sorted(self.special_tokens, key=len, reverse=True)) + ")"
        ids: List[int] = []
        for part in re.split(split, text):
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            elif part:
                ids.extend(self._encode_ordinary(part))
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        inverse_special = {v: k.encode("utf-8") for k, v in self.special_tokens.items()}
        parts = []
        for i in ids:
            if i in self.vocab:
                parts.append(self.vocab[i])
            elif i in inverse_special:
                parts.append(inverse_special[i])
            else:
                raise KeyError(f"unknown token id {i}")
        return b"".join(parts).decode("utf-8", errors="replace")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special_tokens)

    def compression_ratio(self, text: str) -> float:
        """UTF-8 bytes per token: higher means better compression."""
        n_tokens = len(self.encode(text))
        return len(text.encode("utf-8")) / max(n_tokens, 1)

    # ---------------------------------------------------------------- persist
    def save(self, path) -> None:
        data = {
            "version": 1,
            "pattern": self.pattern,
            "merges": [[a, b, idx] for (a, b), idx in self.merges.items()],
            "special_tokens": self.special_tokens,
        }
        Path(path).write_text(json.dumps(data, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls(pattern=data["pattern"])
        for a, b, idx in sorted(data["merges"], key=lambda m: m[2]):
            tok.merges[(a, b)] = idx
            tok.vocab[idx] = tok.vocab[a] + tok.vocab[b]
        tok.special_tokens = {k: int(v) for k, v in data["special_tokens"].items()}
        return tok

    def token_strings(self, ids: Iterable[int]) -> List[str]:
        """Human-readable view of each token (useful for debugging)."""
        return [self.decode([i]) for i in ids]
