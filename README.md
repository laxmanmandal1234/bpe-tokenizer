# bpe-tokenizer

![tests](https://github.com/laxmanmandal1234/bpe-tokenizer/actions/workflows/tests.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

A **byte-level Byte-Pair Encoding (BPE) tokenizer** written from scratch in pure Python — the same family of algorithm used by GPT-style large language models.

The goal of this project is to understand, line by line, how raw text becomes the integer sequences a transformer actually sees.

## Features

- **Byte-level base vocabulary (256 symbols)** — any UTF-8 string (emoji, accents, CJK) can be encoded; there is no `<unk>` token.
- **Regex pre-tokenization** — merges never cross word/number/punctuation boundaries, which keeps tokens linguistically sensible.
- **Special tokens** (e.g. `<|endoftext|>`) that are always encoded atomically.
- **Deterministic training** with explicit tie-breaking, so the same corpus always gives the same vocabulary.
- **Save / load** to a small JSON file.
- Lossless `decode(encode(x)) == x`, verified by unit tests.

## How it works

```
text ──regex split──► pre-tokens ──utf-8──► byte ids ──apply merges──► token ids
```

**Training** (`BPETokenizer.train`)

1. Split the corpus into pre-tokens and count each distinct one (the trainer works on unique words × frequency, not on the raw corpus, so it stays fast).
2. Represent every word as a tuple of byte ids (0–255).
3. Count all adjacent pairs, weighted by word frequency.
4. Merge the most frequent pair into a new id (256, 257, …) and record it.
5. Repeat until the target vocabulary size is reached (or no pairs remain).

**Encoding** (`BPETokenizer.encode`) — for each pre-token, repeatedly apply the *earliest-learned* merge that is present until none apply. Using merge rank (not frequency) at encode time guarantees the same segmentation that training would produce.

**Decoding** — concatenate the byte strings for each id and decode UTF-8.

## Quick start

```bash
git clone https://github.com/laxmanmandal1234/bpe-tokenizer.git
cd bpe-tokenizer
python train.py --input data/sample.txt --vocab-size 512
```

```python
from bpe import BPETokenizer

tok = BPETokenizer(special_tokens=["<|endoftext|>"])
tok.train(open("data/sample.txt", encoding="utf-8").read(), vocab_size=512)

ids = tok.encode("Transformers learn representations of tokens.")
print(tok.token_strings(ids))  # ['T', 'ra', 'ns', 'form', 'ers', ' lear', 'n', ...]
assert tok.decode(ids) == "Transformers learn representations of tokens."
tok.save("tokenizer.json")
```

## Results

Compression on the included ~4 KB sample corpus (`data/sample.txt`), measured as UTF-8 bytes per token (higher = fewer tokens per text):

| vocab size | merges learned | bytes / token |
|-----------:|---------------:|--------------:|
| 300        | 43             | 1.50          |
| 400        | 143            | 2.03          |
| 512        | 255            | 2.50          |
| 768        | 511            | 3.37          |

Compression grows with vocabulary size, illustrating the core trade-off in tokenizer design: a larger vocabulary means shorter sequences (cheaper attention) but a larger embedding matrix and rarer tokens. *(Numbers are on a tiny corpus and are for illustration; they will differ on real-world data.)*

## Tests

```bash
pip install pytest
pytest -q
```

Tests cover round-trip correctness on Unicode, special-token handling, determinism, save/load, and edge cases.

## Project structure

```
bpe/tokenizer.py   # BPETokenizer: train / encode / decode / save / load
train.py           # CLI: train on a file and print statistics
tests/             # pytest unit tests
data/sample.txt    # small sample corpus
```

## Possible extensions

- Faster training with incremental pair-count updates (avoid recounting all pairs each step).
- Compare vocabularies learned on English vs. code vs. multilingual text.
- Measure how tokenization affects downstream language-model perplexity.

## References

- Sennrich, Haddow & Birch (2016). *Neural Machine Translation of Rare Words with Subword Units.*
- Radford et al. (2019). *Language Models are Unsupervised Multitask Learners* (byte-level BPE).

## License

MIT © Laxman Mandal
