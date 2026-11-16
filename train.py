"""Train a BPE tokenizer on a text file and report compression statistics.

Example:
    python train.py --input data/sample.txt --vocab-size 512 --out outputs/tokenizer.json
"""

import argparse
import time
from pathlib import Path

from bpe import BPETokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/sample.txt")
    ap.add_argument("--vocab-size", type=int, default=512)
    ap.add_argument("--out", default="outputs/tokenizer.json")
    ap.add_argument("--special", nargs="*", default=["<|endoftext|>"])
    args = ap.parse_args()

    text = Path(args.input).read_text(encoding="utf-8")
    tok = BPETokenizer(special_tokens=args.special)

    t0 = time.perf_counter()
    tok.train(text, args.vocab_size, verbose=True)
    dt = time.perf_counter() - t0

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tok.save(args.out)

    sample = "Transformers learn representations of tokens. <|endoftext|>"
    ids = tok.encode(sample)
    print(f"\ntrained {len(tok.merges)} merges in {dt:.2f}s -> {args.out}")
    print(f"vocab size           : {tok.vocab_size}")
    print(f"bytes/token (corpus) : {tok.compression_ratio(text):.3f}")
    print(f"example ids          : {ids}")
    print(f"example tokens       : {tok.token_strings(ids)}")
    assert tok.decode(ids) == sample


if __name__ == "__main__":
    main()
