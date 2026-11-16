import pytest

from bpe import BPETokenizer

CORPUS = (
    "the cat sat on the mat. the dog sat on the log. "
    "machine learning models learn from data; learning is fun! 12345 67890 "
) * 20


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer(special_tokens=["<|endoftext|>"]).train(CORPUS, vocab_size=300)


@pytest.mark.parametrize("text", [
    "the cat sat on the mat.",
    "Unseen words: transformer, attention!!",
    "emoji 🤖 and accents: naïve café, 日本語",
    "",
    "   leading and trailing spaces   ",
])
def test_roundtrip(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_vocab_size(tok):
    assert tok.vocab_size == 300
    assert len(tok.merges) == 300 - 256 - 1


def test_compresses_training_text(tok):
    # Learned merges must give fewer tokens than raw bytes.
    assert tok.compression_ratio(CORPUS) > 2.0


def test_special_token_is_atomic(tok):
    ids = tok.encode("hello<|endoftext|>world")
    assert tok.special_tokens["<|endoftext|>"] in ids
    assert tok.decode(ids) == "hello<|endoftext|>world"


def test_merge_ids_are_contiguous(tok):
    assert sorted(tok.merges.values()) == list(range(256, 256 + len(tok.merges)))


def test_save_load(tok, tmp_path):
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = BPETokenizer.load(path)
    text = "the dog learns from the mat <|endoftext|>"
    assert loaded.encode(text) == tok.encode(text)
    assert loaded.decode(loaded.encode(text)) == text


def test_training_is_deterministic():
    a = BPETokenizer().train(CORPUS, 300)
    b = BPETokenizer().train(CORPUS, 300)
    assert a.merges == b.merges


def test_rejects_small_vocab():
    with pytest.raises(ValueError):
        BPETokenizer().train(CORPUS, 100)


def test_stops_when_no_pairs_left():
    # A tiny corpus runs out of pairs before reaching a huge target vocab.
    tok = BPETokenizer().train("aaaa", vocab_size=1000)
    assert tok.decode(tok.encode("aaaa")) == "aaaa"
    assert len(tok.encode("aaaa")) == 1
