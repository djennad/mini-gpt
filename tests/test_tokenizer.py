import pytest

from minigpt.tokenizer import Tokenizer

TEXT = (
    "الذكاء الاصطناعي مجال في علوم الحاسوب. مَرْحَبًا بالعالم، كم يساوي 12 + 30؟\n"
    "Hello world! It's a small GPT model 🤖 with byte-level BPE.\n"
) * 20


@pytest.fixture(scope="module")
def tok():
    return Tokenizer.train(TEXT, vocab_size=400)


@pytest.mark.parametrize("text", [
    TEXT,
    "نص جديد لم يره المرمّز من قبل: ﷺ ١٢٣",
    "",
    "   spaces\t\tand\nnewlines  ",
    "emoji 🎉🚀 and 中文",
])
def test_roundtrip(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_training_compresses(tok):
    assert tok.vocab_size <= 400
    assert len(tok.encode(TEXT)) < len(TEXT.encode("utf-8")) / 2


def test_special_tokens_only_when_allowed(tok):
    end = tok.special_tokens["<|end|>"]
    assert tok.encode("hi<|end|>", allow_special=True)[-1] == end
    # User text must never be able to produce a control token.
    assert end not in tok.encode("hi<|end|>")
    assert tok.decode(tok.encode("hi<|end|>")) == "hi<|end|>"


def test_special_ids_follow_merges(tok):
    ids = sorted(tok.special_tokens.values())
    assert ids == list(range(256 + len(tok.merges), tok.vocab_size))


def test_save_load(tok, tmp_path):
    path = tmp_path / "tokenizer.json"
    tok.save(path)
    loaded = Tokenizer.load(path)
    assert loaded.merges == tok.merges
    assert loaded.special_tokens == tok.special_tokens
    assert loaded.encode(TEXT) == tok.encode(TEXT)


def test_vocab_size_too_small():
    with pytest.raises(ValueError):
        Tokenizer.train("abc", vocab_size=100)
