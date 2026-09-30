import json

import pytest
import torch

from minigpt.data import IGNORE_INDEX, ChatDataset, format_chat, lm_batch
from minigpt.tokenizer import Tokenizer

MESSAGES = [
    {"role": "user", "content": "ما عاصمة مصر؟"},
    {"role": "assistant", "content": "عاصمة مصر هي القاهرة."},
    {"role": "user", "content": "شكرا"},
    {"role": "assistant", "content": "عفواً!"},
]


@pytest.fixture(scope="module")
def tok():
    return Tokenizer()  # bytes only, no merges: enough for testing the layout


def test_format_chat_layout(tok):
    ids, trainable = format_chat(tok, MESSAGES)
    assert len(ids) == len(trainable)
    special = tok.special_tokens
    assert ids[0] == special["<|user|>"]
    assert ids.count(special["<|end|>"]) == 4
    # The trainable tokens are exactly the assistant replies plus their <|end|>.
    learned = [t for t, keep in zip(ids, trainable) if keep]
    expected = tok.encode("عاصمة مصر هي القاهرة.") + [special["<|end|>"]] + tok.encode("عفواً!") + [special["<|end|>"]]
    assert learned == expected


def test_generation_prompt(tok):
    ids, trainable = format_chat(tok, MESSAGES[:1], add_generation_prompt=True)
    assert ids[-1] == tok.special_tokens["<|assistant|>"]
    assert not trainable[-1]


def test_unknown_role(tok):
    with pytest.raises(ValueError):
        format_chat(tok, [{"role": "robot", "content": "hi"}])


def test_chat_dataset_labels(tok, tmp_path):
    path = tmp_path / "chat.jsonl"
    path.write_text(json.dumps({"messages": MESSAGES}, ensure_ascii=False) + "\n\n", encoding="utf-8")
    ds = ChatDataset(path, tok, block_size=512)
    assert len(ds.examples) == 1
    x, y = ds.examples[0]
    ids, trainable = format_chat(tok, MESSAGES)
    assert x == ids[:-1]
    for target, full, keep in zip(y, ids[1:], trainable[1:]):
        assert target == (full if keep else IGNORE_INDEX)

    xb, yb = ds.batch(ds.examples, batch_size=3, device="cpu")
    assert xb.shape == yb.shape == (3, len(x))


def test_chat_dataset_truncates(tok, tmp_path):
    path = tmp_path / "chat.jsonl"
    path.write_text(json.dumps({"messages": MESSAGES}, ensure_ascii=False), encoding="utf-8")
    ds = ChatDataset(path, tok, block_size=40)
    x, y = ds.examples[0]
    assert len(x) == len(y) == 40


def test_lm_batch():
    data = torch.arange(100)
    x, y = lm_batch(data, batch_size=4, block_size=8, device="cpu")
    assert x.shape == y.shape == (4, 8)
    assert torch.equal(y, x + 1)
    with pytest.raises(ValueError):
        lm_batch(torch.arange(5), batch_size=1, block_size=8, device="cpu")
