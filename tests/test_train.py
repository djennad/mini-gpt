import json

import pytest

from minigpt.training import lr_at
from train import read_tokenizer_text


def test_read_tokenizer_text(tmp_path):
    chat = tmp_path / "chat.jsonl"
    messages = [{"role": "user", "content": "مرحبا"}, {"role": "assistant", "content": "أهلاً!"}]
    chat.write_text(json.dumps({"messages": messages}, ensure_ascii=False) + "\n", encoding="utf-8")
    assert read_tokenizer_text(str(chat)) == "مرحبا\nأهلاً!"

    text = tmp_path / "extra.txt"
    text.write_text("نص عادي", encoding="utf-8")
    assert read_tokenizer_text(str(text)) == "نص عادي"

    with pytest.raises(SystemExit):
        read_tokenizer_text(str(tmp_path / "missing.jsonl"))


def test_lr_schedule():
    class Args:
        learning_rate, min_lr, warmup_iters, max_iters = 1e-3, None, 10, 110

    assert lr_at(0, Args) == pytest.approx(1e-4)
    assert lr_at(9, Args) == pytest.approx(1e-3)
    assert lr_at(60, Args) == pytest.approx((1e-3 + 1e-4) / 2)
    assert lr_at(110, Args) == pytest.approx(1e-4)
