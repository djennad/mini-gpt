import torch

from minigpt.assistant import Assistant
from minigpt.model import GPT, GPTConfig
from minigpt.tokenizer import Tokenizer


def make_assistant(tmp_path, block_size=64):
    tok = Tokenizer.train("مرحبا بك في هذا الاختبار. hello there! " * 20, vocab_size=300)
    tok.save(tmp_path / "tokenizer.json")
    torch.manual_seed(0)
    config = GPTConfig(vocab_size=tok.vocab_size, block_size=block_size, n_layer=1, n_head=2, n_embd=16)
    GPT(config).save(tmp_path / "ckpt.pt")
    return Assistant(str(tmp_path), device="cpu")


def test_reply_streams_text(tmp_path):
    bot = make_assistant(tmp_path)
    pieces = list(bot.reply([{"role": "user", "content": "مرحبا"}], max_new_tokens=10, temperature=1.0))
    assert all(isinstance(p, str) for p in pieces)


def test_long_history_is_truncated(tmp_path):
    bot = make_assistant(tmp_path, block_size=64)
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": "مرحبا بك " * 10}
               for i in range(9)]
    # Must not fail even though the history is far longer than the context window.
    text = "".join(bot.reply(history, max_new_tokens=20, temperature=0))
    assert isinstance(text, str)


def test_complete(tmp_path):
    bot = make_assistant(tmp_path)
    text = "".join(bot.complete("hello", max_new_tokens=8, temperature=0))
    assert isinstance(text, str)
