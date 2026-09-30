"""Load a trained model and generate chat replies (used by chat.py and web.py)."""

import codecs
import os

import torch

from .data import format_chat
from .model import GPT
from .tokenizer import Tokenizer
from .training import pick_device


class Assistant:
    def __init__(self, model_dir, device="auto"):
        self.device = pick_device(device)
        self.tokenizer = Tokenizer.load(os.path.join(model_dir, "tokenizer.json"))
        self.model, _ = GPT.load(os.path.join(model_dir, "ckpt.pt"), self.device, dropout=0.0)
        self.model.eval()
        self.block_size = self.model.config.block_size

    def reply(self, messages, max_new_tokens=200, **sampling):
        """Stream the assistant's next reply to `messages`, as text pieces.

        messages: [{"role": "user" | "assistant", "content": str}, ...]
        sampling: temperature, top_k, top_p (see GPT.generate).
        """
        if not messages:
            raise ValueError("messages must not be empty")
        max_new_tokens = min(max_new_tokens, self.block_size // 2)
        budget = self.block_size - max_new_tokens
        # Forget the oldest turns until the conversation fits in the context window.
        for start in range(len(messages)):
            ids, _ = format_chat(self.tokenizer, messages[start:], add_generation_prompt=True)
            if len(ids) <= budget:
                break
        ids = ids[-budget:]
        # Any control token (<|end|>, <|user|>, ...) ends the reply.
        stop_ids = set(self.tokenizer.special_tokens.values())
        yield from self._stream(ids, max_new_tokens, stop_ids, sampling)

    def complete(self, text, max_new_tokens=200, **sampling):
        """Stream a plain continuation of `text` (useful for a model that is only pretrained)."""
        ids = self.tokenizer.encode(text)[-(self.block_size - 1):]
        stop_ids = {self.tokenizer.special_tokens["<|endoftext|>"]}
        yield from self._stream(ids, max_new_tokens, stop_ids, sampling)

    def _stream(self, ids, max_new_tokens, stop_ids, sampling):
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        # A token can end in the middle of a multi-byte character (Arabic letters
        # are 2 bytes in UTF-8), so decode incrementally.
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        for token in self.model.generate(idx, max_new_tokens, stop_ids=stop_ids, **sampling):
            piece = decoder.decode(self.tokenizer.decode_bytes([token]))
            if piece:
                yield piece
        tail = decoder.decode(b"", final=True)
        if tail:
            yield tail
