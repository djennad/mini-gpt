"""Batching for the two training stages: pretraining on raw text, and chat fine-tuning."""

import json
import random

import torch

IGNORE_INDEX = -100
ROLES = ("user", "assistant")


def lm_batch(data, batch_size, block_size, device):
    """Random windows from a 1-D tensor of token ids; y is x shifted by one token."""
    if len(data) <= block_size + 1:
        raise ValueError(
            f"only {len(data)} tokens of data, need more than block_size + 1 = {block_size + 1}"
        )
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


def format_chat(tokenizer, messages, add_generation_prompt=False):
    """Encode a conversation as  <|user|>...<|end|><|assistant|>...<|end|>...

    Returns (ids, trainable) where trainable[i] is True for tokens the model
    should learn to produce: the assistant's replies and their <|end|>. With
    `add_generation_prompt`, a trailing <|assistant|> asks the model to reply.
    """
    end = tokenizer.special_tokens["<|end|>"]
    ids, trainable = [], []
    for message in messages:
        role = message["role"]
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}, expected one of {ROLES}")
        content = tokenizer.encode(message["content"])
        ids += [tokenizer.special_tokens[f"<|{role}|>"], *content, end]
        trainable += [False] + [role == "assistant"] * (len(content) + 1)
    if add_generation_prompt:
        ids.append(tokenizer.special_tokens["<|assistant|>"])
        trainable.append(False)
    return ids, trainable


class ChatDataset:
    """Conversations from a JSONL file, one {"messages": [...]} object per line."""

    def __init__(self, path, tokenizer, block_size):
        self.pad_id = tokenizer.special_tokens["<|endoftext|>"]
        self.examples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                ids, trainable = format_chat(tokenizer, json.loads(line)["messages"])
                ids, trainable = ids[:block_size + 1], trainable[:block_size + 1]
                x = ids[:-1]
                # Only the assistant's tokens contribute to the loss.
                y = [t if keep else IGNORE_INDEX for t, keep in zip(ids[1:], trainable[1:])]
                if any(t != IGNORE_INDEX for t in y):
                    self.examples.append((x, y))

    def split(self, val_fraction, seed=0):
        """Shuffle and split into (train, val) lists of examples."""
        examples = self.examples[:]
        random.Random(seed).shuffle(examples)
        n_val = max(1, int(len(examples) * val_fraction))
        return examples[n_val:], examples[:n_val]

    def batch(self, examples, batch_size, device):
        picks = random.choices(examples, k=batch_size)
        T = max(len(x) for x, _ in picks)
        # Right padding is safe: causal attention never looks at later positions.
        x = torch.full((batch_size, T), self.pad_id, dtype=torch.long)
        y = torch.full((batch_size, T), IGNORE_INDEX, dtype=torch.long)
        for i, (xi, yi) in enumerate(picks):
            x[i, :len(xi)] = torch.tensor(xi)
            y[i, :len(yi)] = torch.tensor(yi)
        return x.to(device), y.to(device)
