"""Stage 1 - pretraining: teach the model the language by predicting the next token of raw text.

    python train.py --data data/corpus.txt --out-dir out/pretrain
"""

import argparse
import json
import os
import sys

import torch

from minigpt import GPT, GPTConfig, Tokenizer
from minigpt.data import lm_batch
from minigpt.training import add_training_args, pick_device, seed_everything, train


def read_tokenizer_text(path):
    if not os.path.exists(path):
        sys.exit(f"{path} not found: run `python data/build_chat_data.py` first, "
                 "or pass --tokenizer-data with other files (or none)")
    with open(path, encoding="utf-8") as f:
        if not path.endswith(".jsonl"):
            return f.read()
        return "\n".join(m["content"] for line in f if line.strip() for m in json.loads(line)["messages"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", nargs="+", default=["data/corpus.txt"], help="one or more UTF-8 text files")
    parser.add_argument("--out-dir", default="out/pretrain")
    parser.add_argument("--tokenizer", help="reuse an existing tokenizer.json instead of training one")
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--tokenizer-data", nargs="*", default=["data/chat.jsonl"],
                        help="extra text (.txt) or conversation (.jsonl) files the tokenizer also learns from, "
                             "so words used in chat get their own tokens; pass no files to disable")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    g = parser.add_argument_group("model")
    g.add_argument("--block-size", type=int, default=256)
    g.add_argument("--n-layer", type=int, default=4)
    g.add_argument("--n-head", type=int, default=4)
    g.add_argument("--n-embd", type=int, default=192)
    g.add_argument("--dropout", type=float, default=0.2)
    add_training_args(parser, max_iters=2000, learning_rate=1e-3, batch_size=16, eval_interval=50)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = pick_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    # Documents are separated by <|endoftext|> so the model learns where texts end.
    texts = []
    for path in args.data:
        with open(path, encoding="utf-8") as f:
            texts.append(f.read())
    text = "<|endoftext|>".join(texts)
    print(f"corpus: {len(text):,} characters from {len(args.data)} file(s)")

    if args.tokenizer:
        tokenizer = Tokenizer.load(args.tokenizer)
    else:
        extra = [read_tokenizer_text(path) for path in args.tokenizer_data]
        print(f"training BPE tokenizer (vocab size {args.vocab_size})...")
        tokenizer = Tokenizer.train("\n".join([text, *extra]), args.vocab_size, verbose=True)
    tokenizer.save(os.path.join(args.out_dir, "tokenizer.json"))

    data = torch.tensor(tokenizer.encode(text, allow_special=True), dtype=torch.long)
    n_val = int(len(data) * args.val_fraction)
    splits = {"train": data[:-n_val] if n_val else data, "val": data[-n_val:] if n_val else data}
    print(f"tokens: {len(data):,} ({len(text) / len(data):.1f} characters per token), "
          f"vocab size {tokenizer.vocab_size}")

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(config).to(device)
    print(f"model: {model.num_params() / 1e6:.2f}M parameters, device {device}")

    def get_batch(split):
        return lm_batch(splits[split], args.batch_size, args.block_size, device)

    train(model, get_batch, args, os.path.join(args.out_dir, "ckpt.pt"))


if __name__ == "__main__":
    main()
