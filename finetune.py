"""Stage 2 - chat fine-tuning (SFT): teach a pretrained model to answer like an assistant.

    python finetune.py --init out/pretrain --data data/chat.jsonl --out-dir out/chat

The loss is computed only on the assistant's replies, so the model learns to
answer questions rather than to imitate the user.
"""

import argparse
import os

from minigpt import GPT, Tokenizer
from minigpt.data import ChatDataset
from minigpt.training import add_training_args, pick_device, seed_everything, train


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--init", default="out/pretrain", help="directory with the pretrained ckpt.pt and tokenizer.json")
    parser.add_argument("--data", default="data/chat.jsonl", help='JSONL file of {"messages": [...]} conversations')
    parser.add_argument("--out-dir", default="out/chat")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=None, help="override the pretrained model's dropout")
    add_training_args(parser, max_iters=1500, learning_rate=1e-3, batch_size=32, eval_interval=250)
    args = parser.parse_args()
    if not os.path.exists(os.path.join(args.init, "ckpt.pt")):
        parser.error(f"no pretrained model in {args.init}, run `python train.py` first")
    if not os.path.exists(args.data):
        parser.error(f"{args.data} not found, run `python data/build_chat_data.py` first")

    seed_everything(args.seed)
    device = pick_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = Tokenizer.load(os.path.join(args.init, "tokenizer.json"))
    tokenizer.save(os.path.join(args.out_dir, "tokenizer.json"))
    overrides = {} if args.dropout is None else {"dropout": args.dropout}
    model, ckpt = GPT.load(os.path.join(args.init, "ckpt.pt"), device, **overrides)
    print(f"loaded pretrained model from {args.init} (step {ckpt.get('step')}, "
          f"val loss {ckpt.get('val_loss', float('nan')):.4f})")

    dataset = ChatDataset(args.data, tokenizer, model.config.block_size)
    splits = dict(zip(("train", "val"), dataset.split(args.val_fraction, seed=args.seed)))
    print(f"conversations: {len(splits['train'])} train, {len(splits['val'])} val")

    def get_batch(split):
        return dataset.batch(splits[split], args.batch_size, device)

    train(model, get_batch, args, os.path.join(args.out_dir, "ckpt.pt"))


if __name__ == "__main__":
    main()
