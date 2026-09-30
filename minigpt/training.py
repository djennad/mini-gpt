"""The training loop shared by pretraining (train.py) and chat fine-tuning (finetune.py)."""

import math
import os
import random
import time

import torch


def pick_device(requested="auto"):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def add_training_args(parser, max_iters, learning_rate, batch_size, eval_interval):
    g = parser.add_argument_group("training")
    g.add_argument("--max-iters", type=int, default=max_iters)
    g.add_argument("--batch-size", type=int, default=batch_size)
    g.add_argument("--learning-rate", type=float, default=learning_rate)
    g.add_argument("--min-lr", type=float, default=None, help="default: learning_rate / 10")
    g.add_argument("--warmup-iters", type=int, default=100)
    g.add_argument("--weight-decay", type=float, default=0.1)
    g.add_argument("--grad-clip", type=float, default=1.0)
    g.add_argument("--eval-interval", type=int, default=eval_interval)
    g.add_argument("--eval-iters", type=int, default=10)
    g.add_argument("--patience", type=int, default=5,
                   help="stop early after this many evaluations without a better val loss (0 = never)")
    g.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    g.add_argument("--seed", type=int, default=1337)


def lr_at(it, args):
    """Linear warmup, then cosine decay down to min_lr."""
    min_lr = args.min_lr if args.min_lr is not None else args.learning_rate / 10
    if it < args.warmup_iters:
        return args.learning_rate * (it + 1) / args.warmup_iters
    progress = (it - args.warmup_iters) / max(1, args.max_iters - args.warmup_iters)
    return min_lr + 0.5 * (args.learning_rate - min_lr) * (1 + math.cos(math.pi * min(progress, 1.0)))


@torch.no_grad()
def estimate_loss(model, get_batch, eval_iters):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = [model(*get_batch(split))[1].item() for _ in range(eval_iters)]
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def train(model, get_batch, args, ckpt_path):
    """Train `model` and save the checkpoint with the best validation loss to `ckpt_path`.

    get_batch(split) must return (x, y) tensors for split "train" or "val".
    """
    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
    optimizer = model.configure_optimizer(args.learning_rate, args.weight_decay)
    best_val = float("inf")
    evals_without_improvement = 0
    model.train()
    t0 = time.time()

    for it in range(args.max_iters + 1):
        lr = lr_at(it, args)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if it % args.eval_interval == 0 or it == args.max_iters:
            losses = estimate_loss(model, get_batch, args.eval_iters)
            saved = ""
            if losses["val"] < best_val:
                best_val = losses["val"]
                evals_without_improvement = 0
                model.save(ckpt_path, step=it, val_loss=best_val)
                saved = "  -> saved"
            else:
                evals_without_improvement += 1
            print(f"step {it:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}{saved}")
            if args.patience and evals_without_improvement >= args.patience:
                print(f"val loss has not improved for {args.patience} evaluations "
                      "(the model is memorising the training data), stopping early")
                break
        if it == args.max_iters:
            break

        x, y = get_batch("train")
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if it % 50 == 0:
            elapsed = time.time() - t0
            print(f"  iter {it:5d} | loss {loss.item():.4f} | lr {lr:.2e} | {elapsed:.0f}s")

    print(f"done. best val loss {best_val:.4f}, checkpoint: {ckpt_path}")
    return best_val
