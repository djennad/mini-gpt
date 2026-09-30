"""Talk to your model in the terminal.

    python chat.py                                   # interactive chat with out/chat
    python chat.py --prompt "ما عاصمة مصر؟"            # ask a single question
    python chat.py --model-dir out/pretrain --complete "الذكاء الاصطناعي"   # continue raw text

In the interactive chat, /reset starts a new conversation and /exit quits.
"""

import argparse
import os
import sys

from minigpt.assistant import Assistant


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", default="out/chat")
    parser.add_argument("--prompt", help="ask one question and exit")
    parser.add_argument("--complete", help="continue this text instead of chatting (for pretrained-only models)")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7, help="0 = always pick the most likely token")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if not os.path.exists(os.path.join(args.model_dir, "ckpt.pt")):
        parser.error(f"no trained model in {args.model_dir}, run `python train.py` and `python finetune.py` first")

    bot = Assistant(args.model_dir, args.device)
    sampling = dict(max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                    top_k=args.top_k, top_p=args.top_p)

    def show(pieces):
        text = ""
        for piece in pieces:
            print(piece, end="", flush=True)
            text += piece
        print()
        return text

    if args.complete is not None:
        print(args.complete, end="")
        show(bot.complete(args.complete, **sampling))
        return
    if args.prompt is not None:
        show(bot.reply([{"role": "user", "content": args.prompt}], **sampling))
        return

    print("Mini-GPT - /reset: new conversation, /exit: quit")
    history = []
    while True:
        try:
            user = input("\nأنت: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/exit", "/quit"):
            break
        if user == "/reset":
            history = []
            print("(new conversation)")
            continue
        history.append({"role": "user", "content": user})
        sys.stdout.write("Mini-GPT: ")
        history.append({"role": "assistant", "content": show(bot.reply(history, **sampling))})


if __name__ == "__main__":
    main()
