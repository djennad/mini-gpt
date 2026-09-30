"""A ChatGPT-style web interface for your model (standard library only).

    python web.py --model-dir out/chat
    # then open http://127.0.0.1:8000
"""

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from minigpt.assistant import Assistant
from minigpt.data import ROLES

MAX_MESSAGES = 50
MAX_CHARS = 4000

PAGE = """<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mini-GPT</title>
<style>
  :root { --bg: #f7f7f8; --panel: #fff; --text: #1f2328; --muted: #6b7280;
          --user: #e8f0fe; --border: #e5e7eb; --accent: #10a37f; }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #17181c; --panel: #212329; --text: #ececf1; --muted: #9ca3af;
            --user: #2b3445; --border: #33363f; --accent: #19c37d; }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text); height: 100vh;
         display: flex; flex-direction: column; font: 16px/1.7 system-ui, "Segoe UI", Tahoma, sans-serif; }
  header { display: flex; justify-content: space-between; align-items: center;
           padding: 12px 16px; border-bottom: 1px solid var(--border); background: var(--panel); }
  header h1 { font-size: 18px; margin: 0; }
  #log { flex: 1; overflow-y: auto; padding: 16px; }
  .msg { max-width: 760px; margin: 0 auto 12px; padding: 10px 14px; border-radius: 12px;
         white-space: pre-wrap; overflow-wrap: anywhere; }
  .user { background: var(--user); }
  .assistant { background: var(--panel); border: 1px solid var(--border); }
  .hint { text-align: center; color: var(--muted); margin-top: 20vh; }
  form { display: flex; gap: 8px; max-width: 792px; width: 100%; margin: 0 auto; padding: 12px 16px 20px; }
  textarea { flex: 1; resize: none; padding: 10px 12px; border-radius: 12px; font: inherit;
             border: 1px solid var(--border); background: var(--panel); color: var(--text); }
  button { border: 0; border-radius: 12px; padding: 0 18px; font: inherit; cursor: pointer;
           background: var(--accent); color: #fff; }
  button:disabled { opacity: .5; cursor: default; }
  #reset { background: transparent; color: var(--muted); border: 1px solid var(--border); padding: 4px 12px; }
</style>
</head>
<body>
<header><h1>Mini-GPT</h1><button id="reset" type="button">محادثة جديدة</button></header>
<div id="log"><p class="hint">اسألني مثلاً: ما عاصمة اليابان؟ أو: كم يساوي 7 × 8؟</p></div>
<form id="form">
  <textarea id="input" rows="1" dir="auto" placeholder="اكتب رسالتك..." autofocus></textarea>
  <button id="send">إرسال</button>
</form>
<script>
const log = document.getElementById("log"), input = document.getElementById("input"),
      form = document.getElementById("form"), send = document.getElementById("send");
let messages = [];

function add(role, text) {
  const hint = log.querySelector(".hint");
  if (hint) hint.remove();
  const div = document.createElement("div");
  div.className = "msg " + role;
  div.dir = "auto";
  div.textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
  return div;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text || send.disabled) return;
  input.value = "";
  send.disabled = true;
  messages.push({role: "user", content: text});
  add("user", text);
  const div = add("assistant", "…");
  let reply = "";
  try {
    const res = await fetch("/api/chat", {method: "POST", headers: {"Content-Type": "application/json"},
                                          body: JSON.stringify({messages})});
    if (!res.ok) throw new Error(await res.text());
    const reader = res.body.getReader(), decoder = new TextDecoder();
    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      reply += decoder.decode(value, {stream: true});
      div.textContent = reply;
      log.scrollTop = log.scrollHeight;
    }
    messages.push({role: "assistant", content: reply});
  } catch (err) {
    div.textContent = "خطأ: " + err.message;
    messages.pop();
  }
  send.disabled = false;
  input.focus();
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
});
document.getElementById("reset").addEventListener("click", () => {
  messages = [];
  log.innerHTML = "";
  input.focus();
});
</script>
</body>
</html>
"""


def parse_messages(body):
    """Validate the request body and return the conversation, or raise ValueError."""
    messages = json.loads(body).get("messages")
    if not isinstance(messages, list) or not 0 < len(messages) <= MAX_MESSAGES:
        raise ValueError(f"messages must be a list of 1 to {MAX_MESSAGES} items")
    for m in messages:
        if (not isinstance(m, dict) or m.get("role") not in ROLES
                or not isinstance(m.get("content"), str) or len(m["content"]) > MAX_CHARS):
            raise ValueError(f"each message needs a role in {ROLES} and content of at most {MAX_CHARS} characters")
    if messages[-1]["role"] != "user":
        raise ValueError("the last message must come from the user")
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def make_handler(bot, sampling):
    lock = threading.Lock()  # one generation at a time

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/":
                self.send_error(404)
                return
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/api/chat":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if length > MAX_MESSAGES * MAX_CHARS * 8:
                    raise ValueError("request too large")
                messages = parse_messages(self.rfile.read(length))
            except (ValueError, AttributeError) as e:
                self.send_error(400, explain=str(e))
                return
            # No Content-Length: the reply is streamed and the connection closes at the end.
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            with lock:
                for piece in bot.reply(messages, **sampling):
                    self.wfile.write(piece.encode("utf-8"))
                    self.wfile.flush()

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", default="out/chat")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if not os.path.exists(os.path.join(args.model_dir, "ckpt.pt")):
        parser.error(f"no trained model in {args.model_dir}, run `python train.py` and `python finetune.py` first")

    bot = Assistant(args.model_dir, args.device)
    sampling = dict(max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                    top_k=args.top_k, top_p=args.top_p)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(bot, sampling))
    print(f"Mini-GPT is running at http://{args.host}:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
