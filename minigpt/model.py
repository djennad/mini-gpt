"""A GPT-style decoder-only Transformer (the architecture behind ChatGPT), at small scale."""

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 2048
    block_size: int = 256  # maximum context length, in tokens
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 192
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where each token can only look at earlier tokens."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim): every head attends independently
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        # softmax(Q K^T / sqrt(head_dim) + causal_mask) V
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """One Transformer layer: attention then MLP, each with a residual connection."""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying: the same matrix embeds tokens and scores the next token.
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # GPT-2 trick: scale down residual projections so deep stacks start stable.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        # Position embeddings are excluded, as is conventional for GPT models.
        return sum(p.numel() for p in self.parameters()) - self.pos_emb.weight.numel()

    def forward(self, idx, targets=None):
        """idx: (B, T) token ids. targets: (B, T) next-token ids, -100 = ignore."""
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.config.block_size}")
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, stop_ids=()):
        """Yield new token ids one at a time, so callers can stream the output.

        idx is a (1, T) tensor holding the prompt. Generation ends after
        `max_new_tokens` or when a token in `stop_ids` is sampled (the stop
        token itself is not yielded).
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            next_id = sample_next(logits[:, -1, :], temperature, top_k, top_p)
            token = next_id.item()
            if token in stop_ids:
                return
            idx = torch.cat([idx, next_id], dim=1)
            yield token

    def configure_optimizer(self, learning_rate, weight_decay, betas=(0.9, 0.95)):
        # Decay matrices only; biases and LayerNorm weights are left alone.
        params = [p for p in self.parameters() if p.requires_grad]
        groups = [
            {"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay},
            {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas)

    # ------------------------------------------------------------- checkpoints

    def save(self, path, **extra):
        torch.save({"model": self.state_dict(), "config": asdict(self.config), **extra}, path)

    @classmethod
    def load(cls, path, device="cpu", **overrides):
        """Load a checkpoint; `overrides` replace config fields (e.g. dropout=0.0)."""
        ckpt = torch.load(path, map_location=device, weights_only=True)
        model = cls(GPTConfig(**{**ckpt["config"], **overrides}))
        model.load_state_dict(ckpt["model"])
        return model.to(device), ckpt


def sample_next(logits, temperature=1.0, top_k=None, top_p=None):
    """Pick the next token from (B, vocab) logits. temperature=0 means greedy."""
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and top_p < 1.0:
        # Nucleus sampling: keep the smallest set of tokens whose probability sums to top_p.
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        remove = probs.cumsum(dim=-1) - probs > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
