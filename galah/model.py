"""Galah — a byte-level GPT, sized for browsers.

A deliberately boring decoder-only transformer: pre-LN blocks, causal SDPA
(flash on CUDA), tanh-GELU MLPs, learned positional embeddings. Vocabulary is
raw bytes (256), which buys three things at this scale:

  1. no tokenizer to train, version, or ship — the WebGPU runtime reads
     TextEncoder output directly;
  2. embedding parameters are negligible even for the 0.3M rung, so the
     6·N·D FLOP accounting used in the scaling fits stays honest;
  3. typo robustness for free, which matters for the terminal finetune.

Everything unusual for a GPT is absent on purpose: the model family is the
*subject of measurement* for the scaling study, so it stays vanilla.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GalahConfig:
    d_model: int = 128
    n_layer: int = 4
    n_head: int = 4
    seq_len: int = 1024
    vocab: int = 256
    dropout: float = 0.0

    @property
    def n_params_non_emb(self) -> int:
        """Non-embedding parameter count ≈ 12·L·d² (the N in the scaling fits)."""
        d, L = self.d_model, self.n_layer
        return L * (4 * d * d + 8 * d * d)  # attn (qkvo) + mlp (4d up, 4d down)

    @property
    def n_params_total(self) -> int:
        d = self.d_model
        return self.n_params_non_emb + self.vocab * d * 2 + self.seq_len * d

    def flops_per_token(self) -> float:
        """Fwd+bwd training FLOPs per token: 6N plus the attention quadratic."""
        return 6.0 * self.n_params_non_emb + 12.0 * self.n_layer * self.d_model * self.seq_len

    def to_dict(self) -> dict:
        return asdict(self)


class Block(nn.Module):
    def __init__(self, cfg: GalahConfig):
        super().__init__()
        d = cfg.d_model
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.up = nn.Linear(d, 4 * d)
        self.down = nn.Linear(4 * d, d)
        self.n_head = cfg.n_head
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(d, dim=2)
        q = q.view(B, T, self.n_head, -1).transpose(1, 2)
        k = k.view(B, T, self.n_head, -1).transpose(1, 2)
        v = v.view(B, T, self.n_head, -1).transpose(1, 2)
        a = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0,
        )
        a = a.transpose(1, 2).reshape(B, T, d)
        x = x + self.proj(a)
        h = self.ln2(x)
        return x + self.down(F.gelu(self.up(h), approximate="tanh"))


class Galah(nn.Module):
    def __init__(self, cfg: GalahConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        self.apply(self._init)
        # GPT-2-style scaled init on residual projections.
        for name, p in self.named_parameters():
            if name.endswith(("proj.weight", "down.weight")):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.cfg.vocab), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new: int, temperature: float = 0.8, top_k: int = 64):
        for _ in range(max_new):
            ctx = idx[:, -self.cfg.seq_len:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k:
                kth = torch.topk(logits, top_k).values[:, -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


# The size ladder for the IsoFLOP sweep. head_dim is pinned at 32 so every
# rung differs only in (depth, width) — the study varies N, nothing else.
LADDER: dict[str, GalahConfig] = {
    "galah-0.1m": GalahConfig(d_model=64, n_layer=2, n_head=2),
    "galah-0.2m": GalahConfig(d_model=96, n_layer=2, n_head=3),
    "galah-0.3m": GalahConfig(d_model=96, n_layer=3, n_head=3),
    "galah-0.8m": GalahConfig(d_model=128, n_layer=4, n_head=4),
    "galah-1.5m": GalahConfig(d_model=160, n_layer=5, n_head=5),
    "galah-2.7m": GalahConfig(d_model=192, n_layer=6, n_head=6),
    "galah-5.5m": GalahConfig(d_model=256, n_layer=7, n_head=8),
    "galah-10m": GalahConfig(d_model=320, n_layer=8, n_head=10),
    "galah-18m": GalahConfig(d_model=384, n_layer=10, n_head=12),
    "galah-38m": GalahConfig(d_model=512, n_layer=12, n_head=16),
    "galah-69m": GalahConfig(d_model=640, n_layer=14, n_head=20),
    "galah-113m": GalahConfig(d_model=768, n_layer=16, n_head=24),
}
