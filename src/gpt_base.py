import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Ensure embedding dimension is divisible by number of heads
        assert config.emb_dim % config.num_head == 0

        self.n_head = config.num_head
        self.n_embd = config.emb_dim
        self.head_size = config.emb_dim // config.num_head

        # Separate projections for Q, K, V instead of a single projection
        self.q_proj = nn.Linear(config.emb_dim, config.emb_dim)
        self.k_proj = nn.Linear(config.emb_dim, config.emb_dim)
        self.v_proj = nn.Linear(config.emb_dim, config.emb_dim)
        self.out_proj = nn.Linear(config.emb_dim, config.emb_dim)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size()  # batch, sequence length, embedding dim

        # Separate projections for Q, K, V
        q = self.q_proj(x)  # (B, T, C)
        k = self.k_proj(x)  # (B, T, C)
        v = self.v_proj(x)  # (B, T, C)

        # Reshape heads
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)

        # Compute attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Apply attention to values
        y = att @ v  # (B, nh, T, hs)

        # Reshape and project output
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        y = self.out_proj(y)
        y = self.resid_dropout(y)

        return y


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.emb_dim, 4 * config.emb_dim)
        self.c_proj = nn.Linear(4 * config.emb_dim, config.emb_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.gelu(self.c_fc(x))
        x = self.dropout(self.c_proj(x))
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.emb_dim)
        self.ln_2 = nn.LayerNorm(config.emb_dim)
        self.attn = MultiHeadAttention(config)
        self.mlp = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.emb_dim),
                "wpe": nn.Embedding(config.block_size, config.emb_dim),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList(
                    [TransformerBlock(config) for _ in range(config.num_layer)]
                ),
                "ln_f": nn.LayerNorm(config.emb_dim),
            }
        )

        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

        # Tie weights between embedding and final linear layer
        self.transformer.wte.weight = self.lm_head.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # Get positions
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # (1, t)

        # Get embeddings
        tok_emb = self.transformer.wte(idx)  # (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Apply transformer blocks
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        return logits
