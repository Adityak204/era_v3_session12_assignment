import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MulitHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_head = config.num_head
        self.n_embd = config.emb_dim

        self.c_attn = nn.Linear(
            config.emb_dim, 3 * config.emb_dim
        )  # linear layer for query, key, value [emb_dim, 3 * emb_dim]
        self.c_proj = nn.Linear(
            config.emb_dim, config.emb_dim
        )  # linear layer for output

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.c_attn(x)
        q, k, v = qkv.split(
            self.n_embd, dim=2
        )  # torch.split(tensor, split_size_or_sections, dim=0) [B, T, emb_dim]
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # [B, nh, T, hs]
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # [B, nh, T, hs]
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # [B, nh, T, hs]

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.c_fc = nn.Linear(
            config.emb_dim, 4 * config.emb_dim
        )  # [emb_dim, 4 * emb_dim]
        self.c_proj = nn.Linear(
            4 * config.emb_dim, config.emb_dim
        )  # [4 * emb_dim, emb_dim]
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = x + self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(config.emb_dim)
        self.ln_2 = nn.LayerNorm(config.emb_dim)
        self.attn = MulitHeadAttention(config)
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
            dict(
                wte=nn.Embedding(config.vocab_size, config.emb_dim),
                wpe=nn.Embedding(config.block_size, config.emb_dim),
                h=nn.ModuleList(
                    [TransformerBlock(config) for _ in range(config.num_layer)]
                ),
                ln_f=nn.LayerNorm(config.emb_dim),
            )
        )
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def apply(self, fn):
        for module in self.modules():
            fn(module)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        # position indices
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(
            pos
        )  # position embeddings of shape (1, t, n_embd)
        x = tok_emb + pos_emb

        # transformer
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        return logits  # note: using list [-1] to preserve the time dim
