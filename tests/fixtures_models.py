from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class TinyTransformerClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(32, 16)
        self.position_embed = nn.Embedding(8, 16)
        self.attn = nn.MultiheadAttention(
            embed_dim=16,
            num_heads=4,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(16)
        self.mlp_in = nn.Linear(16, 32)
        self.activation = nn.GELU()
        self.mlp_out = nn.Linear(32, 16)
        self.norm2 = nn.LayerNorm(16)
        self.head = nn.Linear(16, 5)

    def forward(self, token_ids: Tensor) -> Tensor:
        positions = torch.arange(token_ids.size(1), device=token_ids.device).unsqueeze(0)
        x = self.token_embed(token_ids) + self.position_embed(positions)

        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)

        ff = self.mlp_out(self.activation(self.mlp_in(x)))
        x = self.norm2(x + ff)
        return self.head(x[:, 0])

