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



class CifarBasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.shortcut(x))


class CifarResNet20(nn.Module):
    """ResNet-20 for 32x32 inputs (3 stages x 3 basic blocks)."""

    def __init__(self, num_classes: int = 10, width: int = 16) -> None:
        super().__init__()
        widths = (width, 2 * width, 4 * width)
        self.stem = nn.Conv2d(3, widths[0], 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(widths[0])
        blocks: list[nn.Module] = []
        in_channels = widths[0]
        for stage, out_channels in enumerate(widths):
            for block in range(3):
                stride = 2 if stage > 0 and block == 0 else 1
                blocks.append(CifarBasicBlock(in_channels, out_channels, stride))
                in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn(self.stem(x)))
        x = self.blocks(x)
        return self.fc(x.mean(dim=(2, 3)))
