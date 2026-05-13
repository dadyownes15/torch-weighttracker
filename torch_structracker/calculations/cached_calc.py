from typing import Any

import torch
from torch import nn


class CachedCalculation(nn.Module):
    def __init__(self, calculation: nn.Module) -> None:
        super().__init__()
        self.calculation = calculation

        if not hasattr(calculation, "output_anchor"):
            raise ValueError(
                f"{calculation.__class__.__name__} must define output_anchor"
            )

    @torch.no_grad()
    def refresh_cache(self, *args: Any, **kwargs: Any) -> None:
        self.calculation(*args, **kwargs)

    def forward(self) -> torch.Tensor:
        return self.calculation.output_anchor