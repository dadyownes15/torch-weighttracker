from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..analysis import StructureAnalyzer
from ..analysis.types import StructureAxis


class CoupledGroupLasso(nn.Module):
    def __init__(
        self,
        analyzer: StructureAnalyzer,
        eps: float = 1e-8,
        include_bias: bool = False,
        weighting: str = "sqrt_size",
        reduction: str = "sum",
        axes: tuple[StructureAxis, ...] | None = None,
    ):
        super().__init__()
        self.analyzer = analyzer
        self.eps = eps
        self.include_bias = include_bias
        self.weighting = weighting
        self.reduction = reduction
        self.axes = axes

    def forward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        terms: list[torch.Tensor] = []
        named_terms: dict[str, torch.Tensor] = {}
        for group_view in self.analyzer.iter_groups():
            if self.axes is not None and group_view.axis not in self.axes:
                continue
            for unit in group_view.prune_units:
                tensors = self.analyzer.unit_tensors(
                    group_view,
                    unit,
                    include_bias=self.include_bias,
                )
                if len(tensors) == 0:
                    continue
                unit_norm_sq = None
                total_size = 0
                for tensor in tensors:
                    total_size += tensor.numel()
                    term = tensor.pow(2).sum()
                    unit_norm_sq = term if unit_norm_sq is None else unit_norm_sq + term
                unit_norm = torch.sqrt(unit_norm_sq + self.eps)
                unit_norm = self._apply_weight(unit_norm, total_size)
                terms.append(unit_norm)
                term_name = f"{group_view.group_id}[{unit.unit_index}]"
                named_terms[term_name] = unit_norm

        if len(terms) == 0:
            try:
                device = next(self.analyzer.model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
            return torch.zeros((), device=device), {}

        total = torch.stack(terms).sum()
        if self.reduction == "mean":
            total = total / len(terms)
            named_terms = {name: value / len(terms) for name, value in named_terms.items()}
        elif self.reduction != "sum":
            raise ValueError(f"Unsupported reduction: {self.reduction}")
        return total, named_terms

    def _apply_weight(self, value: torch.Tensor, total_size: int) -> torch.Tensor:
        if self.weighting == "none":
            return value
        if self.weighting == "sqrt_size":
            return value * math.sqrt(total_size)
        if self.weighting == "size":
            return value * total_size
        raise ValueError(f"Unsupported weighting mode: {self.weighting}")
