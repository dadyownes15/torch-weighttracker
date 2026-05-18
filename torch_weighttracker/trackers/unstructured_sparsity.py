from __future__ import annotations

from collections.abc import Iterable

import torch

from torch_weighttracker.calculations import CalcType, CalculationContext
from torch_weighttracker.consumer_ignore import (
    ConsumerFilter,
    FilterItem,
    filter_modules,
)
from torch_weighttracker.trackers.base import BaseTracker


class UnstructuredSparsity(BaseTracker):
    required_calculations = (CalcType.UNSTRUCTURED_SPARSITY_PR_MODULE,)

    def __init__(
        self,
        calculations=None,
        *,
        _module_names: Iterable[str] = (),
    ) -> None:
        super().__init__(calculations=calculations)
        self.module_names = tuple(_module_names)

    @classmethod
    def calculation_context(
        cls,
        owner,
        *,
        include: Iterable[FilterItem] = (),
        ignore: Iterable[FilterItem] = (),
        **kwargs,
    ) -> CalculationContext | None:
        filters = ConsumerFilter(include=include, ignore=ignore)
        if not filters:
            return None

        return owner._calculation_context(
            weighted_modules=filter_modules(owner._get_weighted_modules(), filters),
        )

    @classmethod
    def constructor_kwargs(
        cls,
        owner,
        *,
        context: CalculationContext | None = None,
        **kwargs,
    ) -> dict:
        metric_context = (
            context if context is not None else owner._calculation_context()
        )
        return {
            **kwargs,
            "_module_names": metric_context.weighted_module_names,
        }

    def compute(self) -> torch.Tensor:
        return self.calc(CalcType.UNSTRUCTURED_SPARSITY_PR_MODULE)()

    def toMetric(self, result: torch.Tensor):
        counts = result.view(-1, 2)
        if counts.numel() == 0:
            return {
                "unstructured_sparsity": result.new_zeros(()),
                "layers": {},
            }

        zero_counts = counts[:, 0]
        total_counts = counts[:, 1]
        layer_sparsity = _safe_fraction(zero_counts, total_counts)
        total_sparsity = _safe_fraction(zero_counts.sum(), total_counts.sum())

        return {
            "unstructured_sparsity": total_sparsity,
            "layers": {
                name: sparsity
                for name, sparsity in zip(
                    self.module_names,
                    layer_sparsity,
                    strict=True,
                )
            },
        }


def _safe_fraction(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    safe_denominator = torch.where(
        denominator.ne(0),
        denominator,
        torch.ones_like(denominator),
    )
    fraction = numerator / safe_denominator
    return torch.where(
        denominator.ne(0),
        fraction,
        torch.zeros_like(fraction),
    )
