from __future__ import annotations

from collections.abc import Iterable

import torch

from torch_weighttracker.calculations import CalcType, CalculationContext
from torch_weighttracker.calculations.calcs.block_2_4_sparsity import (
    supports_2_4_block_sparsity,
)
from torch_weighttracker.consumer_ignore import (
    ConsumerFilter,
    FilterItem,
    filter_modules,
)
from torch_weighttracker.trackers.base import BaseTracker


class Nvidia24Sparsity(BaseTracker):
    required_calculations = (CalcType.BLOCK_2_4_SPARSITY,)

    def __init__(
        self,
        calculations=None,
        *,
        log_layerwise_stats: bool = False,
        _module_names: Iterable[str] = (),
    ) -> None:
        super().__init__(calculations=calculations)
        self.log_layerwise_stats = log_layerwise_stats
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
        weighted_modules = owner._get_weighted_modules()
        filtered_modules = (
            filter_modules(weighted_modules, filters) if filters else weighted_modules
        )
        supported_modules = tuple(
            module for module in filtered_modules if supports_2_4_block_sparsity(module)
        )

        if supported_modules == weighted_modules:
            return None

        return owner._calculation_context(weighted_modules=supported_modules)

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
        return self.calc(CalcType.BLOCK_2_4_SPARSITY)()

    def toMetric(self, result: torch.Tensor):
        counts = result.view(-1, 4)
        strict_blocks = counts[:, 0]
        eligible_blocks = counts[:, 1]
        total_blocks = counts[:, 2]
        tail_elements = counts[:, 3]

        strict_layer_mask = (
            total_blocks.gt(0) & tail_elements.eq(0) & strict_blocks.eq(total_blocks)
        )
        eligible_layer_mask = (
            total_blocks.gt(0) & tail_elements.eq(0) & eligible_blocks.eq(total_blocks)
        )

        metrics = {
            "nvidia_2_4_sparsity/strict_block_fraction": _safe_fraction(
                strict_blocks.sum(),
                total_blocks.sum(),
            ),
            "nvidia_2_4_sparsity/nvidia_eligible_block_fraction": _safe_fraction(
                eligible_blocks.sum(),
                total_blocks.sum(),
            ),
            "nvidia_2_4_sparsity/strict_layers": strict_layer_mask.sum().to(
                dtype=result.dtype,
            ),
            "nvidia_2_4_sparsity/nvidia_eligible_layers": (
                eligible_layer_mask.sum().to(dtype=result.dtype)
            ),
            "nvidia_2_4_sparsity/total_layers": result.new_tensor(
                float(counts.shape[0]),
            ),
            "nvidia_2_4_sparsity/tail_elements": tail_elements.sum(),
        }

        if self.log_layerwise_stats:
            metrics.update(
                _layerwise_metrics(
                    self.module_names,
                    strict_blocks,
                    eligible_blocks,
                    total_blocks,
                    tail_elements,
                    strict_layer_mask,
                    eligible_layer_mask,
                )
            )

        return metrics


def _layerwise_metrics(
    module_names: Iterable[str],
    strict_blocks: torch.Tensor,
    eligible_blocks: torch.Tensor,
    total_blocks: torch.Tensor,
    tail_elements: torch.Tensor,
    strict_layer_mask: torch.Tensor,
    eligible_layer_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    metrics = {}
    for index, name in enumerate(module_names):
        prefix = f"nvidia_2_4_sparsity/layers/{name}"
        metrics[f"{prefix}/strict_block_fraction"] = _safe_fraction(
            strict_blocks[index],
            total_blocks[index],
        )
        metrics[f"{prefix}/nvidia_eligible_block_fraction"] = _safe_fraction(
            eligible_blocks[index],
            total_blocks[index],
        )
        metrics[f"{prefix}/strict_blocks"] = strict_blocks[index]
        metrics[f"{prefix}/nvidia_eligible_blocks"] = eligible_blocks[index]
        metrics[f"{prefix}/total_blocks"] = total_blocks[index]
        metrics[f"{prefix}/tail_elements"] = tail_elements[index]
        metrics[f"{prefix}/is_strict_layer"] = strict_layer_mask[index].to(
            dtype=strict_blocks.dtype,
        )
        metrics[f"{prefix}/is_nvidia_eligible_layer"] = eligible_layer_mask[index].to(
            dtype=strict_blocks.dtype,
        )

    return metrics


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
