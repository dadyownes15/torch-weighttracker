from collections.abc import Iterable

import torch

from torch_weighttracker.calculations import CalcType, CalculationContext
from torch_weighttracker.consumer_ignore import (
    ConsumerFilter,
    FilterItem,
    filter_canonical_members,
    filter_modules,
)
from torch_weighttracker.trackers.base import BaseTracker


class StructuredBOPs(BaseTracker):
    required_calculations = (
        CalcType.ACTIVE_MACS_PR_MODULE,
        CalcType.BITRATE_PR_MODULE,
        CalcType.BASELINE_MACS_PR_MODULE,
    )

    def __init__(
        self,
        calculations=None,
        *,
        log_module_names: bool = False,
        log_compression_rate: bool = False,
        log_total_bops: bool = False,
        _module_names: Iterable[str] = (),
    ) -> None:
        super().__init__(calculations=calculations)
        self.log_module_names = log_module_names
        self.log_compression_rate = log_compression_rate
        self.log_total_bops = log_total_bops
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
            canonical_groups=filter_canonical_members(
                owner.canonical_groups,
                filters,
            ),
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

    def compute(self):
        active_macs = self.calc(CalcType.ACTIVE_MACS_PR_MODULE)()
        bitrates = self.calc(CalcType.BITRATE_PR_MODULE)()
        bitrate_product = bitrates.view(-1, 2).prod(dim=1)
        return active_macs * bitrate_product

    def toMetric(self, result):
        total = result.sum()
        baseline = self._baseline_bops_pr_module()
        baseline_total = baseline.sum()
        compression = _compression_rate(total, baseline_total)
        compression_pr_module = _compression_rate(result, baseline)

        metrics = {
            "structured_bops_compression": compression,
            "structured_bops_compression_rate_pr_module": _named_tensor_values(
                self.module_names,
                compression_pr_module,
            ),
        }

        if self.log_module_names:
            metrics["structured_bops_module_names"] = self.module_names

        if self.log_total_bops:
            metrics.update(
                {
                    "structured_bops": total,
                    "structured_bops_pr_module": _named_tensor_values(
                        self.module_names,
                        result,
                    ),
                    "structured_bops_baseline": baseline_total,
                    "structured_bops_baseline_pr_module": _named_tensor_values(
                        self.module_names,
                        baseline,
                    ),
                }
            )

        if self.log_compression_rate:
            metrics["structured_bops_compression_rate"] = compression

        return metrics

    def _baseline_bops_pr_module(self):
        baseline_macs = self.calc(CalcType.BASELINE_MACS_PR_MODULE)()
        return baseline_macs * (32 * 32)


def _compression_rate(active: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
    denominator = torch.where(
        baseline.ne(0),
        baseline,
        torch.ones_like(baseline),
    )
    rate = 1.0 - active / denominator
    return torch.where(
        baseline.ne(0),
        rate,
        torch.zeros_like(baseline),
    )


def _named_tensor_values(
    module_names: Iterable[str],
    values: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {name: value for name, value in zip(module_names, values, strict=True)}
