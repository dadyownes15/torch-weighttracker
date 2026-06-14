from __future__ import annotations

from collections.abc import Iterable

import torch

from torch_weighttracker.calculations import CalcType, CalculationContext
from torch_weighttracker.consumer_ignore import (
    FilterItem,
)
from torch_weighttracker.trackers.base import BaseTracker
from torch_weighttracker.trackers.bops_filter import (
    bops_consumer_filter,
    filter_bops_weighted_modules,
)
from torch_weighttracker.trackers.structured_bops import (
    _compression_rate,
    _named_tensor_values,
)
from torch_weighttracker.trackers.unstructured_sparsity import _safe_fraction


class UnstructuredBOPs(BaseTracker):
    required_calculations = (
        CalcType.UNSTRUCTURED_SPARSITY_PR_MODULE,
        CalcType.BASELINE_MACS_PR_MODULE,
        CalcType.BITRATE_PR_MODULE,
    )

    def __init__(
        self,
        calculations=None,
        *,
        log_module_names: bool = False,
        log_compression_rate: bool = False,
        log_total_bops: bool = False,
        log_layerwise_stats: bool = False,
        _module_names: Iterable[str] = (),
    ) -> None:
        super().__init__(calculations=calculations)
        self.log_module_names = log_module_names
        self.log_compression_rate = log_compression_rate
        self.log_total_bops = log_total_bops
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
        filters = bops_consumer_filter(include=include, ignore=ignore)

        return owner._calculation_context(
            weighted_modules=filter_bops_weighted_modules(
                owner._get_weighted_modules(),
                filters,
            ),
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
        counts = self.calc(CalcType.UNSTRUCTURED_SPARSITY_PR_MODULE)().view(-1, 2)
        zero_counts = counts[:, 0]
        total_counts = counts[:, 1]
        active_fraction = 1.0 - _safe_fraction(zero_counts, total_counts)

        baseline_macs = self.calc(CalcType.BASELINE_MACS_PR_MODULE)()
        bitrates = self.calc(CalcType.BITRATE_PR_MODULE)()
        bitrate_product = bitrates.view(-1, 2).prod(dim=1)
        return baseline_macs * active_fraction * bitrate_product

    def toMetric(self, result: torch.Tensor):
        total = result.sum()
        baseline = self._baseline_bops_pr_module()
        baseline_total = baseline.sum()
        compression = _compression_rate(total, baseline_total)
        compression_pr_module = _compression_rate(result, baseline)

        metrics = {
            "unstructured_bops_compression": compression,
        }

        if self.log_module_names:
            metrics["unstructured_bops_module_names"] = self.module_names

        if self.log_layerwise_stats:
            metrics["unstructured_bops_compression_rate_pr_module"] = (
                _named_tensor_values(
                    self.module_names,
                    compression_pr_module,
                )
            )

        if self.log_total_bops:
            metrics.update(
                {
                    "unstructured_bops": total,
                    "unstructured_bops_baseline": baseline_total,
                }
            )
            if self.log_layerwise_stats:
                metrics.update(
                    {
                        "unstructured_bops_pr_module": _named_tensor_values(
                            self.module_names,
                            result,
                        ),
                        "unstructured_bops_baseline_pr_module": _named_tensor_values(
                            self.module_names,
                            baseline,
                        ),
                    }
                )

        if self.log_compression_rate:
            metrics["unstructured_bops_compression_rate"] = compression

        return metrics

    def _baseline_bops_pr_module(self):
        baseline_macs = self.calc(CalcType.BASELINE_MACS_PR_MODULE)()
        return baseline_macs * (32 * 32)
