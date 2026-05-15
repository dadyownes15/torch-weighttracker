from collections.abc import Iterable

import torch

from torch_weighttracker.calculations import CalcType, CalculationContext
from torch_weighttracker.consumer_ignore import (
    IgnoreItem,
    ModuleIgnore,
    without_ignored_canonical_members,
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
        _module_names: Iterable[str] = (),
    ) -> None:
        super().__init__(calculations=calculations)
        self.log_module_names = log_module_names
        self.log_compression_rate = log_compression_rate
        self.module_names = tuple(_module_names)

    @classmethod
    def calculation_context(
        cls,
        owner,
        *,
        ignore: Iterable[IgnoreItem] = (),
        **kwargs,
    ) -> CalculationContext | None:
        ignored = ModuleIgnore(ignore)
        if not ignored:
            return None

        weighted_modules = tuple(
            module
            for module in owner._get_weighted_modules()
            if not ignored.matches(module)
        )
        return owner._calculation_context(
            canonical_groups=without_ignored_canonical_members(
                owner.canonical_groups,
                ignored,
            ),
            weighted_modules=weighted_modules,
        )

    @classmethod
    def constructor_kwargs(
        cls,
        owner,
        *,
        context: CalculationContext | None = None,
        **kwargs,
    ) -> dict:
        if not kwargs.get("log_module_names", False):
            return kwargs

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
        metrics = {
            "structured_bops": total,
            "structured_bops_pr_module": result,
        }

        if self.log_module_names:
            metrics["structured_bops_module_names"] = self.module_names

        if self.log_compression_rate:
            baseline = self._baseline_bops_pr_module()
            baseline_total = baseline.sum()
            metrics["structured_bops_baseline"] = baseline_total
            metrics["structured_bops_baseline_pr_module"] = baseline
            metrics["structured_bops_compression_rate"] = _compression_rate(
                total,
                baseline_total,
            )

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
