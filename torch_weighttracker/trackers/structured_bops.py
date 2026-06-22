from collections.abc import Iterable

import torch

from torch_weighttracker.calculations import CalcType, CalculationContext
from torch_weighttracker.consumer_ignore import (
    FilterItem,
    filter_canonical_members,
)
from torch_weighttracker.trackers.base import BaseTracker
from torch_weighttracker.trackers.bops_filter import (
    bops_consumer_filter,
    filter_bops_weighted_modules,
)


class StructuredBOPs(BaseTracker):
    metric_namespace = "structured_bops"
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
        log_layerwise_stats: bool = False,
        convert_tensors: bool = True,
        wandb_format: bool = False,
        normalization_macs_pr_module=None,
        _module_names: Iterable[str] = (),
    ) -> None:
        super().__init__(
            calculations=calculations,
            convert_tensors=convert_tensors,
            wandb_format=wandb_format,
        )
        self.log_module_names = log_module_names
        self.log_compression_rate = log_compression_rate
        self.log_total_bops = log_total_bops
        self.log_layerwise_stats = log_layerwise_stats
        self.normalization_macs_pr_module = normalization_macs_pr_module
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
            canonical_groups=filter_canonical_members(
                owner.canonical_groups,
                filters,
            ),
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
            "compression": compression,
        }

        if self.log_module_names:
            metrics["module_names"] = self.module_names

        if self.log_layerwise_stats:
            _add_module_metric(
                metrics,
                self.module_names,
                "compression_rate",
                compression_pr_module,
            )

        if self.log_total_bops:
            metrics.update(
                {
                    "bops": total,
                    "baseline": baseline_total,
                }
            )
            if self.log_layerwise_stats:
                _add_module_metric(
                    metrics,
                    self.module_names,
                    "bops",
                    result,
                )
                _add_module_metric(
                    metrics,
                    self.module_names,
                    "baseline",
                    baseline,
                )

        if self.log_compression_rate:
            metrics["compression_rate"] = compression

        return metrics

    def _baseline_bops_pr_module(self):
        baseline_macs = self.calc(CalcType.BASELINE_MACS_PR_MODULE)()
        if self.normalization_macs_pr_module is not None:
            baseline_macs = _normalization_macs_pr_module(
                self.normalization_macs_pr_module,
                baseline_macs,
            )
        return baseline_macs * (32 * 32)


def _normalization_macs_pr_module(value, reference: torch.Tensor) -> torch.Tensor:
    try:
        normalization = torch.as_tensor(
            value,
            dtype=reference.dtype,
            device=reference.device,
        )
    except (TypeError, ValueError) as error:
        raise TypeError(
            "normalization_macs_pr_module must be a 1D tensor-like raw MAC vector."
        ) from error

    if normalization.ndim != 1:
        raise ValueError(
            "normalization_macs_pr_module must be a 1D tensor-like raw MAC "
            f"vector; got shape {tuple(normalization.shape)}."
        )

    expected = int(reference.numel())
    actual = int(normalization.numel())
    if actual != expected:
        raise ValueError(
            "normalization_macs_pr_module must provide one value per weighted "
            f"module. Expected {expected}; got {actual}."
        )

    return normalization


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


def _add_module_metric(
    metrics: dict,
    module_names: Iterable[str],
    key: str,
    values: torch.Tensor,
) -> None:
    modules = metrics.setdefault("modules", {})
    for name, value in zip(module_names, values, strict=True):
        modules.setdefault(name, {})[key] = value
