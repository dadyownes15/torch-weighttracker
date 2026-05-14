from collections.abc import Iterable

from torch_structracker.calculations import CalcType, CalculationContext
from torch_structracker.consumer_ignore import (
    IgnoreItem,
    ModuleIgnore,
    without_ignored_canonical_members,
)
from torch_structracker.trackers.base import BaseTracker


class StructuredBOPs(BaseTracker):
    required_calculations = (
        CalcType.ACTIVE_MACS_PR_MODULE,
        CalcType.BITRATE_PR_MODULE,
    )

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

    def compute(self):
        active_macs = self.calc(CalcType.ACTIVE_MACS_PR_MODULE)()
        bitrates = self.calc(CalcType.BITRATE_PR_MODULE)()
        bitrate_product = bitrates.view(-1, 2).prod(dim=1)
        return active_macs * bitrate_product

    def toMetric(self, result):
        return {
            "structured_bops": result.sum(),
            "structured_bops_pr_module": result,
        }
