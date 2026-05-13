from torch_structracker.calculations import CalcType
from torch_structracker.trackers.base import BaseTracker


class StructuredBOPs(BaseTracker):
    required_calculations = (
        CalcType.UNIT_ACTIVE_MASK,
        CalcType.UNITS_TO_MODULE_AXIS,
        CalcType.BITRATE_PR_MODULE,
    )

    def compute(self):
        unit_active_mask = self.calc(CalcType.UNIT_ACTIVE_MASK)()
        module_axis = self.calc(CalcType.UNITS_TO_MODULE_AXIS)(unit_active_mask)
        bitrates = self.calc(CalcType.BITRATE_PR_MODULE)()
        return (module_axis * bitrates).view(-1, 2).prod(dim=1)

    def toMetric(self, result):
        return {
            "structured_bops": result.sum(),
            "structured_bops_pr_module": result,
        }
