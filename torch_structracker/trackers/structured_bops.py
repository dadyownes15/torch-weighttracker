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
        print("active units mask", unit_active_mask)
        module_axis = self.calc(CalcType.UNITS_TO_MODULE_AXIS)(unit_active_mask)
        print("active_unit_pr_Axis", module_axis)
        bitrates = self.calc(CalcType.BITRATE_PR_MODULE)()
        print("module bitrate, ", bitrates)
        return (module_axis * bitrates).view(-1, 2).prod(dim=1)

    def toMetric(self, result):
        return {
            "structured_bops": result.sum(),
            "structured_bops_pr_module": result,
        }
