from torch_structracker.calculations import CalcType
from torch_structracker.trackers.base import BaseTracker


class StructuredBOPs(BaseTracker):
    required_calculations = (
        CalcType.ACTIVE_MACS_PR_MODULE,
        CalcType.BITRATE_PR_MODULE,
    )
    
    def required_calculations(**kwargs):
        base_calcs = [
                CalcType.BITRATE_PR_MODULE,
        ]
        if 
    

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
